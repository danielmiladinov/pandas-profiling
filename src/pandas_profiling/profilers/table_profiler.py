import matplotlib
import pandas_profiling.report.formatters as formatters
import pandas as pd
import numpy as np
from pandas_profiling.profilers.column_profiler import ColumnProfiler
from pyspark.sql import DataFrame as SparkDataFrame
from pkg_resources import resource_filename
from threading import Thread
from itertools import product
from pandas_profiling.model.base import Variable
from pyspark.sql import functions as f
from pandas_profiling import __version__
from pandas_profiling.config import config as config

NUMERIC_FIELDS = {"tinyint", "smallint", "int", "bigint", "float", "double", "decimal"}


class TableProfiler(object):
    """
    Table profiler object
    """
    def __init__(self, df, bins, corr_reject):
        if not isinstance(df, SparkDataFrame):
            raise TypeError("df must be of type pyspark.sql.DataFrame")
        self.df = df
        self.bins = bins
        self.corr_reject = corr_reject
        self.table_stats = {}
        self.column_description = {}
        self.output_stats = {}

    def do_analysis(self):
        self.setup_matplotlib()
        self.get_num_rows()
        self.calculate_individual_column_stats()
        self.calculate_correlation_matrix()
        self.calculate_table_stats()

    def get_table_stats(self):
        return self.table_stats

    @staticmethod
    def setup_matplotlib():
        try:
            matplotlib.style.use("default")
        except:
            pass
        matplotlib.style.use(resource_filename(__name__, "../spark_df_profiling.mplstyle"))

    def get_num_rows(self):
        # Number of rows:
        self.table_stats = {"n": self.df.count()}
        if self.table_stats["n"] == 0:
            raise ValueError("df cannot be empty")

    def calculate_individual_column_stats(self):
        def analyze_column(col_profiler):
            col_profiler.do_analysis()

        batch_size = 10
        i = 0
        col_description = {}
        while i < len(self.df.columns):
            threads = []
            batch_columns = self.df.columns[i:i + batch_size]
            numeric_columns = [column for column in batch_columns if self.df.select(column).dtypes[0][1] in NUMERIC_FIELDS]
            non_numeric_columns = [column for column in batch_columns if self.df.select(column).dtypes[0][1] not in NUMERIC_FIELDS]

            # Thread non-numeric columns
            for column in non_numeric_columns:
                column_profiler = ColumnProfiler(self.df, column, self.table_stats["n"],
                                                 col_description, self.bins)
                threads.append(
                    Thread(target=analyze_column, args=[column_profiler]))
                threads[-1].start()
            for t in threads:
                t.join()

            # We cannot thread numeric fields because matplotlib is not thread-safe
            for column in numeric_columns:
                column_profiler = ColumnProfiler(self.df, column, self.table_stats["n"],
                                                 col_description, self.bins)
                column_profiler.do_analysis()
            i += batch_size

        self.column_description = col_description

    def calculate_correlation_matrix(self):
        def corr_matrix(df, columns=None):
            if columns is None:
                columns = df.columns
            combinations = list(product(columns, columns))

            def separate(l, n):
                for i in range(0, len(l), n):
                    yield l[i:i + n]

            grouped = list(separate(combinations, len(columns)))
            df_cleaned = df.select(*columns).na.drop(how="any")

            for i in grouped:
                for j in enumerate(i):
                    i[j[0]] = i[j[0]] + (df_cleaned.corr(str(j[1][0]), str(j[1][1])),)

            df_pandas = pd.DataFrame(grouped).applymap(lambda x: x[2])
            df_pandas.columns = columns
            df_pandas.index = columns
            return df_pandas

        # Compute correlation matrix
        if self.corr_reject is not None:
            computable_correlation = [column for column in self.column_description
                                      if self.column_description[column]["type"] in {"NUM"}]

            if len(computable_correlation) > 0:
                corr = corr_matrix(self.df, columns=computable_correlation)
                for x, corr_x in corr.iterrows():
                    for y, corr in corr_x.iteritems():
                        if x == y:
                            break

                        if corr >= self.corr_reject:
                            self.column_description[x] = pd.Series(['CORR', y, corr],
                                                                   index=['type', 'correlation_var',
                                                                          'correlation'],
                                                                   name=x)

    def calculate_table_stats(self):
        variable_stats = pd.DataFrame(self.column_description)
        # General statistics
        self.table_stats["n_var"] = len(self.df.columns)
        self.table_stats["n_cells_missing"] = float(variable_stats.loc["n_missing"].sum())
        self.table_stats["p_cells_missing"] = self.table_stats["n_cells_missing"] / (
                self.table_stats["n"] * self.table_stats["n_var"])

        supported_columns = variable_stats.transpose()[
            variable_stats.transpose().type != Variable.S_TYPE_UNSUPPORTED
            ].index.tolist()
        duplicates = self.df.groupBy(self.df.columns)\
            .count()\
            .where(f.col('count') > 1)\
            .select(f.sum('count'))\
            .show()
        self.table_stats["n_duplicates"] = duplicates if duplicates is not None else 0
        self.table_stats["p_duplicates"] = (
            (self.table_stats["n_duplicates"] / self.table_stats["n"])
            if (len(supported_columns) > 0 and self.table_stats["n"] > 0)
            else 0
        )
        memsize = 0
        self.table_stats['memory_size'] = formatters.fmt_bytesize(memsize)
        self.table_stats['record_size'] = formatters.fmt_bytesize(memsize / self.table_stats['n'])
        self.table_stats['types'] = {}
        self.table_stats['types'].update({k: 0 for k in ("NUM", "DATE", "CONST", "CAT", "UNIQUE", "CORR")})
        self.table_stats['types'].update(dict(variable_stats.loc['type'].value_counts()))
        self.table_stats['types']['REJECTED'] = self.table_stats['types']['CONST'] + self.table_stats['types']['CORR']

        # this whole section of min_freq/max_freq probably doesn't need to happen!
        max_freqs = {}
        for var in variable_stats:
            if "max_value_counts" not in variable_stats[var]:
                pass
            elif variable_stats[var]["max_value_counts"] is not np.nan:
                max_freqs[var] = variable_stats[var]["max_value_counts"]
            else:
                pass
        try:
            variable_stats = variable_stats.drop("max_value_counts")
        except ValueError:
            pass

        min_freqs = {}
        for var in variable_stats:
            if "min_value_counts" not in variable_stats[var]:
                pass
            elif not (variable_stats[var]["min_value_counts"] is np.nan):
                min_freqs[var] = variable_stats[var]["min_value_counts"]
            else:
                pass
        try:
            variable_stats = variable_stats.drop("min_value_counts")
        except ValueError:
            pass

        self.output_stats = {
            'table': self.table_stats,
            'variables': variable_stats.T,
            'maxfreq': max_freqs,
            'minfreq': min_freqs,
            'messages': [],
            'package': {
                "pandas_profiling_version": __version__,
                "pandas_profiling_config": config.dump(),
            },
        }

    def get_output_stats(self):
        return self.output_stats
