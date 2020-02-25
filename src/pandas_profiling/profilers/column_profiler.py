import pandas as pd
from pyspark.sql.functions import col, count, countDistinct, trim, udf
from pandas_profiling.profilers.column_types.constant_column import ConstantColumn
from pandas_profiling.profilers.column_types.integer_column import IntegerColumn
from pandas_profiling.profilers.column_types.float_column import FloatColumn
from pandas_profiling.profilers.column_types.date_column import DateColumn
from pandas_profiling.profilers.column_types.unique_column import UniqueColumn
from pandas_profiling.profilers.column_types.categorical_column import CategoricalColumn


class ColumnProfiler(object):
    """
    Column Describer Object
    """
    def __init__(self, df, column, num_rows, column_description, bins):
        self.df = df
        self.column = column
        self.num_rows = num_rows
        self.column_description = column_description
        self.bins = bins
        self.stats = {}
        self.column_type = self.df.select(self.column).dtypes[0][1]
        # TODO: think about implementing analysis for complex data types:
        if ("array" in self.column_type) or ("stuct" in self.column_type) or \
                ("map" in self.column_type):
            raise NotImplementedError(
                "Column {c} is of type {t} and cannot be analyzed".format(c=self.column,
                                                                          t=self.column_type))

    def do_analysis(self):
        self.calculate_column_type_agnostic_stats()
        self.calculate_column_type_specific_stats()
        self.generate_mode_stats()

    def calculate_column_type_agnostic_stats(self):
        assign_value_udf = udf(lambda value: None if not value else value)

        trimmed_data = self.df.withColumn("trimmed", trim(col(self.column)))

        trimmed_converted_data =\
                trimmed_data.withColumn("cleaned_trimmed", assign_value_udf(col("trimmed")))

        distinct_count = trimmed_converted_data.select('cleaned_trimmed').agg(
            countDistinct(col('cleaned_trimmed')).alias("distinct_count")).toPandas()

        non_nan_count = trimmed_converted_data.select('cleaned_trimmed').na.drop().select(
            count(col('cleaned_trimmed')).alias("count")).toPandas()

        results_data = pd.concat([distinct_count, non_nan_count], axis=1)

        results_data["p_unique"] = results_data["distinct_count"] / float(results_data["count"])
        results_data["is_unique"] = results_data["distinct_count"] == self.num_rows
        results_data["n_missing"] = self.num_rows - results_data["count"]
        results_data["p_missing"] = results_data["n_missing"] / float(self.num_rows)
        results_data["fill_rate"] = float(1) - results_data["p_missing"]
        results_data["p_infinite"] = 0
        results_data["n_infinite"] = 0
        self.stats = results_data.ix[0].copy()
        self.stats["memorysize"] = 0
        if self.stats["n_missing"] > 0:
            self.stats["distinct_count"] += 1
        self.stats.name = self.column

    def calculate_column_type_specific_stats(self):
        if self.stats["distinct_count"] == 1:
            constant_column = ConstantColumn(self.df, self.column)
            constant_column.do_analysis()
            self.stats = self.stats.append(constant_column.get_stats())
        elif self.column_type in {"tinyint", "smallint", "int", "bigint"}:
            int_column = IntegerColumn(self.df, self.column, self.stats, self.num_rows, self.bins)
            int_column.do_analysis()
            self.stats = self.stats.append(int_column.get_stats())
        elif self.column_type in {"float", "double", "decimal"}:
            float_column = FloatColumn(self.df, self.column, self.stats, self.num_rows, self.bins)
            float_column.do_analysis()
            self.stats = self.stats.append(float_column.get_stats())
        elif self.column_type in {"date", "timestamp"}:
            date_column = DateColumn(self.df, self.column, self.stats, self.num_rows, self.bins)
            date_column.do_analysis()
            self.stats = self.stats.append(date_column.get_stats())
        elif self.stats["is_unique"]:
            unique_column = UniqueColumn(self.df, self.column, self.stats, self.num_rows, self.bins)
            unique_column.do_analysis()
            self.stats = self.stats.append(unique_column.get_stats())
        else:
            categorical_column = CategoricalColumn(self.df, self.column, self.stats, self.num_rows,
                                                   self.bins)
            categorical_column.do_analysis()
            self.stats = self.stats.append(categorical_column.get_stats())

    def generate_mode_stats(self):
        # TODO: check whether it is worth it to implement the "real" mode:
        if self.stats["count"] > self.stats["distinct_count"] > 1:
            try:
                self.stats["mode"] = self.stats["top"]
            except KeyError:
                self.stats["mode"] = 0
        else:
            try:
                self.stats["mode"] = self.stats["value_counts"].index[0]
            except KeyError:
                self.stats["mode"] = 0
            # If and IndexError happens,
            # it is because all columns are NULLs:
            except IndexError:
                self.stats["mode"] = "EMPTY"

        self.column_description[self.column] = self.stats

    def get_stats(self):
        return self.stats

