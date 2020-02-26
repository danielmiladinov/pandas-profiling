import numpy as np
import base64
import pandas_profiling.report.formatters as formatter
import pandas_profiling.report.numeric_helpers as helpers
from matplotlib import pyplot as plt
from pyspark.sql.functions import (abs as df_abs, col, count,
                                   max as df_max, mean, min as df_min,
                                   sum as df_sum, when, variance, stddev, kurtosis, skewness
                                   )

try:
    from urllib import quote
except ImportError:
    from urllib.parse import quote

try:
    from StringIO import BytesIO
except ImportError:
    from io import BytesIO


class FloatColumn(object):
    """
    Float Column
    """

    def __init__(self, df, column, table_stats, num_rows, bins):
        self.df = df
        self.column = column
        self.num_rows = num_rows
        self.table_stats = table_stats
        self.bins = bins
        self.column_stats = {}

    def do_analysis(self):
        stats_df = self.df.select(self.column)\
            .na.drop().agg(mean(col(self.column)).alias("mean"),
                           df_min(col(self.column)).alias("min"),
                           df_max(col(self.column)).alias("max"),
                           variance(col(self.column)).alias("variance"),
                           kurtosis(col(self.column)).alias("kurtosis"),
                           stddev(col(self.column)).alias("std"),
                           skewness(col(self.column)).alias("skewness"),
                           df_sum(col(self.column)).alias("sum")).toPandas()

        for x in np.array([0.05, 0.25, 0.5, 0.75, 0.95]):
            stats_df[formatter.fmt_float(x)] = (self.df.select(self.column)
                                                    .na.drop()
                                                    .selectExpr(
                "percentile_approx({col},CAST({n} AS DOUBLE))"
                .format(col=self.column, n=x)).toPandas().iloc[:, 0])

        self.column_stats = stats_df.iloc[0].copy()
        self.column_stats.name = self.column
        self.column_stats["range"] = self.column_stats["max"] - self.column_stats["min"]
        self.column_stats["iqr"] = self.column_stats[formatter.fmt_float(0.75)] - self.column_stats[
            formatter.fmt_float(0.25)]
        self.column_stats["cv"] = self.column_stats["std"] / float(self.column_stats["mean"])

        self.column_stats["mad"] = (self.df.select(self.column)
                                    .na.drop()
                                    .select(
            df_abs(col(self.column) - self.column_stats["mean"]).alias("delta"))
                                    .agg(df_sum(col("delta"))).toPandas().iloc[0, 0] / float(
            self.table_stats["count"]))

        self.column_stats["type"] = "NUM"
        self.column_stats['n_zeros'] = self.df.select(self.column).where(
            col(self.column) == 0.0).count()
        self.column_stats['p_zeros'] = self.column_stats['n_zeros'] / float(self.num_rows)

        # Large histogram
        img_data = BytesIO()
        hist_data = helpers.create_hist_data(self.df,
                                             self.column,
                                             self.column_stats["min"],
                                             self.column_stats["max"],
                                             self.bins)

        figure = plt.figure(figsize=(6, 4))
        plot = plt.subplot()
        plt.bar(hist_data["left_edge"],
                hist_data["count"],
                width=hist_data["width"],
                facecolor='#337ab7')
        plot.set_ylabel("Frequency")
        plot.figure.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1, wspace=0, hspace=0)
        plot.figure.savefig(img_data)
        img_data.seek(0)
        self.column_stats['histogram'] = 'data:image/png;base64,' + quote(
            base64.b64encode(img_data.getvalue()))

        # TODO Think about writing this to disk instead of caching them in strings
        plt.close(plot.figure)

        self.column_stats['mini_histogram'] = helpers.mini_histogram(hist_data)

    def get_stats(self):
        return self.column_stats
