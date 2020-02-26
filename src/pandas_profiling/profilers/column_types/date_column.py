import numpy as np
import pandas as pd
import base64
import pandas_profiling.report.formatters as formatter
import pandas_profiling.report.numeric_helpers as helpers
from matplotlib import pyplot as plt
from pyspark.sql.functions import (col, count, concat_ws, year, month, mean,
                                   variance, stddev, kurtosis, skewness, min as df_min,
                                   max as df_max
                                   )

try:
    from urllib import quote
except ImportError:
    from urllib.parse import quote

try:
    from StringIO import BytesIO
except ImportError:
    from io import BytesIO


class DateColumn(object):
    """
    Date Column
    """
    def __init__(self, df, column, table_stats, num_rows, bins):
        self.df = df
        self.column = column
        self.num_rows = num_rows
        self.table_stats = table_stats
        self.bins = bins
        self.column_stats = {}

    def do_analysis(self):
        day_counts = (
            self.df.select(self.column)
            .na.drop()
            .groupBy(self.column)
            .agg(count(col(self.column)))
            .orderBy("count({c})".format(c=self.column), ascending=False)).toPandas()

        month_counts = (
            self.df.select(concat_ws('-', year(self.column), month(self.column)).alias('month'))
            .na.drop()
            .groupBy('month')
            .agg(count(col('month')))
            .orderBy("count({c})".format(c='month'), ascending=False)).toPandas()

        day_counts.index += 1
        month_counts.index += 1

        sample_count = 20

        top_days_counts = day_counts.iloc[:sample_count, :]
        top_days_counts.columns = ['top', 'freq']
        top_days_counts['top'] = pd.to_datetime(top_days_counts['top'], errors='coerce') \
            .dt.date

        top_month_counts = month_counts.iloc[:sample_count, :]
        top_month_counts.columns = ['top', 'freq']
        top_month_counts['top'] = pd.to_datetime(top_month_counts['top'], errors='coerce') \
            .dt.date

        other_days = pd.Series(len(day_counts.iloc[sample_count:, 0]),
                               index=["***Other Values***"])
        other_months = pd.Series(len(month_counts.iloc[sample_count:, 0]),
                                 index=["***Other Values***"])
        other_distinct_days = pd.Series(day_counts.iloc[sample_count:, 0].nunique(dropna=True),
                                        index=["***Other Values Distinct Count***"])
        other_distinct_months = pd.Series(month_counts.iloc[sample_count:, 0].nunique(dropna=True),
                                          index=["***Other Values Distinct Count***"])

        top_days = top_days_counts.set_index('top')['freq']
        top_days = top_days.append(other_days)
        top_days = top_days.append(other_distinct_days)

        top_months = top_month_counts.set_index('top')['freq']
        top_months = top_months.append(other_months)
        top_months = top_months.append(other_distinct_months)

        stats_df = self.df.select(self.column).na.drop() \
            .agg(df_min(col(self.column)).alias("min"),
                 df_max(col(self.column)).alias("max")
                 ).toPandas()

        self.column_stats = stats_df.iloc[0].copy()
        self.column_stats["max_value_counts"] = top_days
        self.column_stats["min_value_counts"] = top_months
        self.column_stats["type"] = "DATE"
        self.column_stats.name = self.column

    def get_stats(self):
        return self.column_stats

