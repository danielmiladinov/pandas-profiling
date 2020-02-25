import pandas as pd
from pyspark.sql.functions import col, count, mean, variance, stddev, kurtosis, skewness

try:
    from urllib import quote
except ImportError:
    from urllib.parse import quote

try:
    from StringIO import BytesIO
except ImportError:
    from io import BytesIO


class CategoricalColumn(object):
    """
    Categorical Column
    """

    def __init__(self, df, column, table_stats, num_rows, bins):
        self.df = df
        self.column = column
        self.num_rows = num_rows
        self.table_stats = table_stats
        self.bins = bins
        self.column_stats = {}

    def do_analysis(self):
        value_counts = (self.df.select(self.column).na.drop()
                        .groupBy(self.column)
                        .agg(count(col(self.column)))
                        .orderBy("count({c})".format(c=self.column), ascending=False)
                        ).cache()

        bottom_value_counts = (value_counts
                               .orderBy("count({c})".format(c=self.column), ascending=True)
                               ).cache()

        # Get the most frequent class:
        self.column_stats = (value_counts
                 .limit(1)
                 .withColumnRenamed(self.column, "top")
                 .withColumnRenamed("count({c})".format(c=self.column), "freq")
                 ).toPandas().ix[0]

        # Get the top 50 classes by value count,
        # and put the rest of them grouped at the
        # end of the Series:
        # Then do the same for bottom 50
        top_50 = value_counts.limit(50).toPandas().sort_values("count({c})".format(c=self.column),
                                                               ascending=False)
        top_50_categories = top_50[self.column].values.tolist()

        bottom_50 = bottom_value_counts.limit(50).toPandas().sort_values(
            "count({c})".format(c=self.column),
            ascending=True)
        bottom_50_categories = bottom_50[self.column].values.tolist()

        top_others_count = pd.Series([self.df.select(self.column).na.drop()
                                     .where(~(col(self.column).isin(*top_50_categories)))
                                     .count()
                                      ], index=["***Other Values***"])
        top_others_distinct_count = pd.Series([value_counts
                                              .where(~(col(self.column).isin(*top_50_categories)))
                                              .count()
                                               ], index=["***Other Values Distinct Count***"])

        bottom_others_count = pd.Series([self.df.select(self.column).na.drop()
                                        .where(~(col(self.column).isin(*bottom_50_categories)))
                                        .count()
                                         ], index=["***Other Values***"])
        bottom_others_distinct_count = pd.Series([bottom_value_counts
                                                 .where(~(col(self.column).isin(*bottom_50_categories)))
                                                 .count()
                                                  ], index=["***Other Values Distinct Count***"])

        top = top_50.set_index(self.column)["count({c})".format(c=self.column)]
        top = top.append(top_others_count)
        top = top.append(top_others_distinct_count)
        self.column_stats["max_value_counts"] = top

        bottom = bottom_50.set_index(self.column)["count({c})".format(c=self.column)]
        bottom = bottom.append(bottom_others_count)
        bottom = bottom.append(bottom_others_distinct_count)
        self.column_stats["min_value_counts"] = bottom

        self.column_stats["type"] = "CAT"
        value_counts.unpersist()

    def get_stats(self):
        return self.column_stats
