import pandas as pd

try:
    from urllib import quote
except ImportError:
    from urllib.parse import quote

try:
    from StringIO import BytesIO
except ImportError:
    from io import BytesIO


class UniqueColumn(object):
    """
    Unique Column
    """
    def __init__(self, df, column, table_stats, num_rows, bins):
        self.df = df
        self.column = column
        self.num_rows = num_rows
        self.table_stats = table_stats
        self.column_stats = {}

    def do_analysis(self):
        self.column_stats = pd.Series(['UNIQUE'], index=['type'], name=self.column)
        self.column_stats["value_counts"] = (self.df.select(self.column)
                                             .na.drop()
                                             .limit(50)).toPandas().iloc[:, 0].value_counts()

        pd_df = self.df.select(self.column).toPandas()
        self.column_stats["max_value_counts"] = pd_df.iloc[:50]
        self.column_stats["min_value_counts"] = pd_df.tail(50)
        self.column_stats["type"] = "UNIQUE"

    def get_stats(self):
        return self.column_stats
