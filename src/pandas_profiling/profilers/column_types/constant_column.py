import pandas as pd

class ConstantColumn(object):
    """
    Constant Column
    """
    def __init__(self, df, column):
        self.df = df
        self.column = column
        self.column_stats = {}

    def do_analysis(self):
        self.column_stats = pd.Series(['CONST'], index=['type'], name=self.column)
        self.column_stats["value_counts"] = (self.df.select(self.column)
                                             .na.drop().limit(1)).toPandas().ix[:, 0].value_counts()

    def get_stats(self):
        return self.column_stats
