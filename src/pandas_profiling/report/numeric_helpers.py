import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64

from pyspark.sql.functions import col, count, mean, when

try:
    from StringIO import BytesIO
except ImportError:
    from io import BytesIO

try:
    from urllib import quote
except ImportError:
    from urllib.parse import quote


def create_hist_data(df, column, minim, maxim, bins=10):

    def create_all_conditions(current_col, column, left_edges, count=1):
        """
        Recursive function that exploits the
        ability to call the Spark SQL Column method
        .when() in a recursive way.
        """
        left_edges = left_edges[:]
        if len(left_edges) == 0:
            return current_col
        if len(left_edges) == 1:
            next_col = current_col.when(col(column) >= float(left_edges[0]), count)
            left_edges.pop(0)
            return create_all_conditions(next_col, column, left_edges[:], count+1)
        next_col = current_col.when((float(left_edges[0]) <= col(column))
                                    & (col(column) < float(left_edges[1])), count)
        left_edges.pop(0)
        return create_all_conditions(next_col, column, left_edges[:], count+1)

    num_range = maxim - minim
    bin_width = num_range / float(bins)
    left_edges = [minim]
    for _bin in range(bins):
        left_edges = left_edges + [left_edges[-1] + bin_width]
    left_edges.pop()
    expression_col = when((float(left_edges[0]) <= col(column))
                          & (col(column) < float(left_edges[1])), 0)
    left_edges_copy = left_edges[:]
    left_edges_copy.pop(0)
    bin_data = (df.select(col(column))
                .na.drop()
                .select(col(column),
                        create_all_conditions(expression_col,
                                              column,
                                              left_edges_copy
                                             ).alias("bin_id")
                       )
                .groupBy("bin_id").count()
               ).toPandas()

    # If no data goes into one bin, it won't
    # appear in bin_data; so we should fill
    # in the blanks:
    bin_data.index = bin_data["bin_id"]
    new_index = list(range(bins))
    bin_data = bin_data.reindex(new_index)
    bin_data["bin_id"] = bin_data.index
    bin_data = bin_data.fillna(0)

    # We add the left edges and bin width:
    bin_data["left_edge"] = left_edges
    bin_data["width"] = bin_width

    return bin_data


def mini_histogram(histogram_data):
    # Small histogram
    imgdata = BytesIO()
    hist_data = histogram_data
    figure = plt.figure(figsize=(2, 0.75))
    plot = plt.subplot()
    plt.bar(hist_data["left_edge"],
            hist_data["count"],
            width=hist_data["width"],
            facecolor='#337ab7')
    plot.axes.get_yaxis().set_visible(False)
    if hasattr(plot, 'set_facecolor'):
        plot.set_facecolor("w")
    else:
        plot.set_axis_bgcolor("w")

    xticks = plot.xaxis.get_major_ticks()
    for tick in xticks[1:-1]:
        tick.set_visible(False)
        tick.label.set_visible(False)
    for tick in (xticks[0], xticks[-1]):
        tick.label.set_fontsize(8)
    plot.figure.subplots_adjust(left=0.15, right=0.85, top=1, bottom=0.35, wspace=0, hspace=0)
    plot.figure.savefig(imgdata)
    imgdata.seek(0)
    result_string = 'data:image/png;base64,' + quote(base64.b64encode(imgdata.getvalue()))
    plt.close(plot.figure)
    return result_string
