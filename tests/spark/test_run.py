from pyspark.sql import SparkSession
from pandas_profiling import ProfileReport
from py4j.java_gateway import java_import
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame


def test_generate_report():
    spark_master = "local[*]"
    spark = SparkSession.builder \
        .master(spark_master) \
        .appName("Profiler") \
        .config('spark.jars', "/tmp/flagler.jar") \
        .config('spark.sql.catalogImplementation', 'hive') \
        .config('spark.scheduler.mode', 'FAIR') \
        .getOrCreate()

    sc = spark.sparkContext
    java_import(sc._gateway.jvm, "com.healthverity.flagler.SparkProcessor")
    df = spark.read.option("header", True) \
        .option("inferSchema", "true") \
        .csv("tests/spark/sample_2.csv")
    table_name = 'test_table'
    df.createOrReplaceTempView(table_name)
    profile = ProfileReport(df, title='Profile Report: ' + table_name)
    profile.to_file(output_file=f'/tmp/{table_name}_pandas_profiling.html')

    assert True

