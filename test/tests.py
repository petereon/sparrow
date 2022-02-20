from sparrow import SparrowDataFrame, SparrowRDD, SparrowPipelinedRDD, Filter, Flatten
import pytest


@pytest.fixture(scope="module")
def provide_spark_session():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


@pytest.fixture(scope="module")
def provide_spark_dataframe(provide_spark_session):

    return provide_spark_session.createDataFrame(
        data=[
            (1, 2.0, ["a", "b", "c"]),
            (2, 3.0, ["b", "c", "d"]),
            (3, 4.0, ["c", "d", "e"]),
            (4, 5.0, ["d", "e", "f"]),
            (5, 6.0, ["e", "f", "g"]),
        ],
        schema=["id", "v1", "v2"],
    )


@pytest.fixture(scope="module")
def provide_spark_rdd(provide_spark_session):
    return provide_spark_session.sparkContext.parallelize(
        [
            (1, 2.0, ["a", "b", "c"]),
            (2, 3.0, ["b", "c", "d"]),
            (3, 4.0, ["c", "d", "e"]),
            (4, 5.0, ["d", "e", "f"]),
            (5, 6.0, ["e", "f", "g"]),
        ]
    )


def test_spark_df_to_sparrow_df(provide_spark_dataframe):
    df = provide_spark_dataframe
    assert isinstance(SparrowDataFrame(df), SparrowDataFrame)


def test_spark_rdd_to_sparrow_rdd(provide_spark_rdd):
    rdd = provide_spark_rdd
    assert isinstance(SparrowRDD(rdd), SparrowRDD)


def test_spark_pipelined_rdd_to_sparrow_pipelined_rdd(provide_spark_rdd):
    rdd = provide_spark_rdd
    assert isinstance(SparrowPipelinedRDD(rdd), SparrowPipelinedRDD)


def test_sparrow_map_fn_df(provide_spark_dataframe):
    df = provide_spark_dataframe
    df = SparrowDataFrame(df)

    def just_first_col(x):
        return x[0]

    res = df >> just_first_col
    assert isinstance(res, SparrowPipelinedRDD)
    assert res.collect() == [1, 2, 3, 4, 5]


def test_sparrow_flatmap_fn_df(provide_spark_dataframe):
    df = provide_spark_dataframe
    df = SparrowDataFrame(df)

    def flatten_list(x):
        return x[2]

    res = df >> Flatten(flatten_list)
    assert isinstance(res, SparrowPipelinedRDD)
    assert res.collect() == [
        "a",
        "b",
        "c",
        "b",
        "c",
        "d",
        "c",
        "d",
        "e",
        "d",
        "e",
        "f",
        "e",
        "f",
        "g",
    ]


def test_sparrow_filter_fn_df(provide_spark_dataframe):
    df = provide_spark_dataframe
    df = SparrowDataFrame(df)

    def filter_by_id(x):
        return x[0] % 2 == 0

    res = df >> Filter(filter_by_id) >> (lambda x: x[0])
    assert isinstance(res, SparrowPipelinedRDD)
    assert res.collect() == [2, 4]


def test_sparrow_map_fn_rdd(provide_spark_rdd):
    rdd = provide_spark_rdd
    rdd = SparrowRDD(rdd)

    def just_first_col(x):
        return x[0]

    res = rdd >> just_first_col
    assert isinstance(res, SparrowPipelinedRDD)
    assert res.collect() == [1, 2, 3, 4, 5]


def test_spark_filter_fn_rdd(provide_spark_rdd):
    rdd = provide_spark_rdd
    rdd = SparrowRDD(rdd)

    def filter_by_id(x):
        return x[0] % 2 == 0

    res = rdd >> Filter(filter_by_id) >> (lambda x: x[0])
    assert isinstance(res, SparrowPipelinedRDD)
    assert res.collect() == [2, 4]


def test_sparrow_flatmap_fn_rdd(provide_spark_rdd):
    rdd = provide_spark_rdd
    rdd = SparrowRDD(rdd)

    def flatten_list(x):
        return x[2]

    res = rdd >> Flatten(flatten_list)
    assert isinstance(res, SparrowPipelinedRDD)
    assert res.collect() == [
        "a",
        "b",
        "c",
        "b",
        "c",
        "d",
        "c",
        "d",
        "e",
        "d",
        "e",
        "f",
        "e",
        "f",
        "g",
    ]


def test_sparrow_function_chaning(provide_spark_dataframe):
    df = provide_spark_dataframe
    df = SparrowDataFrame(df)

    def just_first_col(x):
        return x[0]

    def multiply_by_2(x):
        return x * 2

    res = df >> just_first_col >> multiply_by_2
    assert isinstance(res, SparrowPipelinedRDD)
    assert res.collect() == [2, 4, 6, 8, 10]


def test_sparrow_lambdas(provide_spark_dataframe):
    df = provide_spark_dataframe
    df = SparrowDataFrame(df)

    res = df >> (lambda x: x[0]) >> (lambda x: x * 2)
    assert isinstance(res, SparrowPipelinedRDD)
    assert res.collect() == [2, 4, 6, 8, 10]


def test_longer_chains(provide_spark_dataframe):
    df = provide_spark_dataframe
    df = SparrowDataFrame(df)

    def just_first_col(x):
        return x[0]

    def multiply_by_2(x):
        return x * 2

    res = (
        df
        >> just_first_col
        >> multiply_by_2
        >> Filter(lambda x: x % 4 == 0)
        >> Flatten(lambda x: [x, 2 * x])
    )
    assert isinstance(res, SparrowPipelinedRDD)
    assert res.collect() == [4, 8, 8, 16]
