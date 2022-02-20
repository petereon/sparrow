from typing import Union
from pyspark.rdd import PipelinedRDD, RDD
from pyspark.sql import DataFrame


class Flatten:
    def __init__(self, func: callable):
        self.func = func


class Filter:
    def __init__(self, func: callable):
        self.func = func


class SparrowPipelinedRDD(PipelinedRDD):
    def __init__(self, rdd: PipelinedRDD):
        self.__dict__ = rdd.__dict__

    def __rshift__(self, fn: Union[callable, Flatten, Filter]) -> "SparrowPipelinedRDD":
        if isinstance(fn, Flatten):
            res = self.flatMap(fn.func)
        elif isinstance(fn, Filter):
            res = self.filter(fn.func)
        else:
            res = self.map(fn)
        res.__class__ = SparrowPipelinedRDD
        return res


class SparrowDataFrame(DataFrame):
    def __init__(self, df: DataFrame):
        self.__dict__ = df.__dict__

    def __rshift__(self, fn: Union[callable, Flatten, Filter]) -> SparrowPipelinedRDD:
        if isinstance(fn, Flatten):
            res = self.rdd.flatMap(fn.func)
        elif isinstance(fn, Filter):
            res = self.rdd.filter(fn.func)
        else:
            res = self.rdd.map(fn)
        res.__class__ = SparrowPipelinedRDD
        return res


class SparrowRDD(RDD):
    def __init__(self, rdd: RDD):
        self.__dict__ = rdd.__dict__

    def __rshift__(self, fn: Union[callable, Flatten, Filter]) -> SparrowPipelinedRDD:
        if isinstance(fn, Flatten):
            res = self.flatMap(fn.func)
        elif isinstance(fn, Filter):
            res = self.filter(fn.func)
        else:
            res = self.map(fn)
        res.__class__ = SparrowPipelinedRDD
        return res
