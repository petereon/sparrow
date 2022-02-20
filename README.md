# Sparrow

Sparrow (a combination of _Spark_ and _arrow_) is a Python mini library that enhances Spark with an _arrow API_.

Intent is to make mappers and filters over RDD a bit more elegant and exciting. Author also feels that here developed API does have a more consitent feel.

Consider and example of few operations on an RDD in native PySpark
```python
...
rdd = spark.sparkContext.parallelize(
        [
            (1, 2.0, ["a", "b", "c"]),
            (2, 3.0, ["b", "c", "d"]),
            (3, 4.0, ["c", "d", "e"]),
            (4, 5.0, ["d", "e", "f"]),
            (5, 6.0, ["e", "f", "g"]),
        ]
    )
    
res = rdd.map(lambda x: x[2]).flatMap(lambda x: x).filter(lambda x: x == 'b')

```
and then on RDD extended with Sparrow:
```python
rdd = spark.sparkContext.parallelize(
        [
            (1, 2.0, ["a", "b", "c"]),
            (2, 3.0, ["b", "c", "d"]),
            (3, 4.0, ["c", "d", "e"]),
            (4, 5.0, ["d", "e", "f"]),
            (5, 6.0, ["e", "f", "g"]),
        ]
    )

res = (
    SparrowRDD(rdd) 
    >> (lambda x: x[2]) 
    >> Flatten(lambda x: x)
    >> Filter(lambda x: x == 'b')
)
```
