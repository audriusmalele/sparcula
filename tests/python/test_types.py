from datetime import date, datetime
from unittest import TestCase

from sparcula.pyspark.sql import SparkSession, functions as F, types as T

spark = SparkSession.builder.getOrCreate()


class TestTypes(TestCase):
    def test_dtypes(self):
        df = spark.createDataFrame(
            data=[],
            schema=T.StructType([
                T.StructField('n', T.NullType()),
                T.StructField('b', T.BooleanType()),
                T.StructField('y', T.ByteType()),
                T.StructField('h', T.ShortType()),
                T.StructField('i', T.IntegerType()),
                T.StructField('l', T.LongType()),
                T.StructField('f', T.FloatType()),
                T.StructField('d', T.DoubleType()),
                T.StructField('D', T.DateType()),
                T.StructField('T', T.TimestampType()),
                T.StructField('s', T.StringType()),
                T.StructField('S', T.StructType([
                    T.StructField('s', T.StringType()),
                    T.StructField('i', T.IntegerType()),
                    T.StructField('b', T.BooleanType()),
                ])),
                T.StructField('A', T.ArrayType(T.StringType())),
                T.StructField('M', T.MapType(T.StringType(), T.StringType())),
            ]),
        )
        self.assertEqual([
            ('n', 'void'),
            ('b', 'boolean'),
            ('y', 'tinyint'),
            ('h', 'smallint'),
            ('i', 'int'),
            ('l', 'bigint'),
            ('f', 'float'),
            ('d', 'double'),
            ('D', 'date'),
            ('T', 'timestamp'),
            ('s', 'string'),
            ('S', 'struct<s:string,i:int,b:boolean>'),
            ('A', 'array<string>'),
            ('M', 'map<string,string>'),
        ], df.dtypes)

        rows = df.collect()
        self.assertEqual([], rows)

    def test_type_parser(self):
        df = spark.createDataFrame([[]])
        df = df.select(
            F.lit(None).cast('void'),
            F.lit(None).cast('boolean'),
            F.lit(None).cast('tinyint'),
            F.lit(None).cast('smallint'),
            F.lit(None).cast('int'),
            F.lit(None).cast('bigint'),
            F.lit(None).cast('float'),
            F.lit(None).cast('double'),
            F.lit(None).cast('date'),
            F.lit(None).cast('timestamp'),
            F.lit(None).cast('string'),
            F.lit(None).cast('struct<a:int,b:int>'),
            F.lit(None).cast('array<int>'),
            F.lit(None).cast('map<int,int>'),
        )

        self.assertEqual(T.StructType([
            T.StructField('CAST(NULL AS VOID)', T.NullType()),
            T.StructField('CAST(NULL AS BOOLEAN)', T.BooleanType()),
            T.StructField('CAST(NULL AS TINYINT)', T.ByteType()),
            T.StructField('CAST(NULL AS SMALLINT)', T.ShortType()),
            T.StructField('CAST(NULL AS INT)', T.IntegerType()),
            T.StructField('CAST(NULL AS BIGINT)', T.LongType()),
            T.StructField('CAST(NULL AS FLOAT)', T.FloatType()),
            T.StructField('CAST(NULL AS DOUBLE)', T.DoubleType()),
            T.StructField('CAST(NULL AS DATE)', T.DateType()),
            T.StructField('CAST(NULL AS TIMESTAMP)', T.TimestampType()),
            T.StructField('CAST(NULL AS STRING)', T.StringType()),
            T.StructField('NULL', T.StructType([
                T.StructField('a', T.IntegerType()),
                T.StructField('b', T.IntegerType()),
            ])),
            T.StructField('NULL', T.ArrayType(T.IntegerType())),
            T.StructField('NULL', T.MapType(T.IntegerType(), T.IntegerType())),
        ]), df.schema)

        rows = df.collect()
        self.assertEqual([tuple(None for _ in range(len(df.columns)))], rows)

    def test_cast(self):
        df = spark.createDataFrame([[]])
        df = df.select(
            F.lit(datetime(2020, 12, 31, 23, 59, 59)).cast(T.DateType()),
            F.lit(False).cast(T.IntegerType()),
            F.lit(True).cast(T.IntegerType()),
            F.lit('1.23').cast(T.DoubleType()),
            F.lit('abc').cast(T.DoubleType()),
            F.lit('true').cast(T.BooleanType()),
            F.lit('false').cast(T.BooleanType()),
        )

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([{
            "CAST(TIMESTAMP '2020-12-31 23:59:59' AS DATE)": date(2020, 12, 31),
            'CAST(false AS INT)': 0,
            'CAST(true AS INT)': 1,
            'CAST(1.23 AS DOUBLE)': 1.23,
            'CAST(abc AS DOUBLE)': None,
            'CAST(true AS BOOLEAN)': True,
            'CAST(false AS BOOLEAN)': False,
        }], rows)

    def test_implicit_cast(self):
        df = spark.createDataFrame([[]])
        df = df.select(
            F.when(F.lit(True), F.create_map(F.lit('k'), F.lit('v')))
            .otherwise(F.create_map()),
            F.when(F.lit(True), F.array(F.lit('a'), F.lit('b')))
            .otherwise(F.array()),
            F.when(F.lit(True), F.array(F.lit(None)))
            .otherwise(F.array(F.lit('a'), F.lit('b'))),
            F.when(F.lit(True), date(2020, 12, 31)).otherwise(''),
        )

        self.assertEqual(T.StructType([
            T.StructField(
                'CASE WHEN true THEN map(k, v) ELSE map() END',
                T.MapType(T.StringType(), T.StringType(), False),
                False,
            ),
            T.StructField(
                'CASE WHEN true THEN array(a, b) ELSE array() END',
                T.ArrayType(T.StringType(), False),
                False,
            ),
            T.StructField(
                'CASE WHEN true THEN array(NULL) ELSE array(a, b) END',
                T.ArrayType(T.StringType()),
                False,
            ),
            T.StructField(
                "CASE WHEN true THEN DATE '2020-12-31' ELSE  END",
                T.StringType(),
                False,
            ),
        ]), df.schema)
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([{
            'CASE WHEN true THEN map(k, v) ELSE map() END': {'k': 'v'},
            'CASE WHEN true THEN array(a, b) ELSE array() END': ['a', 'b'],
            'CASE WHEN true THEN array(NULL) ELSE array(a, b) END': [None],
            "CASE WHEN true THEN DATE '2020-12-31' ELSE  END": '2020-12-31',
        }], rows)
