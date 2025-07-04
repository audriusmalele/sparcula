from pathlib import Path
from unittest import TestCase

from sparcula.pyspark.sql import SparkSession, Window, functions as F, types as T

resources = Path(__file__).parent.parent.joinpath('resources')
spark = SparkSession.builder.getOrCreate()


class TestDataFrame(TestCase):
    def test_where(self):
        df = spark.read.csv(str(resources.joinpath('test.csv')), header=True)
        df = df.where(F.col('a') == '1')

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([{'a': '1', 'b': '2', 'c': '3'}], rows)

    def test_select(self):
        df = spark.read.csv(str(resources.joinpath('test.csv')), header=True)
        df = df.where(F.col('a') == '4')
        df = df.select(F.col('a').alias('A'), 'b')
        df = df.select('*')
        df = df.select('*')
        df = df.select('*')

        when = F.when(F.col('b') == '5', F.lit('is 5'))
        when = when.when(F.col('b') == '50', F.lit('is 50'))
        when = when.otherwise('is other')
        alias = '*/-100/*'
        df = df.select('A', when.alias(alias))
        df = df.select(['A', alias])
        df = df.where(F.col(alias).isNotNull())

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([{'A': '4', alias: 'is 5'}], rows)

    def test_group(self):
        df = spark.read.csv(str(resources.joinpath('group.csv')))
        df = df.select('*')
        df = df.withColumnRenamed('_c0', 'name')
        df = df.withColumnRenamed('_c1', 'type')
        df = df.select('*')
        df = df.groupBy('name', 'type').agg(
            F.min('name').alias('unused'),
            F.count('*').alias('count'),
        )
        df = df.select('name', 'type', 'count')

        dfs = [df]

        df = (
            df.groupBy(F.col('count').alias('original_count'))
            .agg(F.count('*').alias('meta_count'))
        )

        dfs.append(df)

        df = df.groupBy(F.lit(' ')).agg(F.count('*'))

        dfs.append(df)

        rows_list = [df.collect() for df in dfs]

        rows = [row.asDict() for row in rows_list[0]]
        self.assertEqual([
            {'name': 'n1', 'type': 't1', 'count': 2},
            {'name': 'n1', 'type': 't2', 'count': 1},
            {'name': 'n2', 'type': 't1', 'count': 2},
            {'name': 'n3', 'type': 't1', 'count': 1},
        ], sorted(rows, key=lambda row: (row['name'], row['type'])))

        rows = [row.asDict() for row in rows_list[1]]
        self.assertEqual([
            {'original_count': 1, 'meta_count': 2},
            {'original_count': 2, 'meta_count': 2},
        ], sorted(rows, key=lambda row: row['original_count']))

        rows = [row.asDict() for row in rows_list[2]]
        self.assertEqual([{' ': ' ', 'count(1)': 2}], rows)

    def test_group_complex_agg(self):
        df = spark.read.csv(str(resources.joinpath('group_complex_agg.csv')), header=True)
        df = df.withColumn('number', F.col('number').cast(T.LongType()))
        df = df.groupBy('group').agg(1 + F.sum('number') + F.count('*') + 1)

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'group': 'a', '(((sum(number) + 1) + count(1)) + 1)': 12},
            {'group': 'b', '(((sum(number) + 1) + count(1)) + 1)': 10},
        ], sorted(rows, key=lambda row: row['group']))

    def test_union(self):
        a_df = spark.read.csv(str(resources.joinpath('union_a.csv')), header=True)
        b_df = spark.read.csv(str(resources.joinpath('union_b.csv')), header=True)
        df = a_df.unionByName(b_df)

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'a': 'a1', 'b': 'b1', 'c': 'c1'},
            {'a': 'a2', 'b': 'b2', 'c': 'c2'},
            {'a': 'a3', 'b': 'b3', 'c': 'c3'},
            {'a': 'a4', 'b': 'b4', 'c': 'c4'},
            {'a': 'a5', 'b': 'b5', 'c': 'c5'},
            {'a': 'a6', 'b': 'b6', 'c': 'c6'},
        ], rows)

    def test_union_2(self):
        df = spark.read.csv(str(resources.joinpath('union_a.csv')), header=True)
        a_df = df.where(F.col('a') == 'a1')
        b_df = df.where(F.col('b') == 'b2')
        a_df = a_df.withColumn('message', F.lit('AAH!'))
        b_df = b_df.withColumn('message', F.lit('BRR!'))
        df = a_df.unionByName(b_df)

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'a': 'a1', 'b': 'b1', 'c': 'c1', 'message': 'AAH!'},
            {'a': 'a2', 'b': 'b2', 'c': 'c2', 'message': "BRR!"},
        ], rows)

    def test_union_allow_missing(self):
        left = spark.createDataFrame(
            data=[
                [1, 2, 3, 4, 5],
            ],
            schema=T.StructType([
                T.StructField('a', T.IntegerType(), False),
                T.StructField('b', T.IntegerType(), False),
                T.StructField('c', T.IntegerType(), False),
                T.StructField('d', T.IntegerType(), False),
                T.StructField('e', T.IntegerType(), False),
            ]),
        )
        right = spark.createDataFrame(
            data=[
                [6, 7, 8.0, 9.0, 10],
                [11, None, 12.0, None, None],
            ],
            schema=T.StructType([
                T.StructField('b', T.IntegerType(), False),
                T.StructField('c', T.IntegerType()),
                T.StructField('d', T.DoubleType(), False),
                T.StructField('e', T.DoubleType()),
                T.StructField('f', T.IntegerType()),
            ]),
        )
        df = left.unionByName(right, allowMissingColumns=True)

        self.assertEqual(T.StructType([
            T.StructField('a', T.IntegerType()),
            T.StructField('b', T.IntegerType(), False),
            T.StructField('c', T.IntegerType()),
            T.StructField('d', T.DoubleType(), False),
            T.StructField('e', T.DoubleType()),
            T.StructField('f', T.IntegerType()),
        ]), df.schema)

        rows = df.collect()
        self.assertEqual([
            T.Row(a=1, b=2, c=3, d=4.0, e=5.0, f=None),
            T.Row(a=None, b=6, c=7, d=8.0, e=9.0, f=10),
            T.Row(a=None, b=11, c=None, d=12.0, e=None, f=None),
        ], rows)

    def test_union_type_merge(self):
        left = spark.createDataFrame(
            data=[
                [1, 2, 3, 4],
            ],
            schema=T.StructType([
                T.StructField('a', T.IntegerType(), False),
                T.StructField('b', T.IntegerType(), False),
                T.StructField('c', T.IntegerType(), False),
                T.StructField('d', T.IntegerType(), False),
            ]),
        )
        right = spark.createDataFrame(
            data=[
                [5, 6, 7.0, 8.0],
                [9, None, 10.0, None],
            ],
            schema=T.StructType([
                T.StructField('a', T.IntegerType(), False),
                T.StructField('b', T.IntegerType()),
                T.StructField('c', T.DoubleType(), False),
                T.StructField('d', T.DoubleType()),
            ]),
        )
        df = left.unionByName(right)

        self.assertEqual(T.StructType([
            T.StructField('a', T.IntegerType(), False),
            T.StructField('b', T.IntegerType()),
            T.StructField('c', T.DoubleType(), False),
            T.StructField('d', T.DoubleType()),
        ]), df.schema)

        rows = df.collect()
        self.assertEqual([
            T.Row(a=1, b=2, c=3.0, d=4.0),
            T.Row(a=5, b=6, c=7.0, d=8.0),
            T.Row(a=9, b=None, c=10.0, d=None),
        ], rows)

    def test_window(self):
        df = spark.read.csv(
            str(resources.joinpath('window.csv')),
            T.StructType([
                T.StructField('group', T.StringType()),
                T.StructField('order', T.IntegerType()),
                T.StructField('value', T.IntegerType()),
            ]),
        )
        window_1 = Window.partitionBy('group').orderBy('order')
        window_2 = Window.partitionBy('group')
        window_3 = Window.partitionBy('order')
        df = df.select(
            '*',
            F.sum('value').over(window_1).alias('sum'),
            F.sum('order').over(window_2).alias('order_sum'),
            F.count('*').over(window_1).alias('count'),
            F.count('*').over(window_2).alias('group_count'),
            (F.sum('value').over(window_1) == F.count('*').over(window_2)).alias('equal'),
            F.count('*').over(window_3),
        )
        df = df.select('sum', 'order_sum', 'count', 'group_count', 'equal')

        rows = [row.asDict() for row in df.collect()]
        rows = sorted(rows, key=lambda row: row['sum'])
        self.assertEqual([
            {'sum': 10, 'order_sum': 6, 'count': 1, 'group_count': 3, 'equal': False},
            {'sum': 30, 'order_sum': 6, 'count': 2, 'group_count': 3, 'equal': False},
            {'sum': 40, 'order_sum': 9, 'count': 1, 'group_count': 2, 'equal': False},
            {'sum': 60, 'order_sum': 6, 'count': 3, 'group_count': 3, 'equal': False},
            {'sum': 90, 'order_sum': 9, 'count': 2, 'group_count': 2, 'equal': False},
        ], rows)

    def test_window_order(self):
        df = spark.createDataFrame(
            data=[
                ['', None],
                [None, None],
                ['ab', None],
                ['b', None],
                ['a', 3],
                ['a', 1],
                ['a', None],
                ['a', 2],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
                T.StructField('i', T.IntegerType()),
            ]),
        )
        df = df.select(
            '*',
            F.row_number().over(Window.orderBy('s', 'i')).alias('n1'),
            F.row_number().over(Window.orderBy(F.asc('s'), F.col('i').desc())).alias('n2'),
        )

        rows = df.collect()
        self.assertEqual([
            T.Row(s=None, i=None, n1=1, n2=1),
            T.Row(s='', i=None, n1=2, n2=2),
            T.Row(s='a', i=3, n1=6, n2=3),
            T.Row(s='a', i=2, n1=5, n2=4),
            T.Row(s='a', i=1, n1=4, n2=5),
            T.Row(s='a', i=None, n1=3, n2=6),
            T.Row(s='ab', i=None, n1=7, n2=7),
            T.Row(s='b', i=None, n1=8, n2=8),
        ], rows)

    def test_explode(self):
        df = spark.createDataFrame(
            data=[
                ['s1', 1, 10],
                ['s2', 2, 20],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
                T.StructField('i1', T.IntegerType()),
                T.StructField('i2', T.IntegerType()),
            ]),
        )
        df = df.withColumn('arr', F.array(F.lit(0), F.col('i1'), F.col('i2')))
        df = df.select('*', F.explode('arr'))
        df = df.drop('arr')

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'s': 's1', 'i1': 1, 'i2': 10, 'col': 0},
            {'s': 's1', 'i1': 1, 'i2': 10, 'col': 1},
            {'s': 's1', 'i1': 1, 'i2': 10, 'col': 10},
            {'s': 's2', 'i1': 2, 'i2': 20, 'col': 0},
            {'s': 's2', 'i1': 2, 'i2': 20, 'col': 2},
            {'s': 's2', 'i1': 2, 'i2': 20, 'col': 20},
        ], rows)

    def test_explode_outer(self):
        df = spark.createDataFrame(
            data=[
                [[None, 1, 2]],
                [[]],
                [None],
            ],
            schema=T.StructType([
                T.StructField('a', T.ArrayType(T.IntegerType())),
            ]),
        )
        df_inner = df.select('*', F.explode('a'))

        rows = [row.asDict() for row in df_inner.collect()]
        self.assertEqual([
            {'a': [None, 1, 2], 'col': None},
            {'a': [None, 1, 2], 'col': 1},
            {'a': [None, 1, 2], 'col': 2},
        ], rows)

        df_outer = df.select('*', F.explode_outer('a'))

        rows = [row.asDict() for row in df_outer.collect()]
        self.assertEqual([
            {'a': [None, 1, 2], 'col': None},
            {'a': [None, 1, 2], 'col': 1},
            {'a': [None, 1, 2], 'col': 2},
            {'a': [], 'col': None},
            {'a': None, 'col': None},
        ], rows)

    def test_distinct_with_list(self):
        df = spark.createDataFrame(
            data=[
                [None],
                [None],
                [[]],
                [[]],
                [[None]],
                [[None]],
                [['a']],
                [['a']],
                [['a', 'b']],
                [['a', 'b']],
                [['b', 'a']],
                [['b', 'a']],
            ],
            schema=T.StructType([
                T.StructField('l', T.ArrayType(T.StringType())),
            ]),
        )
        df = df.distinct()

        rows = sorted(df.collect(), key=str)
        self.assertEqual([
            T.Row(l=None),
            T.Row(l=['a', 'b']),
            T.Row(l=['a']),
            T.Row(l=['b', 'a']),
            T.Row(l=[None]),
            T.Row(l=[]),
        ], rows)

    def test_drop_duplicates(self):
        df = spark.createDataFrame(
            data=[
                ['a', 1, 10],
                ['a', 1, 10],
                ['a', 1, 20],
                ['b', 1, 30],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
                T.StructField('i', T.IntegerType()),
                T.StructField('l', T.LongType()),
            ]),
        )
        group_df = df.groupBy(*df.columns).agg(F.lit(None).alias('_')).drop('_')
        distinct_df = df.distinct()
        dedup_df = df.drop_duplicates()
        subset_dedup_df = df.drop_duplicates(['s', 'i'])

        self.assertEqual(group_df.schema, distinct_df.schema)
        self.assertEqual(group_df.schema, dedup_df.schema)
        self.assertEqual(group_df.schema, subset_dedup_df.schema)

        expected = [T.Row(s='a', i=1, l=10), T.Row(s='a', i=1, l=20), T.Row(s='b', i=1, l=30)]
        rows = sorted(group_df.collect(), key=lambda row: row.l)
        self.assertEqual(expected, rows)
        rows = sorted(distinct_df.collect(), key=lambda row: row.l)
        self.assertEqual(expected, rows)
        rows = sorted(dedup_df.collect(), key=lambda row: row.l)
        self.assertEqual(expected, rows)

        rows = sorted(subset_dedup_df.collect(), key=lambda row: row.l)
        self.assertEqual([T.Row(s='a', i=1, l=10), T.Row(s='b', i=1, l=30)], rows)

    def test_alias(self):
        a = spark.createDataFrame(
            data=[
                [False, 1],
            ],
            schema=T.StructType([
                T.StructField('a', T.BooleanType()),
                T.StructField('b', T.IntegerType()),
            ]),
        )
        b = spark.createDataFrame(
            data=[
                [(2,), 'b', (2.0,)],
            ],
            schema=T.StructType([
                T.StructField('a', T.StructType([
                    T.StructField('b', T.LongType()),
                ])),
                T.StructField('b', T.StringType()),
                T.StructField('c', T.StructType([
                    T.StructField('b', T.DoubleType()),
                ])),
            ]),
        )
        join_df = a.alias('a').join(b.alias('b'), F.lit(True))
        df = join_df.select('a.a', 'a.b', 'b.a.b', 'b.b', 'c.b')

        self.assertEqual(df.schema, T.StructType([
            T.StructField('a', T.BooleanType()),
            T.StructField('b', T.IntegerType()),
            T.StructField('b', T.LongType()),
            T.StructField('b', T.StringType()),
            T.StructField('b', T.DoubleType()),
        ]))
        rows = df.collect()
        self.assertEqual([(False, 1, 2, 'b', 2.0)], rows)

        df = join_df.select('a.*')
        rows = df.collect()
        self.assertEqual([T.Row(a=False, b=1)], rows)

        df = join_df.select('b.*')
        rows = df.collect()
        self.assertEqual([T.Row(a=T.Row(b=2), b='b', c=T.Row(b=2.0))], rows)

    def test_replace(self):
        df = spark.createDataFrame(
            data=[
                [None, None],
                ["'", 1],
                ["''", 2],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
                T.StructField('i', T.IntegerType()),
            ]),
        )
        df = df.replace("''", '')

        rows = df.collect()
        self.assertEqual([
            T.Row(s=None, i=None),
            T.Row(s="'", i=1),
            T.Row(s='', i=2),
        ], rows)

    def test_createDataFrame_schema_inference(self):
        df = spark.createDataFrame([
            {'a': 'abc', 'b': 123, 'c': None},
            {'b': 123.5, 'c': 456, 'd': True},
        ])
        self.assertEqual(T.StructType([
            T.StructField('a', T.StringType()),
            T.StructField('b', T.DoubleType(), False),
            T.StructField('c', T.IntegerType()),
            T.StructField('d', T.BooleanType()),
        ]), df.schema)
