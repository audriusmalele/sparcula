from pathlib import Path
from unittest import TestCase

from sparcula.pyspark.sql import SparkSession, functions as F, types as T

resources = Path(__file__).parent.parent.joinpath('resources')
spark = SparkSession.builder.getOrCreate()


class TestJoin(TestCase):
    def test_simple_join(self):
        a_df = spark.read.csv(str(resources.joinpath('join_a.csv')), header=True)
        b_df = spark.read.csv(str(resources.joinpath('join_b.csv')), header=True)
        df = a_df.join(b_df, ['a', 'b'])

        sorted_rows = sorted(df.collect(), key=lambda row: row['e'])
        self.assertEqual([
            T.Row(a='a', b='1', c='A', d='-1', e='010'),
            T.Row(a='b', b='2', c='B', d='-2', e='020'),
            T.Row(a='b', b='2', c='B', d='-2', e='200'),
        ], sorted_rows)

        dfa = df.select(a_df['*'])

        sorted_rows = sorted(dfa.collect(), key=lambda row: row['a'])
        self.assertEqual([
            T.Row(a='a', b='1', c='A', d='-1'),
            T.Row(a='b', b='2', c='B', d='-2'),
            T.Row(a='b', b='2', c='B', d='-2'),
        ], sorted_rows)

        dfb = df.select(b_df['*'])

        sorted_rows = sorted(dfb.collect(), key=lambda row: row['e'])
        self.assertEqual([
            T.Row(a='a', b='1', e='010'),
            T.Row(a='b', b='2', e='020'),
            T.Row(a='b', b='2', e='200'),
        ], sorted_rows)

        df = a_df.join(df, ['a', 'b', 'c', 'd'])

        sorted_rows = sorted(df.collect(), key=lambda row: row['e'])
        self.assertEqual([
            T.Row(a='a', b='1', c='A', d='-1', e='010'),
            T.Row(a='b', b='2', c='B', d='-2', e='020'),
            T.Row(a='b', b='2', c='B', d='-2', e='200'),
        ], sorted_rows)

    def test_join_column_condition(self):
        left = spark.createDataFrame(
            data=[
                ['a1', 'b1', 'c1', 'd1'],
                ['a2', 'b2', 'c2', 'd2'],
                ['a3', 'b3', 'c3', 'd3'],
            ],
            schema=T.StructType([
                T.StructField('a', T.StringType()),
                T.StructField('b', T.StringType()),
                T.StructField('c', T.StringType()),
                T.StructField('d', T.StringType()),
            ]),
        )
        right = spark.createDataFrame(
            data=[
                ['a1', 'b1', 'c10', 'e1'],
                ['a2', 'b2', 'c20', 'e2'],
                ['a3', 'b3', 'c3', 'e3'],
            ],
            schema=T.StructType([
                T.StructField('a', T.StringType()),
                T.StructField('b', T.StringType()),
                T.StructField('c', T.StringType()),
                T.StructField('e', T.StringType()),
            ]),
        )
        right = right.withColumnRenamed('c', 'cr')
        df = left.join(right, (
            (left['a'] == right['a']) &
            right['b'].eqNullSafe(left['b']) &
            (F.col('c') != F.col('cr'))
        ), how=None)
        df = df.drop(right['a'])
        df = df.drop(right['b'])

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'a': 'a1', 'b': 'b1', 'c': 'c1', 'd': 'd1', 'cr': 'c10', 'e': 'e1'},
            {'a': 'a2', 'b': 'b2', 'c': 'c2', 'd': 'd2', 'cr': 'c20', 'e': 'e2'},
        ], rows)

    left_rows = [
        T.Row(a='a3', b='b3', c='c30', d=3, ar=None, br=None, cr=None, e=None),
        T.Row(a=None, b='b4', c='c4', d=4, ar=None, br=None, cr=None, e=None),
    ]
    right_rows = [
        T.Row(a=None, b=None, c=None, d=None, ar='a3', br='b3', cr='c30', e=30),
        T.Row(a=None, b=None, c=None, d=None, ar=None, br='b4', cr='c40', e=40),
    ]

    def test_inner_join(self):
        self._test_join('inner', [])

    def test_left_join(self):
        self._test_join('left', self.left_rows)

    def test_right_join(self):
        self._test_join('right', self.right_rows)

    def test_full_join(self):
        self._test_join('full', self.left_rows + self.right_rows)

    def _test_join(self, how: str, additional_expected_rows: list['T.Row']):
        left = spark.createDataFrame(
            data=[
                ['a1', 'b1', 'c1', 1],
                ['a1', 'b1', 'c1', 2],
                ['a2', 'b2', 'c2', 2],
                ['a3', 'b3', 'c30', 3],
                [None, 'b4', 'c4', 4],
            ],
            schema=T.StructType([
                T.StructField('a', T.StringType()),
                T.StructField('b', T.StringType()),
                T.StructField('c', T.StringType()),
                T.StructField('d', T.IntegerType(), False),
            ]),
        )
        right = spark.createDataFrame(
            data=[
                ['a1', 'b1', 'c10', 10],
                ['a2', 'b2', 'c20', 20],
                ['a3', 'b3', 'c30', 30],
                [None, 'b4', 'c40', 40],
            ],
            schema=T.StructType([
                T.StructField('ar', T.StringType()),
                T.StructField('br', T.StringType()),
                T.StructField('cr', T.StringType()),
                T.StructField('e', T.IntegerType(), False),
            ]),
        )
        df = left.join(F.broadcast(right), (
            (left['a'] == right['ar']) &
            right['br'].eqNullSafe(left['b']) &
            (left['c'] != right['cr'])
        ), how)

        rows = df.collect()
        sorted_rows = sorted(rows, key=lambda row: (row['d'] or 99, row['e'] or 99))
        self.assertEqual([
            T.Row(a='a1', b='b1', c='c1', d=1, ar='a1', br='b1', cr='c10', e=10),
            T.Row(a='a1', b='b1', c='c1', d=2, ar='a1', br='b1', cr='c10', e=10),
            T.Row(a='a2', b='b2', c='c2', d=2, ar='a2', br='b2', cr='c20', e=20),
        ] + additional_expected_rows, sorted_rows)

    def test_semi_join(self):
        self._test_filter_join('semi', [
            T.Row(a='a1', b='b1', c='c1', d='d1'),
            T.Row(a='a2', b='b2', c='c2', d='d2'),
        ])

    def test_anti_join(self):
        self._test_filter_join('anti', [
            T.Row(a='a3', b='b3', c='c30', d='d3'),
            T.Row(a=None, b='b4', c='c4', d='d4'),
        ])

    def _test_filter_join(self, how: str, expected_rows: list['T.Row']):
        left = spark.createDataFrame(
            data=[
                ['a1', 'b1', 'c1', 'd1'],
                ['a2', 'b2', 'c2', 'd2'],
                ['a3', 'b3', 'c30', 'd3'],
                [None, 'b4', 'c4', 'd4'],
            ],
            schema=T.StructType([
                T.StructField('a', T.StringType()),
                T.StructField('b', T.StringType()),
                T.StructField('c', T.StringType()),
                T.StructField('d', T.StringType()),
            ]),
        )
        right = spark.createDataFrame(
            data=[
                ['a1', 'b1', 'c10', 'e1'],
                ['a1', 'b1', 'c10', 'e2'],
                ['a2', 'b2', 'c20', 'e2'],
                ['a3', 'b3', 'c30', 'e3'],
                [None, 'b4', 'c40', 'e4'],
            ],
            schema=T.StructType([
                T.StructField('ar', T.StringType()),
                T.StructField('br', T.StringType()),
                T.StructField('cr', T.StringType()),
                T.StructField('e', T.StringType()),
            ]),
        )
        df = left.join(right, (
            (left['a'] == right['ar']) &
            right['br'].eqNullSafe(left['b']) &
            (left['c'] != right['cr'])
        ), how)

        rows = df.collect()
        sorted_rows = sorted(rows, key=lambda row: row['d'])
        self.assertEqual(expected_rows, sorted_rows)

    def test_cross_join(self):
        left = spark.createDataFrame(
            data=[
                ['a1', 'b1'],
                ['a2', 'b2'],
            ],
            schema=T.StructType([
                T.StructField('a', T.StringType()),
                T.StructField('b', T.StringType()),
            ]),
        )
        right = spark.createDataFrame(
            data=[
                ['c1', 'd1'],
                ['c2', 'd2'],
            ],
            schema=T.StructType([
                T.StructField('c', T.StringType()),
                T.StructField('d', T.StringType()),
            ]),
        )
        df = left.crossJoin(right)

        rows = df.collect()
        rows = sorted(rows, key=lambda row: (row['a'], row['c']))
        self.assertEqual([
            T.Row(a='a1', b='b1', c='c1', d='d1'),
            T.Row(a='a1', b='b1', c='c2', d='d2'),
            T.Row(a='a2', b='b2', c='c1', d='d1'),
            T.Row(a='a2', b='b2', c='c2', d='d2'),
        ], rows)
