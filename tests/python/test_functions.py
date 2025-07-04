import math
from datetime import date, datetime
from unittest import TestCase

import utils

from sparcula.pyspark.sql import SparkSession, functions as F, types as T

spark = SparkSession.builder.getOrCreate()


class TestFunctions(TestCase):
    def test_when(self):
        df = spark.createDataFrame([[]])
        df = df.select(F.when(F.lit(False),  F.lit('abc')).otherwise(F.lit(1.5)))
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([{'CASE WHEN false THEN abc ELSE 1.5 END': '1.5'}], rows)

    def test_like(self):
        df = spark.createDataFrame(
            data=[
                [None],
                [''],
                ['Alice '],
                ['Alice'],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
            ]),
        )
        df = df.select(
            '*',
            F.col('s').like('lice'),
            F.col('s').like('A_ice_'),
            F.col('s').like('%ice%'),
        )

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'s': None, 's LIKE lice': None, 's LIKE A_ice_': None, 's LIKE %ice%': None},
            {'s': '', 's LIKE lice': False, 's LIKE A_ice_': False, 's LIKE %ice%': False},
            {'s': 'Alice ', 's LIKE lice': False, 's LIKE A_ice_': True, 's LIKE %ice%': True},
            {'s': 'Alice', 's LIKE lice': False, 's LIKE A_ice_': False, 's LIKE %ice%': True},
        ], rows)

    def test_rlike(self):
        df = spark.createDataFrame(
            data=[
                [None],
                [''],
                ['Alice '],
                ['Alice'],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
            ]),
        )
        df = df.select(
            '*',
            F.col('s').rlike('ice$'),
            F.col('s').rlike('(?i)alice'),
            F.col('s').rlike('^(?i)alice'),
        )

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {
                's': None,
                'RLIKE(s, ice$)': None,
                'RLIKE(s, (?i)alice)': None,
                'RLIKE(s, ^(?i)alice)': None,
            },
            {
                's': '',
                'RLIKE(s, ice$)': False,
                'RLIKE(s, (?i)alice)': False,
                'RLIKE(s, ^(?i)alice)': False,
            },
            {
                's': 'Alice ',
                'RLIKE(s, ice$)': False,
                'RLIKE(s, (?i)alice)': True,
                'RLIKE(s, ^(?i)alice)': True,
            },
            {
                's': 'Alice',
                'RLIKE(s, ice$)': True,
                'RLIKE(s, (?i)alice)': True,
                'RLIKE(s, ^(?i)alice)': True,
            },
        ], rows)

    def test_regexp_replace(self):
        df = spark.createDataFrame(
            data=[
                [None, None, None],
                ['1-20', '\\d+', 'n'],
                ['1-20', '\\d', 'd'],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
                T.StructField('p', T.StringType()),
                T.StructField('r', T.StringType()),
            ]),
        )
        df = df.select('*', F.regexp_replace('s', F.col('p'), F.col('r')))

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'s': None, 'p': None, 'r': None, 'regexp_replace(s, p, r, 1)': None},
            {'s': '1-20', 'p': '\\d+', 'r': 'n', 'regexp_replace(s, p, r, 1)': 'n-n'},
            {'s': '1-20', 'p': '\\d', 'r': 'd', 'regexp_replace(s, p, r, 1)': 'd-dd'},
        ], rows)

    def test_regexp_extract(self):
        df = spark.createDataFrame(
            data=[
                [None, None],
                ['-2020-12-31-', date(2020, 12, 31)],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
                T.StructField('d', T.DateType()),
            ]),
        )
        df = df.select(
            '*',
            *[
                F.regexp_extract('s', r'(\d{4})-(\d{2})-(\d{2})', i).alias(f's{i}')
                for i in range(4)
            ],
            *[
                F.regexp_extract('d', r'(\d{4})-(\d{2})-(\d{2})', i).alias(f'd{i}')
                for i in range(4)
            ],
        )
        rows = df.collect()
        self.assertEqual(rows, [
            T.Row(
                s=None, d=None,
                s0=None, s1=None, s2=None, s3=None,
                d0=None, d1=None, d2=None, d3=None,
            ),
            T.Row(
                s='-2020-12-31-', d=date(2020, 12, 31),
                s0='2020-12-31', s1='2020', s2='12', s3='31',
                d0='2020-12-31', d1='2020', d2='12', d3='31',
            ),
        ])

    def test_split_and_element_at(self):
        df = spark.createDataFrame(
            data=[
                [None],
                ['a,b,c'],
            ],
            schema=T.StructType([T.StructField('s', T.StringType())]),
        )
        df = df.withColumn('arr', F.split('s', ','))
        df = df.select(*[
            F.element_at('arr', i).alias(str(i))
            for i in [-4, -3, -2, -1, 1, 2, 3, 4]
        ])
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'-4': None, '-3': None, '-2': None, '-1': None, '1': None, '2': None, '3': None, '4': None},
            {'-4': None, '-3': 'a', '-2': 'b', '-1': 'c', '1': 'a', '2': 'b', '3': 'c', '4': None},
        ], rows)

    def test_split_part(self):
        df = spark.createDataFrame(
            data=[
                [None],
                ['a..b..c'],
            ],
            schema=T.StructType([T.StructField('s', T.StringType())]),
        )
        df = df.select(
            '*',
            F.split_part('s', F.lit('..'), F.lit(-4)),
            F.split_part('s', F.lit('..'), F.lit(-3)),
            F.split_part('s', F.lit('..'), F.lit(-2)),
            F.split_part('s', F.lit('..'), F.lit(-1)),
            F.split_part('s', F.lit('..'), F.lit(1)),
            F.split_part('s', F.lit('..'), F.lit(2)),
            F.split_part('s', F.lit('..'), F.lit(3)),
            F.split_part('s', F.lit('..'), F.lit(4)),
        )

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {
                's': None,
                'split_part(s, .., -1)': None,
                'split_part(s, .., -2)': None,
                'split_part(s, .., -3)': None,
                'split_part(s, .., -4)': None,
                'split_part(s, .., 1)': None,
                'split_part(s, .., 2)': None,
                'split_part(s, .., 3)': None,
                'split_part(s, .., 4)': None,
            },
            {
                's': 'a..b..c',
                'split_part(s, .., -1)': 'c',
                'split_part(s, .., -2)': 'b',
                'split_part(s, .., -3)': 'a',
                'split_part(s, .., -4)': '',
                'split_part(s, .., 1)': 'a',
                'split_part(s, .., 2)': 'b',
                'split_part(s, .., 3)': 'c',
                'split_part(s, .., 4)': '',
            },
        ], rows)

    def test_date_format(self):
        df = spark.createDataFrame(
            data=[
                [None],
                [date(2020, 12, 31)],
            ],
            schema=T.StructType([
                T.StructField('d', T.DateType()),
            ]),
        )
        df = df.select(
            '*',
            F.date_format('d', 'yyyy-MM-dd'),
        )

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {
                'd': None,
                'date_format(d, yyyy-MM-dd)': None,
            },
            {
                'd': date(2020, 12, 31),
                'date_format(d, yyyy-MM-dd)': '2020-12-31',
            },
        ], rows)

    def test_to_date(self):
        df = spark.createDataFrame(
            data=[
                [None, None, None, None, None, None, None, None, None, None],
                [
                    '2020-12-31', '20201231', '2020/12/31', '20-12-31', '31-12-2020',
                    '31-12-20', '31-Dec-2020', '12-31-2020', 12312020, '1.01.2020',
                ],
                ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 9, 'j'],
            ],
            schema=T.StructType([
                T.StructField('a', T.StringType()),
                T.StructField('b', T.StringType()),
                T.StructField('c', T.StringType()),
                T.StructField('d', T.StringType()),
                T.StructField('e', T.StringType()),
                T.StructField('f', T.StringType()),
                T.StructField('g', T.StringType()),
                T.StructField('h', T.StringType()),
                T.StructField('i', T.IntegerType()),
                T.StructField('j', T.StringType()),
            ]),
        )
        df = df.select(
            '*',
            F.to_date('a'),
            F.to_date('a', 'yyyy-MM-dd'),
            F.to_date('b', 'yyyyMMdd'),
            F.to_date('c', 'yyyy/MM/dd'),
            F.to_date('d', 'yy-MM-dd'),
            F.to_date('e', 'dd-MM-yyyy'),
            F.to_date('f', 'dd-MM-yy'),
            F.to_date('g', 'dd-MMM-yyyy'),
            F.to_date('h', 'MM-dd-yyyy'),
            F.to_date('i', 'MMddyyyy'),
            F.to_date('j', 'd.M.yyyy'),
        )
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {
                'a': None,
                'b': None,
                'c': None,
                'd': None,
                'e': None,
                'f': None,
                'g': None,
                'h': None,
                'i': None,
                'j': None,
                'to_date(a)': None,
                'to_date(a, yyyy-MM-dd)': None,
                'to_date(b, yyyyMMdd)': None,
                'to_date(c, yyyy/MM/dd)': None,
                'to_date(d, yy-MM-dd)': None,
                'to_date(e, dd-MM-yyyy)': None,
                'to_date(f, dd-MM-yy)': None,
                'to_date(g, dd-MMM-yyyy)': None,
                'to_date(h, MM-dd-yyyy)': None,
                'to_date(i, MMddyyyy)': None,
                'to_date(j, d.M.yyyy)': None,
            },
            {
                'a': '2020-12-31',
                'b': '20201231',
                'c': '2020/12/31',
                'd': '20-12-31',
                'e': '31-12-2020',
                'f': '31-12-20',
                'g': '31-Dec-2020',
                'h': '12-31-2020',
                'i': 12312020,
                'j': '1.01.2020',
                'to_date(a)': date(2020, 12, 31),
                'to_date(a, yyyy-MM-dd)': date(2020, 12, 31),
                'to_date(b, yyyyMMdd)': date(2020, 12, 31),
                'to_date(c, yyyy/MM/dd)': date(2020, 12, 31),
                'to_date(d, yy-MM-dd)': date(2020, 12, 31),
                'to_date(e, dd-MM-yyyy)': date(2020, 12, 31),
                'to_date(f, dd-MM-yy)': date(2020, 12, 31),
                'to_date(g, dd-MMM-yyyy)': date(2020, 12, 31),
                'to_date(h, MM-dd-yyyy)': date(2020, 12, 31),
                'to_date(i, MMddyyyy)': date(2020, 12, 31),
                'to_date(j, d.M.yyyy)': date(2020, 1, 1),
            },
            {
                'a': 'a',
                'b': 'b',
                'c': 'c',
                'd': 'd',
                'e': 'e',
                'f': 'f',
                'g': 'g',
                'h': 'h',
                'i': 9,
                'j': 'j',
                'to_date(a)': None,
                'to_date(a, yyyy-MM-dd)': None,
                'to_date(b, yyyyMMdd)': None,
                'to_date(c, yyyy/MM/dd)': None,
                'to_date(d, yy-MM-dd)': None,
                'to_date(e, dd-MM-yyyy)': None,
                'to_date(f, dd-MM-yy)': None,
                'to_date(g, dd-MMM-yyyy)': None,
                'to_date(h, MM-dd-yyyy)': None,
                'to_date(i, MMddyyyy)': None,
                'to_date(j, d.M.yyyy)': None,
            },
        ], rows)

    def test_to_timestamp(self):
        df = spark.createDataFrame(
            data=[
                [None],
                ['2020-12-31 23:59:59'],
                ['a'],
            ],
            schema=T.StructType([
                T.StructField('a', T.StringType()),
            ]),
        )
        df = df.select(
            '*',
            F.to_timestamp('a'),
            F.to_timestamp('a', 'yyyy-MM-dd HH:mm:ss'),
        )

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {
                'a': None,
                'to_timestamp(a)': None,
                'to_timestamp(a, yyyy-MM-dd HH:mm:ss)': None,
            },
            {
                'a': '2020-12-31 23:59:59',
                'to_timestamp(a)': datetime(2020, 12, 31, 23, 59, 59),
                'to_timestamp(a, yyyy-MM-dd HH:mm:ss)': datetime(2020, 12, 31, 23, 59, 59),
            },
            {
                'a': 'a',
                'to_timestamp(a)': None,
                'to_timestamp(a, yyyy-MM-dd HH:mm:ss)': None,
            },
        ], rows)

    def test_isin(self):
        df = spark.createDataFrame(
            data=[
                ['a'],
                ['b'],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
            ]),
        )
        df = df.select(
            '*',
            F.col('s').isin(['b', 'c']),
            F.col('s').isin({'b'}),
            F.col('s').isin(F.col('s'), None),
        )
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual(rows, [
            {'s': 'a', '(s IN (b, c))': False, '(s IN (b))': False, '(s IN (s, NULL))': True},
            {'s': 'b', '(s IN (b, c))': True, '(s IN (b))': True, '(s IN (s, NULL))': True},
        ])

    def test_logical_operators(self):
        df = spark.createDataFrame(
            data=[
                [None, None],
                [None, False],
                [None, True],
                [False, None],
                [False, False],
                [False, True],
                [True, None],
                [True, False],
                [True, True],
            ],
            schema=T.StructType([
                T.StructField('a', T.BooleanType()),
                T.StructField('b', T.BooleanType()),
            ]),
        )
        df = df.select('*', ~F.col('a'), F.col('a') & F.col('b'), F.col('a') | F.col('b'))
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'a': None, 'b': None, '(NOT a)': None, '(a AND b)': None, '(a OR b)': None},
            {'a': None, 'b': False, '(NOT a)': None, '(a AND b)': False, '(a OR b)': None},
            {'a': None, 'b': True, '(NOT a)': None, '(a AND b)': None, '(a OR b)': True},
            {'a': False, 'b': None, '(NOT a)': True, '(a AND b)': False, '(a OR b)': None},
            {'a': False, 'b': False, '(NOT a)': True, '(a AND b)': False, '(a OR b)': False},
            {'a': False, 'b': True, '(NOT a)': True, '(a AND b)': False, '(a OR b)': True},
            {'a': True, 'b': None, '(NOT a)': False, '(a AND b)': None, '(a OR b)': True},
            {'a': True, 'b': False, '(NOT a)': False, '(a AND b)': False, '(a OR b)': True},
            {'a': True, 'b': True, '(NOT a)': False, '(a AND b)': True, '(a OR b)': True},
        ], rows)

    def test_eqNullSafe(self):
        a = datetime(2020, 12, 31, 23, 59, 58)
        b = datetime(2020, 12, 31, 23, 59, 59)
        df = spark.createDataFrame(
            data=[
                [None, None],
                [None, a],
                [a, a],
                [a, b],
            ],
            schema=T.StructType([
                T.StructField('a', T.TimestampType()),
                T.StructField('b', T.TimestampType()),
            ]),
        )
        df = df.select('*', F.col('a').eqNullSafe(F.col('b')))
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'a': None, 'b': None, '(a <=> b)': True},
            {'a': None, 'b': a, '(a <=> b)': False},
            {'a': a, 'b': a, '(a <=> b)': True},
            {'a': a, 'b': b, '(a <=> b)': False},
        ], rows)

    def test_not_equal(self):
        df = spark.createDataFrame(
            data=[
                ['a'],
                ['b'],
                [None],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
            ]),
        )
        df = df.select('*', F.col('s') != F.lit('a'))
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'s': 'a', '(NOT (s = a))': False},
            {'s': 'b', '(NOT (s = a))': True},
            {'s': None, '(NOT (s = a))': None},
        ], rows)

    def test_is_null_and_is_not_null(self):
        df = spark.createDataFrame(
            data=[
                [None],
                [0],
            ],
            schema=T.StructType([
                T.StructField('i', T.IntegerType()),
            ]),
        )
        df = df.select(
            '*', F.col('i').isNull(), F.col('i').isNotNull(), F.isnull('i'), F.isnotnull('i')
        )

        self.assertEqual(T.StructType([
            T.StructField('i', T.IntegerType()),
            T.StructField('(i IS NULL)', T.BooleanType(), False),
            T.StructField('(i IS NOT NULL)', T.BooleanType(), False),
            T.StructField('(i IS NULL)', T.BooleanType(), False),
            T.StructField('(i IS NOT NULL)', T.BooleanType(), False),
        ]), df.schema)

        rows = df.collect()
        self.assertEqual([
            (None, True, False, True, False),
            (0, False, True, False, True),
        ], rows)

    def test_comparison_operators(self):
        df = spark.createDataFrame(
            data=[
                [None],
                [date(2020, 12, 20)],
                [date(2020, 12, 21)],
                [date(2020, 12, 22)],
            ],
            schema=T.StructType([
                T.StructField('d', T.DateType()),
            ]),
        )
        df = df.select(
            '*',
            F.col('d') < date(2020, 12, 21),
            F.col('d') > date(2020, 12, 21),
            F.col('d') <= date(2020, 12, 21),
            F.col('d') >= date(2020, 12, 21),
            F.col('d').between(date(2020, 12, 21), date(2020, 12, 23)),
        )
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {
                'd': None,
                "(d < DATE '2020-12-21')": None,
                "(d > DATE '2020-12-21')": None,
                "(d <= DATE '2020-12-21')": None,
                "(d >= DATE '2020-12-21')": None,
                "((d >= DATE '2020-12-21') AND (d <= DATE '2020-12-23'))": None,
            },
            {
                'd': date(2020, 12, 20),
                "(d < DATE '2020-12-21')": True,
                "(d > DATE '2020-12-21')": False,
                "(d <= DATE '2020-12-21')": True,
                "(d >= DATE '2020-12-21')": False,
                "((d >= DATE '2020-12-21') AND (d <= DATE '2020-12-23'))": False,
            },
            {
                'd': date(2020, 12, 21),
                "(d < DATE '2020-12-21')": False,
                "(d > DATE '2020-12-21')": False,
                "(d <= DATE '2020-12-21')": True,
                "(d >= DATE '2020-12-21')": True,
                "((d >= DATE '2020-12-21') AND (d <= DATE '2020-12-23'))": True,
            },
            {
                'd': date(2020, 12, 22),
                "(d < DATE '2020-12-21')": False,
                "(d > DATE '2020-12-21')": True,
                "(d <= DATE '2020-12-21')": False,
                "(d >= DATE '2020-12-21')": True,
                "((d >= DATE '2020-12-21') AND (d <= DATE '2020-12-23'))": True,
            },
        ], rows)

    def test_comparison_operator_implicit_casts(self):
        df = spark.createDataFrame([[]])
        df = df.select(
            F.lit(123) > '23',
            F.lit(123) < '2 3',
            F.lit('2030-12-31') <= date(2020, 12, 31),
            F.lit('2020 12 31') >= date(2020, 12, 31),
        )

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([{
            '(123 > 23)': True,
            '(123 < 2 3)': None,
            "(2030-12-31 <= DATE '2020-12-31')": False,
            "(2020 12 31 >= DATE '2020-12-31')": None,
        }], rows)

    def test_size(self):
        df = spark.createDataFrame(
            data=[
                [None, None],
                [[], {}],
                [['a', None, 'b'], {'a': 1, 'b': None}],
            ],
            schema=T.StructType([
                T.StructField('a', T.ArrayType(T.StringType())),
                T.StructField('m', T.MapType(T.StringType(), T.IntegerType())),
            ]),
        )
        df = df.select('*', F.size('a'), F.size('m'))
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'a': None, 'm': None, 'size(a)': -1, 'size(m)': -1},
            {'a': [], 'm': {}, 'size(a)': 0, 'size(m)': 0},
            {'a': ['a', None, 'b'], 'm': {'a': 1, 'b': None}, 'size(a)': 3, 'size(m)': 2},
        ], rows)

    def test_concat_ws(self):
        df = spark.createDataFrame(
            data=[
                [None, None],
                ['a', ['b', None, 'c']],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
                T.StructField('a', T.ArrayType(T.StringType())),
            ]),
        )
        df = df.select('*', F.concat_ws(',', 's', 'a'))
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'s': None, 'a': None, 'concat_ws(,, s, a)': ''},
            {'s': 'a', 'a': ['b', None, 'c'], 'concat_ws(,, s, a)': 'a,b,c'},
        ], rows)

    def test_concat(self):
        df = spark.createDataFrame([[]])
        df = df.select(
            F.concat(),

            F.concat(F.lit(None)),
            F.concat(F.lit(1)),
            F.concat(F.lit('a')),
            F.concat(F.lit([])),

            F.concat(F.lit('a'), F.lit(None)),
            F.concat(F.lit([1]), F.lit(None).cast(T.ArrayType(T.IntegerType(), False))),

            F.concat(F.lit('a'), F.lit(1), F.lit('b')),
            F.concat(F.lit([1]), F.lit([2, None, 3]), F.lit([4])),
        )

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([{
            'concat()': '',

            'concat(NULL)': None,
            'concat(1)': '1',
            'concat(a)': 'a',
            'concat(array())': [],

            'concat(a, NULL)': None,
            'concat(array(1), NULL)': None,

            'concat(a, 1, b)': 'a1b',
            'concat(array(1), array(2, NULL, 3), array(4))': [1, 2, None, 3, 4],
        }], rows)

    def test_array_contains(self):
        df = spark.createDataFrame(
            data=[
                [[], None],
                [[2, 1, 3, 2], ['a', 'bc', 'd']],
            ],
            schema=T.StructType([
                T.StructField('i', T.ArrayType(T.IntegerType()), False),
                T.StructField('s', T.ArrayType(T.StringType())),
            ]),
        )
        df = df.select(
            '*',
            F.array_contains('i', 2),
            F.array_contains('i', 0),
            F.array_contains('s', F.lit(None).cast(T.StringType())),
            F.array_contains('s', 'bc'),
            F.array_contains('s', 'ab'),
        )

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {
                'i': [],
                's': None,
                'array_contains(i, 2)': False,
                'array_contains(i, 0)': False,
                'array_contains(s, CAST(NULL AS STRING))': None,
                'array_contains(s, bc)': None,
                'array_contains(s, ab)': None},
            {
                'i': [2, 1, 3, 2],
                's': ['a', 'bc', 'd'],
                'array_contains(i, 2)': True,
                'array_contains(i, 0)': False,
                'array_contains(s, CAST(NULL AS STRING))': None,
                'array_contains(s, bc)': True,
                'array_contains(s, ab)': False,
            },
        ], rows)

    def test_array_except(self):
        df = spark.createDataFrame(
            data=[
                [None, None],
                [[2, 1, 3, 2], [2, 1, 4]],
            ],
            schema=T.StructType([
                T.StructField('a', T.ArrayType(T.IntegerType())),
                T.StructField('b', T.ArrayType(T.IntegerType())),
            ]),
        )
        df = df.select('*', F.array_except('a', 'b'), F.array_except('a', F.array()))
        rows = [row.asDict() for row in df.collect()]
        utils.sort_lists(rows, 'array_except(a, array())')
        self.assertEqual([
            {
                'a': None, 'b': None,
                'array_except(a, b)': None,
                'array_except(a, array())': None,
            },
            {
                'a': [2, 1, 3, 2], 'b': [2, 1, 4],
                'array_except(a, b)': [3],
                'array_except(a, array())': [1, 2, 3],
            },
        ], rows)

    def test_array_sort(self):
        df = spark.createDataFrame(
            data=[
                [None, None],
                [[2, 1, None, 3, 2], ["10", "1", None, "2"]],
            ],
            schema=T.StructType([
                T.StructField('a', T.ArrayType(T.IntegerType())),
                T.StructField('b', T.ArrayType(T.StringType())),
            ]),
        )
        df = df.select('*', F.array_sort('a').alias('as'), F.array_sort('b').alias('bs'))

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {
                'a': None, 'b': None,
                'as': None, 'bs': None,
            },
            {
                'a': [2, 1, None, 3, 2], 'b': ['10', '1', None, '2'],
                'as': [1, 2, 2, 3, None], 'bs': ['1', '10', '2', None],
            },
        ], rows)

    def test_lit(self):
        df = spark.createDataFrame([[]])
        df = df.select(F.lit([None, [1, 2], [3, 4, 5]]))
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'array(NULL, array(1, 2), array(3, 4, 5))': [None, [1, 2], [3, 4, 5]]},
        ], rows)

    def test_trim(self):
        df = spark.createDataFrame(
            data=[
                [None],
                ['  some text  '],
                [' some text'],
                ['some text '],
                ['some text'],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
            ])
        )
        df = df.select('*', F.trim('s'))
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'s': None, 'trim(s)': None},
            {'s': '  some text  ', 'trim(s)': 'some text'},
            {'s': ' some text', 'trim(s)': 'some text'},
            {'s': 'some text ', 'trim(s)': 'some text'},
            {'s': 'some text', 'trim(s)': 'some text'},
        ], rows)

    def test_arithmetic_operators(self):
        df = spark.createDataFrame([[]])
        df = df.select(
            -F.lit(None).cast(T.DoubleType()),
            -F.lit(-1.5),
            -F.lit('2'),

            F.lit(1) + None,
            F.lit('2') + '1',
            3 + F.lit(2),

            F.lit(1) - None,
            F.lit(2) - '3',
            5 - F.lit(2),

            F.lit(1) * None,
            F.lit('2') * 3,
            5 * F.lit(2),

            F.lit(None) / 1,
            F.lit('3') / 2,
            3 / F.lit(4),
        )

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([{
            '(- CAST(NULL AS DOUBLE))': None,
            '(- -1.5)': 1.5,
            '(- 2)': -2.0,

            '(1 + NULL)': None,
            '(2 + 1)': 3.0,
            '(2 + 3)': 5,

            '(1 - NULL)': None,
            '(2 - 3)': -1.0,
            '(5 - 2)': 3,

            '(1 * NULL)': None,
            '(2 * 3)': 6.0,
            '(2 * 5)': 10,

            '(NULL / 1)': None,
            '(3 / 2)': 1.5,
            '(3 / 4)': 0.75,
        }], rows)

    def test_coalesce(self):
        df = spark.createDataFrame([[]])
        df = df.select(
            F.coalesce(F.lit(None), F.lit(None).cast(T.IntegerType())),
            F.coalesce(F.lit(None), F.lit(1), F.lit(2)),
        )

        self.assertEqual(T.StructType([
            T.StructField('coalesce(NULL, CAST(NULL AS INT))', T.IntegerType(), True),
            T.StructField('coalesce(NULL, 1, 2)', T.IntegerType(), False),
        ]), df.schema)

        rows = df.collect()
        self.assertEqual([(None, 1)], rows)

    def test_contains(self):
        df = spark.createDataFrame(
            data=[
                [None, None],
                [None, ''],
                ['', None],
                ['', ''],
                ['bc', 'abcd'],
                ['abcd', 'bc'],
                ['ab cd', 'bc'],
            ],
            schema=T.StructType([
                T.StructField('str', T.StringType()),
                T.StructField('sub', T.StringType()),
            ]),
        )
        df = df.select('*', F.col('str').contains(F.col('sub')))

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'str': None, 'sub': None, 'contains(str, sub)': None},
            {'str': None, 'sub': '', 'contains(str, sub)': None},
            {'str': '', 'sub': None, 'contains(str, sub)': None},
            {'str': '', 'sub': '', 'contains(str, sub)': True},
            {'str': 'bc', 'sub': 'abcd', 'contains(str, sub)': False},
            {'str': 'abcd', 'sub': 'bc', 'contains(str, sub)': True},
            {'str': 'ab cd', 'sub': 'bc', 'contains(str, sub)': False},
        ], rows)

    def test_startswith(self):
        df = spark.createDataFrame(
            data=[
                [None, None],
                [None, ''],
                ['', None],
                ['', ''],
                ['a', 'ab'],
                ['ab', 'ab'],
                ['abc', 'ab'],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
                T.StructField('p', T.StringType()),
            ]),
        )
        df = df.select('*', F.col('s').startswith(F.col('p')), F.lit(123).startswith(F.lit(12)))

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual(rows, [
            {'s': None, 'p': None, 'startswith(s, p)': None, 'startswith(123, 12)': True},
            {'s': None, 'p': '', 'startswith(s, p)': None, 'startswith(123, 12)': True},
            {'s': '', 'p': None, 'startswith(s, p)': None, 'startswith(123, 12)': True},
            {'s': '', 'p': '', 'startswith(s, p)': True, 'startswith(123, 12)': True},
            {'s': 'a', 'p': 'ab', 'startswith(s, p)': False, 'startswith(123, 12)': True},
            {'s': 'ab', 'p': 'ab', 'startswith(s, p)': True, 'startswith(123, 12)': True},
            {'s': 'abc', 'p': 'ab', 'startswith(s, p)': True, 'startswith(123, 12)': True},
        ])

    def test_endswith(self):
        df = spark.createDataFrame(
            data=[
                [None, None],
                [None, ''],
                ['', None],
                ['', ''],
                ['c', 'bc'],
                ['bc', 'bc'],
                ['abc', 'bc'],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
                T.StructField('e', T.StringType()),
            ]),
        )
        df = df.select('*', F.col('s').endswith(F.col('e')), F.lit(123).endswith(F.lit(23)))

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual(rows, [
            {'s': None, 'e': None, 'endswith(s, e)': None, 'endswith(123, 23)': True},
            {'s': None, 'e': '', 'endswith(s, e)': None, 'endswith(123, 23)': True},
            {'s': '', 'e': None, 'endswith(s, e)': None, 'endswith(123, 23)': True},
            {'s': '', 'e': '', 'endswith(s, e)': True, 'endswith(123, 23)': True},
            {'s': 'c', 'e': 'bc', 'endswith(s, e)': False, 'endswith(123, 23)': True},
            {'s': 'bc', 'e': 'bc', 'endswith(s, e)': True, 'endswith(123, 23)': True},
            {'s': 'abc', 'e': 'bc', 'endswith(s, e)': True, 'endswith(123, 23)': True},
        ])

    def test_lpad(self):
        df = spark.createDataFrame(
            data=[
                [None],
                [''],
                ['1'],
                ['21'],
                ['321'],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
            ]),
        )
        df = df.select('*', F.lpad('s', 2, '0'))

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'s': None, 'lpad(s, 2, 0)': None},
            {'s': '', 'lpad(s, 2, 0)': '00'},
            {'s': '1', 'lpad(s, 2, 0)': '01'},
            {'s': '21', 'lpad(s, 2, 0)': '21'},
            {'s': '321', 'lpad(s, 2, 0)': '32'},
        ], rows)

    def test_length(self):
        df = spark.createDataFrame(
            data=[
                [None],
                [''],
                ['a'],
                [' a b c '],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
            ]),
        )
        df = df.select('*', F.length('s'), F.length(F.lit(123)))

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual(rows, [
            {'s': None, 'length(s)': None, 'length(123)': 3},
            {'s': '', 'length(s)': 0, 'length(123)': 3},
            {'s': 'a', 'length(s)': 1, 'length(123)': 3},
            {'s': ' a b c ', 'length(s)': 7, 'length(123)': 3},
        ])

    def test_arrays_zip(self):
        df = spark.createDataFrame(
            data=[
                [None, None],
                [[], None],
                [None, []],
                [[], []],
                [['a', 'b'], [1, 2]],
                [['c', 'd'], [10, 20, 30]],
            ],
            schema=T.StructType([
                T.StructField('a1', T.ArrayType(T.StringType(), False)),
                T.StructField('a2', T.ArrayType(T.LongType(), False)),
            ]),
        )
        df = df.select('*', F.arrays_zip('a1', 'a2', F.lit([0]), F.lit([False]).alias('b')))

        self.assertEqual(T.StructType([
            T.StructField('a1', T.ArrayType(T.StringType(), False)),
            T.StructField('a2', T.ArrayType(T.LongType(), False)),
            T.StructField('arrays_zip(a1, a2, array(0), array(false) AS b)', T.ArrayType(
                T.StructType([
                    T.StructField('a1', T.StringType()),
                    T.StructField('a2', T.LongType()),
                    T.StructField('2', T.IntegerType()),
                    T.StructField('b', T.BooleanType()),
                ]),
                False,
            )),
        ]), df.schema)
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'a1': None, 'a2': None, 'arrays_zip(a1, a2, array(0), array(false) AS b)': None},
            {'a1': [], 'a2': None, 'arrays_zip(a1, a2, array(0), array(false) AS b)': None},
            {'a1': None, 'a2': [], 'arrays_zip(a1, a2, array(0), array(false) AS b)': None},
            {
                'a1': [], 'a2': [],
                'arrays_zip(a1, a2, array(0), array(false) AS b)': [
                    (None, None, 0, False),
                ],
            },
            {
                'a1': ['a', 'b'], 'a2': [1, 2],
                'arrays_zip(a1, a2, array(0), array(false) AS b)': [
                    ('a', 1, 0, False),
                    ('b', 2, None, None),
                ],
            },
            {
                'a1': ['c', 'd'], 'a2': [10, 20, 30],
                'arrays_zip(a1, a2, array(0), array(false) AS b)': [
                    ('c', 10, 0, False),
                    ('d', 20, None, None),
                    (None, 30, None, None),
                ],
            },
        ], rows)

    def test_aggregate(self):
        df = spark.createDataFrame(
            data=[
                [None, None],
                [None, 0],
                [[], None],
                [[], 0],
                [[1, -2, 3], 100],
            ],
            schema=T.StructType([
                T.StructField('a', T.ArrayType(T.IntegerType())),
                T.StructField('i', T.IntegerType()),
            ]),
        )
        df = df.select(
            '*',
            F.aggregate(
                'a',
                'i',
                lambda acc, x: acc + F.abs(x),
                lambda acc: acc.cast(T.StringType()),
            ),
            F.aggregate(
                'a',
                'i',
                lambda acc, x: acc + F.abs(x),
            ),
            F.aggregate(
                F.lit([1, 2, 3]),
                F.lit(0),
                lambda acc, x: acc + F.abs(x),
                lambda acc: acc.cast(T.StringType()),
            ),
        )

        var = 'namedlambdavariable()'
        merge = f'lambdafunction(({var} + abs({var})), {var}, {var})'
        finish = f'lambdafunction(CAST({var} AS STRING), {var})'
        default = f'lambdafunction({var}, {var})'

        self.assertEqual(T.StructType([
            T.StructField('a', T.ArrayType(T.IntegerType())),
            T.StructField('i', T.IntegerType()),
            T.StructField(f'aggregate(a, i, {merge}, {finish})', T.StringType()),
            T.StructField(f'aggregate(a, i, {merge}, {default})', T.IntegerType()),
            T.StructField(f'aggregate(array(1, 2, 3), 0, {merge}, {finish})', T.StringType(), False),
        ]), df.schema)

        rows = df.collect()
        self.assertEqual([
            (None, None, None, None, '6'),
            (None, 0, None, None, '6'),
            ([], None, None, None, '6'),
            ([], 0, '0', 0, '6'),
            ([1, -2, 3], 100, '106', 106, '6'),
        ], rows)

    def test_getField(self):
        df = spark.createDataFrame(
            data=[
                [None],
                [(None, None, None)],
                [('abc', 123, {'s': 'def', 'i': 456})],
            ],
            schema=T.StructType([
                T.StructField('t', T.StructType([
                    T.StructField('s', T.StringType()),
                    T.StructField('i', T.IntegerType()),
                    T.StructField('t', T.StructType([
                        T.StructField('s', T.StringType(), False),
                        T.StructField('i', T.IntegerType(), False),
                    ])),
                ])),
            ]),
        )
        df = df.select(
            '*',
            F.col('t').getField('s'),
            F.col('t').getField('i'),
            F.col('t.i'),
            F.col('t').getField('t'),
            F.col('t').getField('t').getField('s'),
            F.col('t.t.s'),
            F.col('t').getField('t').getField('i'),
        )

        self.assertEqual(T.StructType([
            T.StructField('t', T.StructType([
                T.StructField('s', T.StringType(), True),
                T.StructField('i', T.IntegerType(), True),
                T.StructField('t', T.StructType([
                    T.StructField('s', T.StringType(), False),
                    T.StructField('i', T.IntegerType(), False),
                ]), True),
            ]), True),
            T.StructField('t.s', T.StringType(), True),
            T.StructField('t.i', T.IntegerType(), True),
            T.StructField('i', T.IntegerType(), True),
            T.StructField('t.t', T.StructType([
                T.StructField('s', T.StringType(), False),
                T.StructField('i', T.IntegerType(), False),
            ]), True),
            T.StructField('t.t.s', T.StringType(), True),
            T.StructField('s', T.StringType(), True),
            T.StructField('t.t.i', T.IntegerType(), True),
        ]), df.schema)

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {
                't': None,
                't.s': None, 't.i': None, 'i': None, 't.t': None,
                't.t.s': None, 's': None, 't.t.i': None},
            {
                't': (None, None, None),
                't.s': None, 't.i': None, 'i': None, 't.t': None,
                't.t.s': None, 's': None, 't.t.i': None,
            },
            {
                't': ('abc', 123, ('def', 456)),
                't.s': 'abc', 't.i': 123, 'i': 123, 't.t': ('def', 456),
                't.t.s': 'def', 's': 'def', 't.t.i': 456,
            }
        ], rows)

    def test_getItem(self):
        df = spark.createDataFrame(
            data=[
                [None, None, None],
                [('abc', 123), [1, 2], {'a': 0.5, 'b': 1.2}],
            ],
            schema=T.StructType([
                T.StructField('t', T.StructType([
                    T.StructField('s', T.StringType()),
                    T.StructField('i', T.IntegerType()),
                ])),
                T.StructField('a', T.ArrayType(T.IntegerType())),
                T.StructField('m', T.MapType(T.StringType(), T.FloatType())),
            ]),
        )
        df = df.select(
            '*',
            F.col('t')['s'],
            F.col('t').getItem('i'),
            F.col('a')[-1],
            F.col('a').getItem(0),
            F.col('a')[1],
            F.col('a').getItem(2),
            F.col('m')['a'],
            F.col('m').getItem('b'),
            F.col('m')['c'],
        )

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {
                't': None, 'a': None, 'm': None,
                't.s': None, 't.i': None,
                'a[-1]': None, 'a[0]': None, 'a[1]': None, 'a[2]': None,
                'm[a]': None, 'm[b]': None, 'm[c]': None,
            },
            {  # todo 1.2000000476837158
                't': T.Row(s='abc', i=123), 'a': [1, 2], 'm': {'a': 0.5, 'b': 1.2000000476837158},
                't.s': 'abc', 't.i': 123,
                'a[-1]': None, 'a[0]': 1, 'a[1]': 2, 'a[2]': None,
                'm[a]': 0.5, 'm[b]': 1.2000000476837158, 'm[c]': None,
            },
        ], rows)

    def test_struct(self):
        df = spark.createDataFrame(
            data=[
                [None, None, None],
                ['abc', 123, T.Row(s='def', i=456)],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
                T.StructField('i', T.IntegerType()),
                T.StructField('t', T.StructType([
                    T.StructField('s', T.StringType(), False),
                    T.StructField('i', T.IntegerType(), False),
                ])),
            ]),
        )
        df = df.select('*', F.struct('s', 'i', 't', F.lit('ghi'), F.lit(789), F.lit(None)))

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {
                's': None, 'i': None, 't': None,
                'struct(s, i, t, ghi, 789, NULL)': T.Row(
                    s=None, i=None, t=None, col4='ghi', col5=789, col6=None
                ),
            },
            {
                's': 'abc', 'i': 123, 't': T.Row(s='def', i=456),
                'struct(s, i, t, ghi, 789, NULL)': T.Row(
                    s='abc', i=123, t=T.Row(s='def', i=456), col4='ghi', col5=789, col6=None
                ),
            },
        ], rows)

    def test_current_timestamp(self):
        df = spark.createDataFrame([[]])
        df = df.select(F.current_timestamp())

        self.assertEqual(T.StructType([
            T.StructField('current_timestamp()', T.TimestampType(), False),
        ]), df.schema)

        from datetime import timedelta
        start = datetime.now() - timedelta(microseconds=500)
        rows = df.collect()
        end = datetime.now() + timedelta(microseconds=500)

        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual(1, len(row))
        value = row[0]
        self.assertTrue(start <= value <= end, f"{start} <= {value} <= {end}")

    def test_hash(self):
        df = spark.createDataFrame([[]])
        df = df.select(F.hash(F.lit(None), F.lit(123), F.lit('abc')))

        self.assertEqual(T.StructType([
            T.StructField('hash(NULL, 123, abc)', T.IntegerType(), False),
        ]), df.schema)

        rows = df.collect()
        self.assertEqual([(12585979005506212136,)], rows)

    def test_upper(self):
        df = spark.createDataFrame([[]])
        df = df.select(
            F.upper(F.lit(None)),
            F.upper(F.lit(False)),
            F.upper(F.lit('aBc1DeF2')),
        )
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'upper(NULL)': None, 'upper(false)': 'FALSE', 'upper(aBc1DeF2)': 'ABC1DEF2'},
        ], rows)

    def test_lower(self):
        df = spark.createDataFrame([[]])
        df = df.select(
            F.lower(F.lit(None)),
            F.lower(F.lit(False)),
            F.lower(F.lit('aBc1DeF2')),
        )
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'lower(NULL)': None, 'lower(false)': 'false', 'lower(aBc1DeF2)': 'abc1def2'},
        ], rows)

    def test_substring(self):
        df = spark.createDataFrame([[]])
        df = df.select(
            F.substring(F.lit(None), 1, 1),
            F.substring(F.lit('abcd'), 0, 2),
            F.substring(F.lit('abcd'), 1, 2),
            F.substring(F.lit('abcd'), -1, 1),
            F.substring(F.lit(1234), 2, 4),
            F.substring(F.lit(1234), 2, -2),
            F.substring(F.lit(1234), 5, 1),
            F.substring(F.lit(1234), -5, 1),
        )
        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([{
            'substring(NULL, 1, 1)': None,
            'substring(abcd, 0, 2)': 'ab',
            'substring(abcd, 1, 2)': 'ab',
            'substring(abcd, -1, 1)': 'd',
            'substring(1234, 2, 4)': '234',
            'substring(1234, 2, -2)': '',
            'substring(1234, 5, 1)': '',
            'substring(1234, -5, 1)': '',
        }], rows)

    def test_abs(self):
        df = spark.createDataFrame(
            data=[
                [None, None, None],
                [1, 2.5, '+123'],
                [-1, -2.5, '-123'],
            ],
            schema=T.StructType([
                T.StructField('l', T.LongType()),
                T.StructField('d', T.DoubleType()),
                T.StructField('s', T.StringType()),
            ]),
        )
        df = df.select('*', F.abs('l'), F.abs('d'), F.abs('s'))

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'l': None, 'd': None, 's': None, 'abs(l)': None, 'abs(d)': None, 'abs(s)': None},
            {'l': 1, 'd': 2.5, 's': '+123', 'abs(l)': 1, 'abs(d)': 2.5, 'abs(s)': 123.0},
            {'l': -1, 'd': -2.5, 's': '-123', 'abs(l)': 1, 'abs(d)': 2.5, 'abs(s)': 123.0},
        ], rows)

    def test_round(self):
        df = spark.createDataFrame(
            data=[
                [None, None, None],
                [123456789, 1234.5678, '1234.5678'],
            ],
            schema=T.StructType([
                T.StructField('l', T.LongType()),
                T.StructField('d', T.DoubleType()),
                T.StructField('s', T.StringType()),
            ]),
        )
        df = df.select('*', F.round('l', -3), F.round('d', 3), F.round('s'))

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {
                'l': None, 'd': None, 's': None,
                'round(l, -3)': None, 'round(d, 3)': None, 'round(s, 0)': None,
            },
            {
                'l': 123456789, 'd': 1234.5678, 's': '1234.5678',
                'round(l, -3)': 123457000, 'round(d, 3)': 1234.568, 'round(s, 0)': 1235.0,
            },
        ], rows)

    def test_create_map(self):
        df = spark.createDataFrame(
            data=[
                ['', None, ' ', None],
                ['a', 1, 'b', 2],
                ['c', 3, 'd', 4],
            ],
            schema=T.StructType([
                T.StructField('k1', T.StringType()),
                T.StructField('v1', T.IntegerType()),
                T.StructField('k2', T.StringType()),
                T.StructField('v2', T.IntegerType()),
            ]),
        )
        df = df.select(
            '*',
            F.create_map(),
            F.create_map('k1', 'v1', 'k2', 'v2'),
            F.create_map(['k1', 'v1']),
        )

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {
                'k1': '', 'v1': None, 'k2': ' ', 'v2': None,
                'map()': {},
                'map(k1, v1, k2, v2)': {'': None, ' ': None},
                'map(k1, v1)': {'': None},
            },
            {
                'k1': 'a', 'v1': 1, 'k2': 'b', 'v2': 2,
                'map()': {},
                'map(k1, v1, k2, v2)': {'a': 1, 'b': 2},
                'map(k1, v1)': {'a': 1},
            },
            {
                'k1': 'c', 'v1': 3, 'k2': 'd', 'v2': 4,
                'map()': {},
                'map(k1, v1, k2, v2)': {'c': 3, 'd': 4},
                'map(k1, v1)': {'c': 3},
            },
        ], rows)

    def test_map_keys_and_map_values(self):
        df = spark.createDataFrame(
            data=[
                [None],
                [{'a': 0.5, 'b': 1.2}],
            ],
            schema=T.StructType([
                T.StructField('m', T.MapType(T.StringType(), T.DoubleType())),
            ]),
        )
        df = df.select('*', F.map_keys('m'), F.map_values('m'))

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'m': None, 'map_keys(m)': None, 'map_values(m)': None},
            {'m': {'a': 0.5, 'b': 1.2}, 'map_keys(m)': ['a', 'b'], 'map_values(m)': [0.5, 1.2]},
        ], rows)

    def test_map_concat(self):
        df = spark.createDataFrame([[]])
        map_1 = F.create_map(F.lit('a'), F.lit(0.5), F.lit('b'), F.lit(1.2))
        map_2 = F.create_map(F.lit('c'), F.lit(1.5), F.lit('d'), F.lit(2.3))
        df = df.select(
            F.map_concat(),
            F.map_concat(F.create_map().cast('map<string,double>'), map_1),
            F.map_concat(map_1, map_2),
        )

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([{
            'map_concat()': {},
            'map_concat(map(), map(a, 0.5, b, 1.2))': {'a': 0.5, 'b': 1.2},
            'map_concat(map(a, 0.5, b, 1.2), map(c, 1.5, d, 2.3))': {'a': 0.5, 'b': 1.2, 'c': 1.5, 'd': 2.3},
        }], rows)

    def test_nan_inf(self):
        df = spark.createDataFrame(
            data=[
                [None, None],
                [1.5, '1.5'],
                [math.nan, 'NaN'],
                [math.inf, 'Infinity'],
            ],
            schema=T.StructType([
                T.StructField('d', T.DoubleType()),
                T.StructField('s', T.StringType()),
            ]),
        )
        df = df.select(
            '*',
            F.col('d').isin(math.nan),
            F.col('d').isin(math.inf),
            F.col('d').cast(T.StringType()),
            F.col('s').cast(T.DoubleType()),
        )

        self.assertEqual(T.StructType([
            T.StructField('d', T.DoubleType()),
            T.StructField('s', T.StringType()),
            T.StructField('(d IN (NaN))', T.BooleanType()),
            T.StructField('(d IN (Infinity))', T.BooleanType()),
            T.StructField('CAST(d AS STRING)', T.StringType()),
            T.StructField('CAST(s AS DOUBLE)', T.DoubleType()),
        ]), df.schema)

        rows = [
            tuple(
                str(v) if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
                for v in row
            )
            for row in df.collect()
        ]
        self.assertEqual([
            (None, None, None, None, None, None),
            (1.5, '1.5', False, False, '1.5', 1.5),
            ('nan', 'NaN', True, False, 'nan', 'nan'),  # For pyspark, [-2] = 'NaN'
            ('inf', 'Infinity', False, True, 'inf', 'inf'),  # For pyspark, [-2] = 'Infinity'
        ], rows)

    def test_date_add_and_date_sub(self):
        df = spark.createDataFrame(
            data=[
                [None, None, None],
                [date(2020, 12, 31), '2020-12-01', 1],
                [date(2020, 2, 29), '20201231', -1],
            ],
            schema=T.StructType([
                T.StructField('d', T.DateType()),
                T.StructField('s', T.StringType()),
                T.StructField('i', T.IntegerType()),
            ]),
        )
        df = df.select(
            '*',
            F.date_add('d', 'i'),
            F.date_add('s', 'i'),
            F.date_sub('d', 'i'),
            F.date_sub('s', 'i'),
        )

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {
                'd': None, 's': None, 'i': None,
                'date_add(d, i)': None,
                'date_add(s, i)': None,
                'date_sub(d, i)': None,
                'date_sub(s, i)': None,
            },
            {
                'd': date(2020, 12, 31), 's': '2020-12-01', 'i': 1,
                'date_add(d, i)': date(2021, 1, 1),
                'date_add(s, i)': date(2020, 12, 2),
                'date_sub(d, i)': date(2020, 12, 30),
                'date_sub(s, i)': date(2020, 11, 30),
            },
            {
                'd': date(2020, 2, 29), 's': '20201231', 'i': -1,
                'date_add(d, i)': date(2020, 2, 28),
                'date_add(s, i)': None,
                'date_sub(d, i)': date(2020, 3, 1),
                'date_sub(s, i)': None,
            },
        ], rows)

    def test_dayofweek(self):
        df = spark.createDataFrame(
            data=[
                [None, None],
                [date(2020, 12, 31), '2020-12-27'],
                [date(2020, 12, 30), '2020-12-26'],
                [date(2020, 12, 29), '2020-12-25'],
                [date(2020, 12, 28), '2020 12 31'],
            ],
            schema=T.StructType([
                T.StructField('d', T.DateType()),
                T.StructField('s', T.StringType()),
            ]),
        )
        df = df.select(
            '*',
            F.dayofweek('d'),
            F.dayofweek('s'),
        )

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'d': None, 's': None, 'dayofweek(d)': None, 'dayofweek(s)': None},
            {'d': date(2020, 12, 31), 's': '2020-12-27', 'dayofweek(d)': 5, 'dayofweek(s)': 1},
            {'d': date(2020, 12, 30), 's': '2020-12-26', 'dayofweek(d)': 4, 'dayofweek(s)': 7},
            {'d': date(2020, 12, 29), 's': '2020-12-25', 'dayofweek(d)': 3, 'dayofweek(s)': 6},
            {'d': date(2020, 12, 28), 's': '2020 12 31', 'dayofweek(d)': 2, 'dayofweek(s)': None},
        ], rows)

    def test_from_json(self):
        df = spark.createDataFrame(
            data=[
                [None],
                ['null'],
                ['{"a":1,"b":2}'],
            ],
            schema=T.StructType([
                T.StructField('s', T.StringType()),
            ]),
        )
        data_type = T.MapType(T.StringType(), T.IntegerType())
        type_string = data_type.simpleString()
        df = df.select('*', F.from_json('s', type_string), F.from_json('s', data_type).alias('m'))

        rows = [row.asDict() for row in df.collect()]
        self.assertEqual([
            {'s': None, 'entries': None, 'm': None},
            {'s': 'null', 'entries': None, 'm': None},
            {'s': '{"a":1,"b":2}', 'entries': {'a': 1, 'b': 2}, 'm': {'a': 1, 'b': 2}},
        ], rows)
