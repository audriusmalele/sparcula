from unittest import TestCase

import utils

from sparcula.pyspark.sql import SparkSession, Window, functions as F, types as T

spark = SparkSession.builder.getOrCreate()


class TestAggregations(TestCase):
    def test_collect_list_and_collect_set(self):
        df = spark.createDataFrame(
            data=[
                ['a', '1', 1],
                ['a', '2', 2],
                ['a', '2', 2],
                ['a', None, None],
                ['b', None, None],
            ],
            schema=T.StructType([
                T.StructField('k', T.StringType()),
                T.StructField('s', T.StringType()),
                T.StructField('i', T.IntegerType()),
            ]),
        )
        df = df.groupBy('k').agg(
            F.collect_list('s'), F.collect_set('s'), F.collect_list('i'), F.collect_set('i')
        )
        rows = [row.asDict() for row in df.collect()]
        utils.sort_lists(rows, 'collect_set(s)', 'collect_set(i)')
        rows = sorted(rows, key=lambda row: row['k'])
        self.assertEqual([
            {
                'k': 'a',
                'collect_list(s)': ['1', '2', '2'],
                'collect_set(s)': ['1', '2'],
                'collect_list(i)': [1, 2, 2],
                'collect_set(i)': [1, 2],
            },
            {
                'k': 'b',
                'collect_list(s)': [],
                'collect_set(s)': [],
                'collect_list(i)': [],
                'collect_set(i)': [],
            },
        ], rows)

    def test_count_and_approx_count_distinct_and_row_number(self):
        df = spark.createDataFrame(
            data=[
                ['a', None],
                ['a', None],
                ['a', 1],
                ['a', 2],
                ['a', 2],
                ['a', 3],
                ['b', -3],
                ['b', -2],
                ['b', -1],
            ],
            schema=T.StructType([
               T.StructField('p', T.StringType()),
               T.StructField('n', T.IntegerType()),
            ]),
        )
        w = Window.partitionBy('p')
        df = df.select(
            '*',
            F.count('*').over(w).alias('c'),
            F.approx_count_distinct('n').over(w).alias('acd'),
            F.row_number().over(w.orderBy('n')).alias('rn'),
        )

        rows = df.collect()
        rows = sorted(rows, key=lambda row: (row['p'], row['rn']))
        self.assertEqual([
            T.Row(p='a', n=None, c=6, acd=3, rn=1),
            T.Row(p='a', n=None, c=6, acd=3, rn=2),
            T.Row(p='a', n=1, c=6, acd=3, rn=3),
            T.Row(p='a', n=2, c=6, acd=3, rn=4),
            T.Row(p='a', n=2, c=6, acd=3, rn=5),
            T.Row(p='a', n=3, c=6, acd=3, rn=6),
            T.Row(p='b', n=-3, c=3, acd=3, rn=1),
            T.Row(p='b', n=-2, c=3, acd=3, rn=2),
            T.Row(p='b', n=-1, c=3, acd=3, rn=3),
        ], rows)

    def test_first_and_last(self):
        df = spark.createDataFrame(
            data=[
                ['a', 2, 2, 'b', 12],
                ['a', 0, None, None, 10],
                ['a', 1, 1, 'a', 11],
                ['b', 5, None, 'd', 15],
                ['b', 3, 3, None, 13],
                ['b', 4, 4, 'c', 14],
            ],
            schema=T.StructType([
                T.StructField('p', T.StringType()),
                T.StructField('o', T.IntegerType()),
                T.StructField('i', T.IntegerType()),
                T.StructField('s', T.StringType()),
                T.StructField('l', T.LongType(), False),
            ]),
        )
        w = (
            Window.partitionBy('p')
            .orderBy('o')
            .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
        df = df.select(
            '*',

            F.first('i').over(w).alias('fi'),
            F.first('i', ignorenulls=True).over(w).alias('fin'),
            F.first('s').over(w).alias('fs'),
            F.first('s', ignorenulls=True).over(w).alias('fsn'),
            F.first('l').over(w).alias('fl'),

            F.last('i').over(w).alias('li'),
            F.last('i', ignorenulls=True).over(w).alias('lin'),
            F.last('s').over(w).alias('ls'),
            F.last('s', ignorenulls=True).over(w).alias('lsn'),
            F.last('l').over(w).alias('ll'),
        )

        rows = df.collect()
        rows = sorted(rows, key=lambda row: row['o'])
        self.assertEqual([
            T.Row(
                p='a', o=0, i=None, s=None, l=10,
                fi=None, fin=1, fs=None, fsn='a', fl=10,
                li=2, lin=2, ls='b', lsn='b', ll=12,
            ),
            T.Row(
                p='a', o=1, i=1, s='a', l=11,
                fi=None, fin=1, fs=None, fsn='a', fl=10,
                li=2, lin=2, ls='b', lsn='b', ll=12,
            ),
            T.Row(
                p='a', o=2, i=2, s='b', l=12,
                fi=None, fin=1, fs=None, fsn='a', fl=10,
                li=2, lin=2, ls='b', lsn='b', ll=12,
            ),
            T.Row(
                p='b', o=3, i=3, s=None, l=13,
                fi=3, fin=3, fs=None, fsn='c', fl=13,
                li=None, lin=4, ls='d', lsn='d', ll=15,
            ),
            T.Row(
                p='b', o=4, i=4, s='c', l=14,
                fi=3, fin=3, fs=None, fsn='c', fl=13,
                li=None, lin=4, ls='d', lsn='d', ll=15,
            ),
            T.Row(
                p='b', o=5, i=None, s='d', l=15,
                fi=3, fin=3, fs=None, fsn='c', fl=13,
                li=None, lin=4, ls='d', lsn='d', ll=15,
            ),
        ], rows)

    def test_sum(self):
        df = spark.createDataFrame(
            data=[
                ['a', 0, None, None, None],
                ['a', 1, 1, 1.0, '1'],
                ['a', 2, 2, 2.0, '2'],
                ['b', 3, None, None, None],
                ['b', 4, 4, 4.0, '4'],
                ['b', 5, 5, 5.0, '5'],
            ],
            schema=T.StructType([
               T.StructField('p', T.StringType()),
               T.StructField('o', T.IntegerType()),
               T.StructField('i', T.IntegerType()),
               T.StructField('f', T.FloatType()),
               T.StructField('s', T.StringType()),
            ]),
        )
        w = Window.partitionBy('p').orderBy('o')
        df = df.select(
            '*',
            F.sum('i').over(w).alias('si'),
            F.sum('f').over(w).alias('sf'),
            F.sum('s').over(w).alias('ss'),
        )
        rows = df.collect()
        rows = sorted(rows, key=lambda row: row['o'])
        self.assertEqual([
            T.Row(p='a', o=0, i=None, f=None, s=None, si=None, sf=None, ss=None),
            T.Row(p='a', o=1, i=1, f=1.0, s='1', si=1, sf=1.0, ss=1.0),
            T.Row(p='a', o=2, i=2, f=2.0, s='2', si=3, sf=3.0, ss=3.0),
            T.Row(p='b', o=3, i=None, f=None, s=None, si=None, sf=None, ss=None),
            T.Row(p='b', o=4, i=4, f=4.0, s='4', si=4, sf=4.0, ss=4.0),
            T.Row(p='b', o=5, i=5, f=5.0, s='5', si=9, sf=9.0, ss=9.0),
        ], rows)

    def test_rank_and_dense_rank(self):
        df = spark.createDataFrame(
            data=[
                ['a', None],
                ['a', None],
                ['a', 1],
                ['a', 2],
                ['a', 2],
                ['a', 3],
                ['b', -3],
                ['b', -2],
                ['b', -1],
            ],
            schema=T.StructType([
               T.StructField('p', T.StringType()),
               T.StructField('n', T.IntegerType()),
            ]),
        )
        w = Window.partitionBy(['p']).orderBy(['n'])
        df = df.select(
            '*',
            F.rank().over(w).alias('r'),
            F.dense_rank().over(w).alias('dr'),
        )

        rows = df.collect()
        rows = sorted(rows, key=lambda row: (row['p'], row['n'] or -999))
        self.assertEqual([
            T.Row(p='a', n=None, r=1, dr=1),
            T.Row(p='a', n=None, r=1, dr=1),
            T.Row(p='a', n=1, r=3, dr=2),
            T.Row(p='a', n=2, r=4, dr=3),
            T.Row(p='a', n=2, r=4, dr=3),
            T.Row(p='a', n=3, r=6, dr=4),
            T.Row(p='b', n=-3, r=1, dr=1),
            T.Row(p='b', n=-2, r=2, dr=2),
            T.Row(p='b', n=-1, r=3, dr=3),
        ], rows)

    def test_min_by_and_max_by(self):
        df = spark.createDataFrame(
            data=[
                ['a', None, None, 3],
                ['a', 3, 'b', 2],
                ['a', 1, 'c', None],
                ['a', 2, 'a', 0],
                ['b', -1, 'a', -1],
                ['b', -3, 'aa', -2],
                ['b', -2, '', -3],
            ],
            schema=T.StructType([
                T.StructField('p', T.StringType()),
                T.StructField('i', T.IntegerType()),
                T.StructField('s', T.StringType()),
                T.StructField('v', T.IntegerType()),
            ]),
        )
        df = df.groupBy('p').agg(
            F.min_by('v', 'i'),
            F.max_by('v', 'i'),
            F.min_by('v', 's'),
            F.max_by('v', 's'),
        )

        rows = [row.asDict() for row in df.collect()]
        rows = sorted(rows, key=lambda row: row['p'])
        self.assertEqual([
            {
                'p': 'a',
                'min_by(v, i)': None, 'max_by(v, i)': 2,
                'min_by(v, s)': 0, 'max_by(v, s)': None,
            },
            {
                'p': 'b',
                'min_by(v, i)': -2, 'max_by(v, i)': -1,
                'min_by(v, s)': -3, 'max_by(v, s)': -2,
            },
        ], rows)

    def test_min_and_max(self):
        df = spark.createDataFrame(
            data=[
                ['a', None, None, 1],
                ['a', 3, 'b', 3],
                ['a', 1, 'c', 0],
                ['a', 2, 'a', 2],
                ['b', -1, 'a', -2],
                ['b', -3, 'aa', -1],
                ['b', -2, '', -3],
            ],
            schema=T.StructType([
                T.StructField('p', T.StringType()),
                T.StructField('i', T.IntegerType()),
                T.StructField('s', T.StringType()),
                T.StructField('l', T.LongType(), False),
            ]),
        )
        df = df.groupBy('p').agg(
            F.min('i'), F.max('i'),
            F.min('s'), F.max('s'),
            F.min('l'), F.max('l'),
        )

        rows = [row.asDict() for row in df.collect()]
        rows = sorted(rows, key=lambda row: row['p'])
        self.assertEqual([
            {
                'p': 'a',
                'min(i)': 1, 'max(i)': 3,
                'min(s)': 'a', 'max(s)': 'c',
                'min(l)': 0, 'max(l)': 3,
            },
            {
                'p': 'b',
                'min(i)': -3, 'max(i)': -1,
                'min(s)': '', 'max(s)': 'aa',
                'min(l)': -3, 'max(l)': -1,
            },
        ], rows)

    def test_lag_and_lead(self):
        df = spark.createDataFrame(
            data=[
                ['a', 1, 's1', 10, 100],
                ['a', 2, 's2', 20, 200],
                ['a', 3, 's3', 30, 300],
                ['b', 6, 's6', 60, 600],
                ['b', 5, 's5', 50, 500],
                ['b', 4, 's4', 40, 400],
            ],
            schema=T.StructType([
                T.StructField('p', T.StringType()),
                T.StructField('o', T.IntegerType()),
                T.StructField('s', T.StringType()),
                T.StructField('i', T.IntegerType()),
                T.StructField('l', T.LongType(), False),
            ]),
        )
        w = Window.partitionBy('p').orderBy('o')
        df = df.select(
            '*',
            F.lag(F.concat(F.lit('_'), 's'), 1, 'default').over(w).alias('lag_s'),
            F.lead(F.concat(F.lit('_'), 's'), -1, 'default').over(w).alias('lead_s'),
            F.lag(F.col('i') * 10, 0, None).over(w).alias('lag_i'),
            F.lead(F.col('i') * 10, 0, None).over(w).alias('lead_i'),
            F.lag(F.col('l') * 10, -1, -1).over(w).alias('lag_l'),
            F.lead(F.col('l') * 10, 1, -1).over(w).alias('lead_l'),
        )

        self.assertEqual(T.StructType([
            T.StructField('p', T.StringType()),
            T.StructField('o', T.IntegerType()),
            T.StructField('s', T.StringType()),
            T.StructField('i', T.IntegerType()),
            T.StructField('l', T.LongType(), False),
            T.StructField('lag_s', T.StringType()),
            T.StructField('lead_s', T.StringType()),
            T.StructField('lag_i', T.IntegerType()),
            T.StructField('lead_i', T.IntegerType()),
            T.StructField('lag_l', T.LongType(), False),
            T.StructField('lead_l', T.LongType(), False),
        ]), df.schema)
        rows = [row.asDict() for row in df.collect()]
        rows = sorted(rows, key=lambda row: row['o'])
        self.assertEqual([
            {
                'p': 'a', 'o': 1, 's': 's1', 'i': 10, 'l': 100,
                'lag_s': 'default', 'lag_i': 100, 'lag_l': 2000,
                'lead_s': 'default', 'lead_i': 100, 'lead_l': 2000,
            },
            {
                'p': 'a', 'o': 2, 's': 's2', 'i': 20, 'l': 200,
                'lag_s': '_s1', 'lag_i': 200, 'lag_l': 3000,
                'lead_s': '_s1', 'lead_i': 200, 'lead_l': 3000,
            },
            {
                'p': 'a', 'o': 3, 's': 's3', 'i': 30, 'l': 300,
                'lag_s': '_s2', 'lag_i': 300, 'lag_l': -1,
                'lead_s': '_s2', 'lead_i': 300, 'lead_l': -1,
            },
            {
                'p': 'b', 'o': 4, 's': 's4', 'i': 40, 'l': 400,
                'lag_s': 'default', 'lag_i': 400, 'lag_l': 5000,
                'lead_s': 'default', 'lead_i': 400, 'lead_l': 5000,
            },
            {
                'p': 'b', 'o': 5, 's': 's5', 'i': 50, 'l': 500,
                'lag_s': '_s4', 'lag_i': 500, 'lag_l': 6000,
                'lead_s': '_s4', 'lead_i': 500, 'lead_l': 6000,
            },
            {
                'p': 'b', 'o': 6, 's': 's6', 'i': 60, 'l': 600,
                'lag_s': '_s5', 'lag_i': 600, 'lag_l': -1,
                'lead_s': '_s5', 'lead_i': 600, 'lead_l': -1,
            },
        ], rows)
