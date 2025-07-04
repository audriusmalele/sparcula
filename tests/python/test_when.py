from unittest import TestCase

from sparcula.pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.getOrCreate()


class TestWhen(TestCase):
    def test_type_and_otherwise_validation(self):
        literal = F.lit(1)
        when = F.when(F.col('condition'), 'true').otherwise('false')
        with self.assertRaises(Exception):
            literal.otherwise(1)
        with self.assertRaises(Exception):
            literal.when(F.col('condition'), 'true')
        with self.assertRaises(Exception):
            when.otherwise('bad')
        with self.assertRaises(Exception):
            when.when(F.col('condition'), 'bad')

    def test_column_alias_and_str(self):
        w1 = F.when(F.lit(False), F.lit(0))
        w2 = w1.when(F.lit(True), F.lit(1))
        w3 = w2.otherwise(F.lit(None))
        w4 = w3.alias('a')
        self.assertEqual("Column<'CASE WHEN false THEN 0 END'>", str(w1))
        self.assertEqual("Column<'CASE WHEN false THEN 0 WHEN true THEN 1 END'>", str(w2))
        self.assertEqual("Column<'CASE WHEN false THEN 0 WHEN true THEN 1 ELSE NULL END'>", str(w3))
        self.assertEqual("Column<'CASE WHEN false THEN 0 WHEN true THEN 1 ELSE NULL END AS a'>", str(w4))
