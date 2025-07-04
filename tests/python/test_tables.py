from unittest import TestCase

from sparcula.pyspark.sql import SparkSession, types as T

spark = SparkSession.builder.getOrCreate()


class TestTables(TestCase):
    def _test_parquet(self):
        from pyspark.sql import SparkSession as Session
        s = Session.builder.getOrCreate()
        data = [
            (None, None, None, None, None, None),
            ('abc', 123, 123.5, True, (1, 'a'), [{'a': 1, 'b': 2}, {'c': 3, 'd': 4}]),
        ]
        df = s.createDataFrame(data)
        df.printSchema()
        print(df.collect())
        df.write.parquet('test_parquet_in', compression='uncompressed')
        schema = df.schema

        df = spark.read.schema(schema.simpleString()).parquet('test_parquet_in')
        df = df.select('*')
        df.printSchema()
        print(df.collect())
        df.write.parquet('test_parquet_out', compression='uncompressed')

        df = s.read.parquet('test_parquet_out')
        df.printSchema()
        rows = df.collect()
        rows = sorted(rows, key=lambda row: row[0] or '')
        print(rows)

        self.assertEqual(schema, df.schema)
        self.assertEqual(data, rows)

    def _test_collect(self):
        from pyspark.sql import SparkSession as Session
        s = Session.builder.getOrCreate()
        data = [
            (None, None, None, None, None, None),
            ('abc', 123, 123.5, True, (1, 'a'), [{'a': 1, 'b': 2}, {'c': 3, 'd': 4}]),
        ]
        df = s.createDataFrame(data)
        in_schema = df.schema
        in_rows = df.collect()

        df = spark.createDataFrame(in_rows, in_schema.simpleString())
        df = df.select('*')
        out_rows = df.collect()

        df = s.createDataFrame(out_rows, df.schema.simpleString())
        rows = df.collect()
        rows = sorted(rows, key=lambda row: row[0] or '')

        self.assertEqual(df.schema, in_schema)
        self.assertEqual(rows, in_rows)

    def _test_pandas(self):
        from pyspark.sql import SparkSession as Session
        from datetime import datetime
        import duckdb
        import pandas

        print(datetime.now())

        # data = [
        #     ["a" * 10 for j in range(100)]
        #     for i in range(1_000_000)
        # ]
        data = [
            [('abc', 123, True), ['a', 'b'], {'a': 1, 'b': 2}],
            # [(None, None, None), [], {'b': 20, 'c': 30, 'd': None}],
            [None, None, None],
        ]
        # data = [
        #     [{'_1': 'abc', '_2': 123, '_3': True}, ['a', 'b'], {'key': ['a', 'b'], 'value': [1, 2]}],
        #     # [(None, None, None), [], {'b': 20, 'c': 30, 'd': None}],
        #     [None, None, None],
        # ]
        pd = pandas.DataFrame.from_records(data)

        print(datetime.now(), pd.shape)

        s = Session.builder.getOrCreate()
        df = s.createDataFrame(pd)
        df.printSchema()
        df.show(truncate=False)

        pd2 = df.toPandas()

        for t in pd2.itertuples():
            print(t)

        df = s.createDataFrame(pd2)
        df.printSchema()
        df.show(truncate=False)

        print(datetime.now())

        rel = duckdb.sql("FROM pd")
        print(rel)

        rel = duckdb.sql("SELECT {'_1': 1, '_2': 'abc'}, MAP {'a': 1, 'b': 2}")
        print(rel.columns[:3])
        print(rel.types[:3])
        rows = rel.fetchall()
        pd = rel.df()

        for t in pd.itertuples():
            print(t)

        # df = spark.createDataFrame(pd, df.schema.simpleString())
        # df = df.select('*')
        # df.printSchema()
        # rows = df.collect()
        # print(rows[:1])
        #
        # print(datetime.now(), len(rows))

        # import duckdb
        # print(duckdb.sql("select * from duckdb_tables()"))

    def _test(self):
        # from pyspark.sql import SparkSession as Session
        # s = Session.builder.getOrCreate()
        from datetime import datetime
        print(datetime.now())
        data = [
            ["a" * 10 for j in range(100)]
            for i in range(100_000)
        ]
        schema = T.StructType([T.StructField(f'_{j + 1}', T.StringType()) for j in range(100)])
        df = spark.createDataFrame(data, None)
        print(datetime.now())
        df.printSchema()
        rows = df.collect()
        print(datetime.now(), len(rows))
        # df = s.createDataFrame(data).toPandas()
        # print(datetime.now(), df.shape)
        # import duckdb
        # print(datetime.now())

        # df = spark.createDataFrame(df, df.schema.simpleString())
        # # duckdb.register('create_1', df)
        # rows = duckdb.sql("SELECT * FROM generate_series(100_000_000)").fetchall()
        # print(datetime.now(), len(rows))
        # import duckdb
        # print(duckdb.sql("select * from duckdb_tables()"))

    def _test_table_read_and_write(self):
        # print(spark.catalog.listCatalogs())
        # print(spark.catalog.listDatabases())
        # print(spark.catalog.listTables())

        data = [(123, 'abc', [False, True], (1, 2))]
        df = spark.createDataFrame(
            data=data,
            schema=T.StructType([
                T.StructField('i', T.IntegerType(), False),
                T.StructField('s', T.StringType(), False),
                T.StructField('a', T.ArrayType(T.BooleanType(), False), False),
                T.StructField('p', T.StructType([
                    T.StructField('x', T.IntegerType(), False),
                    T.StructField('y', T.IntegerType(), False),
                ]), False),
            ]),
        )
        # spark.sql('DROP TABLE IF EXISTS spark_catalog.default.my_table')
        df.write.saveAsTable('spark_catalog.default.my_table')
        df = spark.table('spark_catalog.default.my_table')

        self.assertEqual(T.StructType([
            T.StructField('i', T.IntegerType()),
            T.StructField('s', T.StringType()),
            T.StructField('a', T.ArrayType(T.BooleanType())),
            T.StructField('p', T.StructType([
                T.StructField('x', T.IntegerType()),
                T.StructField('y', T.IntegerType()),
            ])),
        ]), df.schema)

        rows = df.collect()
        self.assertEqual(data, rows)

        # print(spark.catalog.listCatalogs())
        # print(spark.catalog.listDatabases())
        # print(spark.catalog.listTables())
