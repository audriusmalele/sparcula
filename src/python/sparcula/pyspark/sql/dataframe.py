from datetime import datetime
from functools import reduce
from typing import TYPE_CHECKING, Optional, Union

from .._internal._column import ColBy, Lit
from .._internal._dataframe import (
    CacheDataFrame, DistinctDataFrame, JoinDataFrame, SelectDataFrame,
    UnionDataFrame, WhereDataFrame,
    Code, CodeGenerator, distance,
)
from .._internal._functions import Cast
from .._internal._types import FullType, get_common_type
from . import Column, types as T, functions as F
from .readwriter import DataFrameWriter, _execute, _log_call_stack, _open_params, _close_params
from .window import _flatten

if TYPE_CHECKING:
    from logging import Logger

    import pandas
    from duckdb import DuckDBPyRelation

    from .._internal._column import BaseColumn
    from .._internal._dataframe import BaseDataFrame
    from ._typing import ColumnOrName
    from .group import GroupedData


class DataFrame:
    def __init__(self, df: 'BaseDataFrame'):
        self._df = df

    @property
    def schema(self) -> 'T.StructType':
        return T.StructType([
            T.StructField(f.name, f.type.data_type, f.type.nullable)
            for f in self._df.fields
        ])

    @property
    def dtypes(self) -> list[tuple[str, str]]:
        return [(f.name, f.type.data_type.simpleString()) for f in self._df.fields]

    @property
    def columns(self) -> list[str]:
        return [f.name for f in self._df.fields]

    def __getitem__(self, item: Union[int, str]) -> 'Column':
        df = self._df
        if isinstance(item, str):
            if item == '*':
                return Column(ColBy('*', df))
            f = df.get_field(ColBy(item))
        else:
            f = df.fields[item]
        return Column(ColBy(f.name, df, f.index))

    def __getattr__(self, name: str) -> 'Column':
        return self[name]

    def alias(self, alias: str) -> 'DataFrame':
        return DataFrame(self._df.with_alias(alias))

    def filter(self, condition: 'ColumnOrName') -> 'DataFrame':
        return self.where(condition)

    def where(self, condition: 'ColumnOrName') -> 'DataFrame':
        return DataFrame(WhereDataFrame(self._df, Column._get_column_from_name(condition)))

    def withColumn(self, name: str, col: 'Column') -> 'DataFrame':
        return self.withColumns({name: col})

    def withColumns(self, *colsMap: dict[str, 'Column']) -> 'DataFrame':
        column_by_name = {}
        for m in colsMap:
            for k, v in m.items():
                assert k not in column_by_name, k
                column_by_name[k] = v._column
        return DataFrame(SelectDataFrame.from_withColumns(self._df, column_by_name))

    def withColumnRenamed(self, existing: str, new: str) -> 'DataFrame':
        return self.withColumnsRenamed({existing: new})

    def withColumnsRenamed(self, colsMap: dict[str, str]) -> 'DataFrame':
        df = self._df
        name_by_index = {}
        for current_name, new_name in colsMap.items():
            try:
                f = df.get_field(ColBy(current_name))
                name_by_index[f.index] = new_name
            except RuntimeError as e:
                if not str(e).startswith('Missing column'):
                    raise e
        if not name_by_index:
            return self
        columns: list['BaseColumn'] = []
        for i, f in enumerate(df.fields):
            column = ColBy(f.name, df, i)
            name = name_by_index.get(i)
            if name is not None:
                column.alias = name
            columns.append(column)
        return DataFrame(SelectDataFrame.from_select(df, columns, 'withColumnsRenamed'))

    def drop(self, *cols: 'ColumnOrName') -> 'DataFrame':
        df = self._df
        indices = set()
        for c in cols:
            if isinstance(c, str):
                col_by = ColBy(c)
            else:
                column = c._column
                assert isinstance(column, ColBy)
                col_by = column
            try:
                indices.add(df.get_field(col_by).index)
            except RuntimeError as e:
                if not str(e).startswith('Missing column'):
                    raise e
        if not indices:
            return self
        columns: list['BaseColumn'] = [
            ColBy(f.name, df, i) for i, f in enumerate(df.fields) if i not in indices
        ]
        return DataFrame(SelectDataFrame.from_select(df, columns, 'drop'))

    def select(self, *cols: Union['ColumnOrName', list['ColumnOrName']]) -> 'DataFrame':
        df = self._df
        columns: list['BaseColumn'] = []
        for col in _flatten(*cols):
            if isinstance(col, Column):
                column = col._column
            else:
                assert isinstance(col, str)
                column = ColBy(col)

            if isinstance(column, ColBy):
                split = column.name.split('.')
                if split[-1] == '*':
                    if len(split) == 1:
                        if column.df:
                            assert column.fi is None
                            for f in df.fields:
                                if distance(df, f, column.df) != -1:
                                    columns.append(ColBy(f.name, df, f.index))
                        else:
                            for f in df.fields:
                                columns.append(ColBy(f.name, df, f.index))
                    else:
                        assert len(split) == 2
                        assert split[1] == '*'
                        alias = split[0]
                        for f in df.fields:
                            if alias in f.df_aliases:
                                columns.append(ColBy(f.name, df, f.index))
                else:
                    columns.append(column)
            else:
                columns.append(column)
        return DataFrame(SelectDataFrame.from_select(df, columns, 'select'))

    def crossJoin(self, other: 'DataFrame') -> 'DataFrame':
        return self.join(other, how='cross')

    def join(
        self,
        other: 'DataFrame',
        on: Optional[Union[str, list[str], 'Column', list['Column']]] = None,
        how: Optional[str] = None,
    ) -> 'DataFrame':
        if how is None:
            how = 'inner'
        elif how == 'cross':
            assert on is None
        elif how in ('leftouter', 'left_outer'):
            how = 'left'
        elif how in ('rightouter', 'right_outer'):
            how = 'right'
        elif how in ('fullouter', 'full_outer', 'outer'):
            how = 'full'
        elif how in ('leftsemi', 'left_semi'):
            how = 'semi'
        elif how in ('leftanti', 'left_anti'):
            how = 'anti'
        assert how in ('cross', 'inner', 'left', 'right', 'full', 'semi', 'anti')

        if on is None:
            condition = None
            to_merge = set()
        elif isinstance(on, str):
            condition = self[on] == other[on]
            to_merge = {on}
        elif isinstance(on, list) and isinstance(on[0], str):
            names = set()
            for c in on:
                assert isinstance(c, str)
                names.add(c)
            condition = reduce(Column.__and__, [self[c] == other[c] for c in names])
            to_merge = names
        elif isinstance(on, list) and isinstance(on[0], Column):
            columns = []
            for c in on:
                assert isinstance(c, Column)
                columns.append(c)
            condition = reduce(Column.__and__, columns)
            to_merge = set()
        else:
            assert isinstance(on, Column)
            condition = on
            to_merge = set()

        condition_column = condition._column if condition else None
        df = DataFrame(JoinDataFrame(self._df, other._df, condition_column, how, to_merge))
        return df

    def distinct(self) -> 'DataFrame':
        return self.drop_duplicates()

    def drop_duplicates(self, subset: Optional[list[str]] = None) -> 'DataFrame':
        return self.dropDuplicates(subset)

    def dropDuplicates(self, subset: Optional[list[str]] = None) -> 'DataFrame':
        return DataFrame(DistinctDataFrame.of(self._df, subset))

    def groupby(self, *columns: 'ColumnOrName') -> 'GroupedData':
        return self.groupBy(*columns)

    def groupBy(self, *columns: 'ColumnOrName') -> 'GroupedData':
        from .group import GroupedData

        return GroupedData(self._df, [Column._get_column_from_name(c) for c in columns])

    def unionByName(self, other: 'DataFrame', allowMissingColumns: bool = False) -> 'DataFrame':
        left_df = self._df
        right_df = other._df
        left_set = {f.name for f in left_df.fields}
        right_set = {f.name for f in right_df.fields}
        assert len(left_set) == len(left_df.fields)
        assert len(right_set) == len(right_df.fields)

        left_only = left_set
        type_by_name = {f.name: f.type for f in left_df.fields}
        for f in right_df.fields:
            left_type = type_by_name.get(f.name)
            if left_type:
                left_only.remove(f.name)
                type = get_common_type([left_type, f.type])
            else:
                type = FullType(f.type.data_type, True)
            type_by_name[f.name] = type
        for name in left_only:
            type_by_name[name] = FullType(type_by_name[name].data_type, True)

        def adjust(df: 'BaseDataFrame') -> 'BaseDataFrame':
            field_by_name = {f.name: f for f in df.fields}
            columns: list['BaseColumn'] = []
            for name, type in type_by_name.items():
                field = field_by_name.get(name)
                if field:
                    column = Cast(ColBy(field.name), type.data_type, type.nullable)
                else:
                    assert allowMissingColumns
                    column = Lit.of(None).with_type(type.data_type)
                column.alias = name
                columns.append(column)
            if (
                any([isinstance(c, Cast) for c in columns])
                or [f.name for f in df.fields] != [n for n in type_by_name]
            ):
                df = SelectDataFrame.from_select(df, columns, 'union')
            return df

        return DataFrame(UnionDataFrame(adjust(left_df), adjust(right_df)))

    def replace(self, to_replace: str, value: str) -> 'DataFrame':
        return self.select([
            F.when(self[i] == F.lit(to_replace), value).otherwise(self[i]).alias(f.name)
            if isinstance(f.type.data_type, T.StringType)
            else self[i]
            for i, f in enumerate(self._df.fields)
        ])

    def toPandas(self) -> 'pandas.DataFrame':
        def terminate(
            relation: 'DuckDBPyRelation', logger: 'Logger', t0: 'datetime'
        ) -> 'pandas.DataFrame':
            pd = relation.df()
            t1 = datetime.now()
            logger.info(f'Fetched in {t1 - t0}')
            return pd

        return _execute(self._df, terminate)

    @property
    def write(self) -> 'DataFrameWriter':
        return DataFrameWriter(self._df)

    def cache(self) -> 'DataFrame':
        t0 = datetime.now()

        parent = self._df
        session = parent.session
        logger = session.logger

        df: 'BaseDataFrame'
        if session.cache_path is not None:
            import uuid
            from .._internal._dataframe import ParquetDataFrame

            name = str(uuid.uuid4())
            path = f'{session.cache_path}/{name}'
            logger.info(f'CACHE START - {name}')

            self.write.parquet(path, compression='uncompressed')

            df = ParquetDataFrame(session, path + '/*.parquet', self.schema)
        else:
            cache_df = CacheDataFrame(session, parent.fields)
            df = cache_df
            name = cache_df.table
            logger.info(f'CACHE START - {name}')

            _log_call_stack(logger, session.call_stack_depth)

            code, params = CodeGenerator(parent, 'cache').generate()
            sql = (
                Code()
                .line(f"CREATE TEMPORARY TABLE {name} AS SELECT * FROM (")
                .code(code)
                .line(")")
                .to_string()
            )

            _open_params(params, session)
            try:
                session.connection.sql(sql)
            except Exception as e:
                a = 1  # for debugging
                raise e
            finally:
                _close_params(params, session)

        t1 = datetime.now()
        logger.info(f'CACHE END - {name} - duration: {t1 - t0}')
        return DataFrame(df)

    def collect(self) -> list['T.Row']:
        def terminate(relation, logger, t0):
            tuples = relation.fetchall()

            t1 = datetime.now()
            logger.info(f'Fetched in {t1 - t0}')

            def convert_object(value: object, data_type: 'T.DataType') -> object:
                if value is None:
                    return None
                if isinstance(data_type, T.StructType):
                    assert isinstance(value, dict)
                    return T.Row(*[
                        (f.name, convert_object(value.get(f.name), f.dataType))
                        for f in data_type.fields
                    ])
                if isinstance(data_type, T.ArrayType):
                    assert isinstance(value, list)
                    return [convert_object(v, data_type.elementType) for v in value]
                return value

            fields = self._df.fields
            rows = [
                T.Row(*[
                    (f.name, convert_object(v, f.type.data_type))
                    for f, v in zip(fields, t)
                ])
                for t in tuples
            ]
            return rows

        return _execute(self._df, terminate)

    def count(self) -> int:
        def terminate(relation, _1, _2) -> int:
            count = relation.fetchall()[0][0]
            assert isinstance(count, int)
            return count

        return _execute(self._df, terminate, purpose='count')

    def isEmpty(self) -> bool:
        def terminate(relation, _1, _2) -> bool:
            exists = relation.fetchall()[0][0]
            assert isinstance(exists, bool)
            return not exists

        return _execute(self._df, terminate, purpose='isEmpty')

    def printSchema(self) -> None:
        self._print_type(0, self.schema)

    def _print_type(
        self,
        level: int,
        data_type: 'T.DataType',
        nullable: Optional[bool] = None,
        nullable_name: str = 'nullable',
    ) -> None:
        level_part = ' |   ' * level + ' |-- '
        type_part = data_type.typeName()
        if nullable is not None:
            type_part += f' ({nullable_name} = {"true" if nullable else "false"})'
        print('root' if level == 0 else type_part)
        if isinstance(data_type, T.StructType):
            for field in data_type.fields:
                print(f'{level_part}{field.name}: ', end='')
                self._print_type(level + 1, field.dataType, field.nullable)
        elif isinstance(data_type, T.ArrayType):
            print(f'{level_part}element: ', end='')
            self._print_type(level + 1, data_type.elementType,
                             data_type.containsNull, 'containsNull')
        elif isinstance(data_type, T.MapType):
            print(f'{level_part}key: ', end='')
            self._print_type(level + 1, data_type.keyType)
            print(f'{level_part}value: ', end='')
            self._print_type(level + 1, data_type.valueType,
                             data_type.valueContainsNull, 'valueContainsNull')

    def unpersist(self, blocking: bool = False) -> 'DataFrame':
        return self
