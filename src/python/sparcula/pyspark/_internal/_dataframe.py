import math
from copy import copy
from itertools import zip_longest
from typing import Optional, TYPE_CHECKING, Union

from ._column import (
    Agg, Col, ColBy, Explode, Fn, Lit, Over, WhenBasic, WhenComposite, adjust_when_name
)
from ._functions import GetItem, When
from ._types import (
    FullType, data_type_from_string, escape_name, escape_string,
    full_type_from_value, get_common_type, type_literal,
)
from ..sql import types as T

if TYPE_CHECKING:
    import pandas

    from ._column import BaseColumn
    from ._session import Session
    from .._typing import CellType


class Field:
    def __init__(
        self,
        name: str,
        type: 'FullType',
        index: int,
        previous: list[tuple[int, int]],
        df_aliases: list[str],
    ):
        self.name = name
        self.type = type
        self.index = index
        self.previous = previous
        self.df_aliases = df_aliases
        self._ref = f'"{index + 1}:{escape_name(name)}"'

    def ref(self) -> str:
        return self._ref

    def type_comment(self) -> str:
        return f"{self.type.data_type.simpleString()}{'?' if self.type.nullable else ''}"

    def __repr__(self) -> str:
        return f'{self.index}: {self.name}: {self.type}'


total_calls = 0
max_depth = 0


def distance(
    df: 'BaseDataFrame',
    field: 'Field',
    target_df: 'BaseDataFrame',
    target_fi: Optional[int] = None,
    depth: int = 1,
) -> int:
    global total_calls, max_depth
    total_calls += 1
    max_depth = max(max_depth, depth)

    if depth == 1000:
        return -1
    if df.original_df == target_df.original_df and (target_fi is None or field.index == target_fi):
        return 0
    ds = []
    for di, fi in field.previous:
        dependency = df.dependencies[di]
        d = distance(dependency, dependency.fields[fi], target_df, target_fi, depth + 1)
        if d != -1:
            ds.append(d)
    if ds:
        return min(ds) + 1
    return -1


def get_di_fi_fs(dfs: list['BaseDataFrame'], col: 'ColBy') -> tuple[int, int, list[str]]:
    names = col.name.split('.')

    # HOT SPOT START

    di_fi_fs_list_1: list[tuple[int, int, list[str]]] = []
    di_fi_fs_list_2: list[tuple[int, int, list[str]]] = []
    di_fi_fs_list_3: list[tuple[int, int, list[str]]] = []

    names_0 = names[0]
    if len(names) == 1:  # hotter
        for di, df in enumerate(dfs):
            for f in df.fields_by_name.get(names_0, []):
                di_fi_fs_list_3.append((di, f.index, []))
    else:  # colder
        names_1 = names[1]
        fs1 = names[1:]
        fs2 = names[2:]
        equal_names = names_0 == names_1
        for di, df in enumerate(dfs):
            df_fields_by_name = df.fields_by_name
            for f in df_fields_by_name.get(names_1, []):
                if df.alias == names_0:
                    di_fi_fs = (di, f.index, fs2)
                    di_fi_fs_list_1.append(di_fi_fs)
                    di_fi_fs_list_2.append(di_fi_fs)
                    di_fi_fs_list_3.append(di_fi_fs)
                elif names_0 in f.df_aliases:
                    di_fi_fs = (di, f.index, fs2)
                    di_fi_fs_list_2.append(di_fi_fs)
                    di_fi_fs_list_3.append(di_fi_fs)
                elif equal_names:
                    di_fi_fs_list_3.append((di, f.index, fs1))
            if not equal_names:
                for f in df_fields_by_name.get(names_0, []):
                    di_fi_fs_list_3.append((di, f.index, fs1))

    # hotter
    if len(di_fi_fs_list_3) == 1:
        return di_fi_fs_list_3[0]
    # colder
    if len(di_fi_fs_list_2) == 1:
        return di_fi_fs_list_2[0]
    if len(di_fi_fs_list_1) == 1:
        return di_fi_fs_list_1[0]

    # HOT SPOT END

    if len(di_fi_fs_list_3) == 0:
        available_columns = [f.name for df in dfs for f in df.fields]
        raise RuntimeError(f"Missing column '{col.name}', "
                           f"available columns: {available_columns}")

    if col.df is not None:
        di_fi_fs_list_4 = []
        di_fi_fs_list_5 = []
        for di, fi, fs in di_fi_fs_list_3:
            df = dfs[di]
            f = df.fields[fi]
            d = distance(df, f, col.df, col.fi)
            if d != -1:
                if d == 0:
                    di_fi_fs_list_4.append((di, fi, fs))
                di_fi_fs_list_5.append((di, fi, fs))

        if len(di_fi_fs_list_4) == 1:
            return di_fi_fs_list_4[0]
        if len(di_fi_fs_list_5) == 1:
            return di_fi_fs_list_5[0]

    matches = [dfs[di].fields[fi].name for di, fi, _ in di_fi_fs_list_3]
    raise RuntimeError(f"Ambiguous column '{col.name}', multiple matches: {matches}")


class Code:
    _INDENT = 2 * ' '

    def __init__(self) -> None:
        self._list: list[Union[str, int, tuple[int, 'Code']]] = []
        self._indent = 0
        self._line_in_progress = False

    def increase_indent(self) -> 'Code':
        self._indent += 1
        return self

    def decrease_indent(self) -> 'Code':
        self._indent -= 1
        return self

    def text(self, text: str) -> 'Code':
        if not self._line_in_progress:
            self._list.append(self._indent)
        self._list.append(text)
        self._line_in_progress = True
        return self

    def newline(self) -> 'Code':
        self._list.append('\n')
        self._line_in_progress = False
        return self

    def line(self, text: str) -> 'Code':
        self._list.append(self._indent)
        self._list.append(text)
        return self.newline()

    def code(self, code: 'Code') -> 'Code':
        self._list.append((self._indent, code))
        self._line_in_progress = False
        return self

    def to_string(self) -> str:
        sink: list[str] = []
        self._collect_strings(sink, 0)
        return ''.join(sink)

    def _collect_strings(self, sink: list[str], base_indent: int) -> None:
        for e in self._list:
            if isinstance(e, str):
                sink.append(e)
            elif isinstance(e, int):
                e += base_indent
                sink.append(e * self._INDENT)
            else:
                indent, code = e
                indent += base_indent
                code._collect_strings(sink, indent)


class BaseDataFrame:
    def __init__(
        self, session: 'Session', fields: list['Field'], dependencies: list['BaseDataFrame']
    ):
        self.original_df = self
        self.session = session
        self.fields = fields
        self.dependencies = dependencies
        self.alias: Optional[str] = None

        fields_by_name: dict[str, list['Field']] = {}
        for f in fields:
            fields_by_name.setdefault(f.name, []).append(f)
        self.fields_by_name = fields_by_name

    def with_alias(self, alias: str) -> 'BaseDataFrame':
        copied = copy(self)
        copied.alias = alias
        return copied

    def __repr__(self) -> str:
        type = self.__class__.__name__.removesuffix("DataFrame")
        return f'{type}({len(self.fields)}):{id(self)}'

    def get_field(self, col: 'ColBy') -> 'Field':
        return self.fields[get_di_fi_fs([self], col)[1]]

    def generate_code(self, h: 'DataFrameCodeHelper') -> None:
        raise NotImplementedError()

    # for debugging
    def sql(self) -> str:
        return CodeGenerator(self, 'collect').generate()[0].to_string()


class FromDataFrame(BaseDataFrame):
    pass


class CacheDataFrame(FromDataFrame):
    def __init__(self, session: 'Session', fields: list['Field']):
        fields = [Field(f.name, f.type, i, [], []) for i, f in enumerate(fields)]
        super().__init__(session, fields, [])
        self.table = f"cache_{id(self)}"

    def generate_code(self, h: 'DataFrameCodeHelper') -> None:
        h.code.line("-- from/cache").line("SELECT *").line(f"FROM {self.table}")


class CreateDataFrame(FromDataFrame):
    def __init__(
        self,
        session: 'Session',
        data: Union[list[Union[list['CellType'], dict[str, 'CellType']]], 'pandas.DataFrame'],
        schema: Optional[Union['T.StructType', str, list[str]]],
    ):
        import pandas

        if isinstance(schema, str):
            data_type = data_type_from_string(schema)
            assert isinstance(data_type, T.StructType)
            schema = data_type

        if isinstance(data, pandas.DataFrame):
            assert isinstance(schema, T.StructType)
            fields = [
                Field(f.name, FullType(f.dataType, f.nullable), i, [], [])
                for i, f in enumerate(schema.fields)
            ]
            self.df = data
        else:
            types: list[Optional['FullType']]
            if isinstance(schema, T.StructType):
                names = schema.names
                types = [FullType(f.dataType, f.nullable) for f in schema.fields]
            elif isinstance(schema, list):
                names = schema
                types = [None for _ in names]
            else:
                assert data
                row = data[0]
                if isinstance(row, dict):
                    all_names = {}
                    for row in data:
                        assert isinstance(row, dict)
                        for name in row.keys():
                            all_names[name] = True
                    names = list(all_names.keys())
                else:
                    names = [f'_{i + 1}' for i in range(len(row))]
                types = [None for _ in names]

            if not isinstance(schema, T.StructType):
                for row in data:
                    if isinstance(row, dict):
                        values = [row.get(n) for n in names]
                    else:
                        values = row
                    for i, value in enumerate(values):
                        new_type = full_type_from_value(value)
                        current_type = types[i]
                        if current_type is None:
                            types[i] = new_type
                        else:
                            types[i] = get_common_type([current_type, new_type])

            fields = []
            for i in range(len(names)):
                name = names[i]
                type = types[i]
                assert type
                fields.append(Field(name, type, i, [], []))

            def convert_value(value: object, data_type: 'T.DataType') -> object:
                if value is None:
                    return None
                if isinstance(data_type, T.FractionalType):
                    assert isinstance(value, (int, float))
                    return str(value) if math.isnan(value) else value
                if isinstance(data_type, T.MapType):
                    assert isinstance(value, dict)
                    return {
                        'key': [convert_value(k, data_type.keyType) for k in value.keys()],
                        'value': [convert_value(v, data_type.valueType) for v in value.values()],
                    }
                if isinstance(data_type, T.StructType):
                    if isinstance(value, dict):
                        return {
                            f.name: convert_value(value.get(f.name), f.dataType)
                            for f in data_type.fields
                        }
                    assert isinstance(value, tuple)
                    return {
                        f.name: convert_value(v, f.dataType)
                        for f, v in zip(data_type.fields, value)
                    }
                return value

            rows: list[list[object]] = []
            for row in data:
                if isinstance(row, dict):
                    values = [row.get(n) for n in names]
                else:
                    values = row
                converted_values = [
                    convert_value(v, f.type.data_type) for v, f in zip(values, fields)
                ]
                rows.append(converted_values or [None])

            columns = [str(i) for i in range(len(fields) or 1)]
            self.df = pandas.DataFrame.from_records(rows, columns=columns)

        super().__init__(session, fields, [])

    def generate_code(self, h: 'DataFrameCodeHelper') -> None:
        view_name = f"create_{h.number}"
        h.params.append((view_name, self.df))
        types = [f.type for f in self.fields] or [FullType(T.NullType(), True)]
        code = h.code.line("-- from/create")
        code.line("SELECT")
        code.increase_indent()
        for i, (t, f) in enumerate(zip_longest(types, self.fields, fillvalue=None)):
            assert t
            code.line(f"#{i + 1}::{type_literal(t.data_type)} AS {f.ref() if f else '_'},"
                     f" -- {f.type_comment() if f else 'void?'}")
        code.decrease_indent()
        code.line(f"FROM {view_name}")


class CsvDataFrame(FromDataFrame):
    def __init__(
        self,
        session: 'Session',
        path: str,
        schema: Optional['T.StructType'],
        header: bool,
    ):
        if schema is None:
            sql = f"SELECT Columns FROM sniff_csv('{escape_string(path)}', sample_size=1)"
            columns = [c['name'] for c in session.connection.sql(sql).fetchall()[0][0]]
            assert columns
            fields = [
                Field(column if header else f'_c{i}', FullType(T.StringType(), True), i, [], [])
                for i, column in enumerate(columns)
            ]
        else:
            fields = [
                Field(f.name, FullType(f.dataType, f.nullable), i, [], [])
                for i, f in enumerate(schema.fields)
            ]
        super().__init__(session, fields, [])
        self.path = path
        self.header = header

    def generate_code(self, h: 'DataFrameCodeHelper') -> None:
        code = h.code.line("-- from/csv")
        code.line("SELECT * FROM read_csv(")
        code.increase_indent()
        code.line(f"'{escape_string(self.path)}',")
        code.line(f"header={self.header},")
        code.line("auto_detect=false,")
        code.line("escape='\"',")
        code.line("nullstr=['', 'null'],")
        code.line("null_padding=true,")
        code.line("columns={")
        code.increase_indent()
        for i, f in enumerate(self.fields):
            code.line(f"'{escape_string(f'{i + 1}:{f.name}')}': '{type_literal(f.type.data_type)}',"
                     f" -- {f.type_comment()}")
        code.decrease_indent()
        code.line("}")
        code.decrease_indent()
        code.line(")")


class ParquetDataFrame(FromDataFrame):
    def __init__(self, session: 'Session', path: str, schema: 'T.StructType'):
        self.path = path
        fields = [
            Field(f.name, FullType(f.dataType, f.nullable), i, [], [])
            for i, f in enumerate(schema.fields)
        ]
        super().__init__(session, fields, [])

    def generate_code(self, h: 'DataFrameCodeHelper') -> None:
        code = h.code.line("-- from/parquet")
        code.line("SELECT")
        code.increase_indent()
        for i, f in enumerate(self.fields):
            code.line(f"#{i + 1} AS {f.ref()}, -- {f.type_comment()}")
        code.decrease_indent()
        code.line(f"FROM read_parquet('{escape_string(self.path)}')")


class TableDataFrame(FromDataFrame):
    def __init__(self, session: 'Session', table_name: str):
        from databricks.sdk import WorkspaceClient

        tables_api = WorkspaceClient().tables
        table_info = tables_api.get(table_name)
        self.suffix = ''
        assert table_info.table_type
        if table_info.table_type.name == 'VIEW':
            self.suffix = f' (via view {table_name})'
            view_dependencies = table_info.view_dependencies
            assert view_dependencies
            dependencies = view_dependencies.dependencies
            assert dependencies
            table = dependencies[0].table
            assert table
            table_name = table.table_full_name
            table_info = tables_api.get(table_name)
        self.table_name = table_name
        assert table_info.table_id
        self.table_id: str = table_info.table_id
        assert table_info.storage_location
        self.storage_location: str = table_info.storage_location

        if self.storage_location is None:
            print(table_info)
            raise RuntimeError('no storage_location')

        fields = []
        assert table_info.columns
        for i, column_info in enumerate(table_info.columns):
            assert column_info.type_text
            data_type = data_type_from_string(column_info.type_text)
            assert column_info.name
            fields.append(Field(column_info.name, FullType(data_type, True), i, [], []))

        super().__init__(session, fields, [])

    def generate_code(self, h: 'DataFrameCodeHelper') -> None:
        code = h.code.line("-- from/table/delta")
        code.line("SELECT")
        code.increase_indent()
        for i, f in enumerate(self.fields):
            code.line(f"#{i + 1} AS {f.ref()}, -- {f.type_comment()}")
        code.decrease_indent()
        code.line(f"-- {self.table_name}{self.suffix}")
        code.line(f"FROM delta_scan('{escape_string(self.storage_location)}')")


class SelectDataFrame(BaseDataFrame):
    @staticmethod
    def from_withColumns(
        parent: 'BaseDataFrame', column_by_name: dict[str, 'BaseColumn']
    ) -> 'SelectDataFrame':
        original_parent = parent
        original_column_by_name = column_by_name
        column_by_name = {}
        previous_explode = None
        for name, column in original_column_by_name.items():
            if isinstance(column, Explode):
                assert previous_explode is None, (f'More than one explode in a select: '
                                                  f'{previous_explode}, {name}')
                previous_explode = name
                parent = ExplodeDataFrame(parent, column)
                f = parent.fields[-1]
                column = ColBy(f.name, parent, f.index)
            column_by_name[name] = column

        resolver = TypeResolver([parent])
        typed_columns = []
        fields = []
        for i, f in enumerate(original_parent.fields):
            if f.name in column_by_name:
                column = column_by_name.pop(f.name)
                c = resolver.typed(column)
                typed_columns.append(c)
                if isinstance(c, Col):
                    previous = [c.di_fi]
                    df_aliases = parent.fields[c.di_fi[1]].df_aliases
                else:
                    previous = []
                    df_aliases = []
                fields.append(Field(f.name, c.get_type(), i, previous, df_aliases))
            else:
                typed_columns.append(Col(f.name, f.type, (0, i)))
                fields.append(Field(f.name, f.type, i, [(0, i)], f.df_aliases))
        for name, column in column_by_name.items():
            c = resolver.typed(column)
            typed_columns.append(c)
            fields.append(Field(name, c.get_type(), len(fields), [], []))

        return SelectDataFrame(parent, fields, typed_columns, 'withColumns')

    @staticmethod
    def from_select(
        parent: 'BaseDataFrame', columns: list['BaseColumn'], source: str
    ) -> 'SelectDataFrame':
        original_columns = columns
        columns = []
        previous_explode = None
        for column in original_columns:
            if isinstance(column, Explode):
                assert previous_explode is None, (f'More than one explode in a select: '
                                                  f'{previous_explode}, {column.get_alias()}')
                previous_explode = column.get_alias()
                parent = ExplodeDataFrame(parent, column)
                f = parent.fields[-1]
                column = ColBy(f.name, parent, f.index)
                column.alias = previous_explode
            columns.append(column)

        resolver = TypeResolver([parent])
        typed_columns: list['BaseColumn'] = []
        fields = []
        for i, c in enumerate(columns):
            c = resolver.typed(c)
            typed_columns.append(c)
            if isinstance(c, Col):
                previous = [c.di_fi]
                df_aliases = parent.fields[c.di_fi[1]].df_aliases
            else:
                previous = []
                df_aliases = []
            fields.append(Field(c.get_alias(), c.get_type(), i, previous, df_aliases))

        return SelectDataFrame(parent, fields, typed_columns, source)

    def __init__(
        self,
        parent: 'BaseDataFrame',
        fields: list['Field'],
        columns: list['BaseColumn'],
        source: str,
    ):
        super().__init__(parent.session, fields, [parent])
        self.columns = columns
        self.source = source

    def generate_code(self, h: 'DataFrameCodeHelper') -> None:
        parent = self.dependencies[0]

        parent_columns = {f.ref() for f in parent.fields}
        unchanged_indices = set()
        modified = []
        added = []
        for column, field in zip(self.columns, self.fields):
            unchanged = False
            if isinstance(column, Col):
                di, fi = column.di_fi
                f = self.dependencies[di].fields[fi]
                if f.name == field.name and f.index == field.index:
                    unchanged = True

            alias = field.ref()
            if unchanged:
                unchanged_indices.add(field.index)
                parent_columns.remove(alias)
            else:
                if alias in parent_columns:
                    modified.append((column, field))
                    parent_columns.remove(alias)
                else:
                    added.append((column, field))

        code = Code()
        generator = ExpressionGenerator(code)
        if parent_columns:
            code.line("SELECT")
            code.increase_indent()
            for column, field in zip(self.columns, self.fields):
                generator.generate(column)
                if field.index not in unchanged_indices:
                    code.text(" AS ").text(field.ref())
                code.text(",").newline()
            code.decrease_indent()
            code.line(f"FROM {h.refs[0]}")
        else:
            if modified:
                code.line("SELECT * REPLACE (")
                code.increase_indent()
                for column, field in modified:
                    generator.generate(column)
                    code.text(" AS ").text(field.ref()).text(",").newline()
                code.decrease_indent()
                code.line("),")
            else:
                code.line("SELECT *,")
            if added:
                code.increase_indent()
                for column, field in added:
                    generator.generate(column)
                    code.text(" AS ").text(field.ref()).text(",").newline()
                code.decrease_indent()
            code.line(f"FROM {h.refs[0]}")
        windows = len(generator.window_name_by_sql)
        if windows != 0:
            code.line("WINDOW")
            code.increase_indent()
            for i, (sql, name) in enumerate(generator.window_name_by_sql.items()):
                suffix = "," if i != windows - 1 else ""
                code.line(f"{name} AS ({sql}){suffix}")
            code.decrease_indent()
        h.code.line(f"-- {'select' if windows == 0 else 'window'}/{self.source}").code(code)


class WhereDataFrame(BaseDataFrame):
    def __init__(self, parent: 'BaseDataFrame', condition: 'BaseColumn'):
        fields = [
            Field(f.name, f.type, i, [(0, i)], f.df_aliases)
            for i, f in enumerate(parent.fields)
        ]
        self.condition = TypeResolver([parent]).typed(condition)
        super().__init__(parent.session, fields, [parent])

    def generate_code(self, h: 'DataFrameCodeHelper') -> None:
        h.code.line("-- where").line(f"SELECT * FROM {h.refs[0]}").text(f"WHERE ")
        ExpressionGenerator(h.code).generate(self.condition)
        h.code.newline()


class UnionDataFrame(BaseDataFrame):
    def __init__(self, df: 'BaseDataFrame', other: 'BaseDataFrame'):
        fields = []
        assert len(df.fields) == len(other.fields)
        for i, (a, b) in enumerate(zip(df.fields, other.fields)):
            assert a.name == b.name
            assert a.type == b.type
            fields.append(Field(a.name, a.type, i, [(0, i), (1, i)], []))

        super().__init__(df.session, fields, [df, other])

    def generate_code(self, h: 'DataFrameCodeHelper') -> None:
        (
            h.code.line("-- union")
            .line(f"SELECT * FROM {h.refs[0]}")
            .line("UNION ALL")
            .line(f"SELECT * FROM {h.refs[1]}")
        )


class GroupDataFrame(BaseDataFrame):
    def __init__(
        self,
        parent: 'BaseDataFrame',
        key_columns: list['BaseColumn'],
        agg_columns: list['BaseColumn'],
    ):
        self.keys = len(key_columns)
        self.columns: list['BaseColumn'] = []
        fields = []
        resolver = TypeResolver([parent])
        for i, c in enumerate([*key_columns, *agg_columns]):
            c = resolver.typed(c)
            self.columns.append(c)
            if isinstance(c, Col):
                previous = [c.di_fi]
                df_aliases = parent.fields[c.di_fi[1]].df_aliases
            else:
                previous = []
                df_aliases = []
            fields.append(Field(c.get_alias(), c.get_type(), i, previous, df_aliases))

        super().__init__(parent.session, fields, [parent])

    def generate_code(self, h: 'DataFrameCodeHelper') -> None:
        code = h.code.line("-- group")
        code.line("SELECT")
        code.increase_indent()
        generator = ExpressionGenerator(code)
        for i, (column, field) in enumerate(zip(self.columns, self.fields)):
            if i == self.keys:
                code.line('-- aggregations:')

            unchanged = False
            if isinstance(column, Col):
                di, fi = column.di_fi
                f = self.dependencies[di].fields[fi]
                if f.name == field.name and f.index == field.index:
                    unchanged = True

            generator.generate(column)
            if unchanged:
                code.text(",").newline()
            else:
                code.text(f" AS {field.ref()},").newline()
        code.decrease_indent()
        code.line(f"FROM {h.refs[0]}").line("GROUP BY ALL")


class DistinctDataFrame(BaseDataFrame):
    @staticmethod
    def of(parent: 'BaseDataFrame', subset: Optional[list[str]]) -> 'DistinctDataFrame':
        fields = [
            Field(f.name, f.type, i, [(0, i)], f.df_aliases)
            for i, f in enumerate(parent.fields)
        ]
        keys = []
        if subset:
            resolver = TypeResolver([parent])
            for name in subset:
                column = resolver.typed(ColBy(name))
                assert isinstance(column, Col)
                keys.append(column)
        return DistinctDataFrame(parent, fields, keys)

    def __init__(self, parent: 'BaseDataFrame', fields: list['Field'], keys: list['Col']):
        super().__init__(parent.session, fields, [parent])
        self.keys = keys

    def generate_code(self, h: 'DataFrameCodeHelper') -> None:
        code = h.code.line("-- distinct")
        on_part = ""
        if self.keys:
            on_part = f" ON ({', '.join([self.fields[key.di_fi[1]].ref() for key in self.keys])})"
        code.line(f"SELECT DISTINCT{on_part} *").line(f"FROM {h.refs[0]}")


class ExplodeDataFrame(BaseDataFrame):
    def __init__(self, parent: 'BaseDataFrame', column: 'Explode'):
        self.outer = column.outer
        self.array_column = TypeResolver([parent]).typed(column.array_column)
        fields = []
        for i, f in enumerate(parent.fields):
            fields.append(Field(f.name, f.type, i, [(0, i)], f.df_aliases))
        data_type = self.array_column.get_type().data_type
        assert isinstance(data_type, T.ArrayType)
        element_type = FullType(data_type.elementType, data_type.containsNull)
        field = Field('<!-- explode --!>', element_type, len(fields), [], [])
        fields.append(field)

        super().__init__(parent.session, fields, [parent])

    def generate_code(self, h: 'DataFrameCodeHelper') -> None:
        code = h.code.line(f"-- explode/{'outer' if self.outer else 'inner'}")
        generator = ExpressionGenerator(code)
        ref = self.fields[-1].ref()
        if self.outer:
            adjusted = f"CASE WHEN {ref} IS NULL OR ARRAY_LENGTH({ref}) = 0 THEN [NULL] ELSE {ref} END"
            code.line(f"SELECT * REPLACE (")
            code.increase_indent()
            code.line(f"unnest({adjusted}) AS {ref}")
            code.decrease_indent()
            code.line(")")
            code.line(f"FROM (")
            code.increase_indent()
            code.line("SELECT *,")
            code.increase_indent()
            generator.generate(self.array_column)
            code.text(f" AS {ref}").newline()
            code.decrease_indent()
            code.line(f"FROM {h.refs[0]}")
            code.decrease_indent()
            code.line(")")
        else:
            code.line("SELECT *,")
            code.increase_indent()
            code.text("unnest(")
            generator.generate(self.array_column)
            code.text(f") AS {ref}").newline()
            code.decrease_indent()
            code.line(f"FROM {h.refs[0]}")


class JoinDataFrame(BaseDataFrame):
    def __init__(
        self,
        left: 'BaseDataFrame',
        right: 'BaseDataFrame',
        condition: Optional['BaseColumn'],
        how: str,
        to_merge: set[str],
    ):
        self.how = how
        if condition:
            condition = TypeResolver([left, right]).typed(condition)
            assert isinstance(condition.get_type().data_type, T.BooleanType)
        self.condition = condition

        self.selects = []
        fields: list['Field'] = []
        for f in left.fields:
            rf = None
            previous = [(0, f.index)]
            df_aliases = []
            if left.alias:
                df_aliases.append(left.alias)
            if f.name in to_merge:
                rf = right.get_field(ColBy(f.name))
                previous.append((1, rf.index))
                if right.alias:
                    df_aliases.append(right.alias)

            if how == 'right':
                if f.name in to_merge:
                    assert rf
                    type = rf.type
                    expression = f"r.{rf.ref()}"
                else:
                    type = self._make_nullable(f.type)
                    expression = f"l.{f.ref()}"
            elif how == 'full':
                if f.name in to_merge:
                    assert rf
                    type = get_common_type([f.type, rf.type])
                    expression = f"coalesce(l.{f.ref()}, r.{rf.ref()})"
                else:
                    type = self._make_nullable(f.type)
                    expression = f"l.{f.ref()}"
            else:
                type = f.type
                expression = f"l.{f.ref()}"

            field = Field(f.name, type, len(fields), previous, df_aliases)
            fields.append(field)
            self.selects.append(f"{expression} AS {field.ref()}")
        if how not in ('semi', 'anti'):
            for f in right.fields:
                if f.name in to_merge:
                    continue

                if how in ('left', 'full'):
                    type = self._make_nullable(f.type)
                else:
                    type = f.type

                df_aliases = []
                if right.alias:
                    df_aliases.append(right.alias)
                field = Field(f.name, type, len(fields), [(1, f.index)], df_aliases)
                fields.append(field)
                self.selects.append(f"r.{f.ref()} AS {field.ref()}")

        super().__init__(left.session, fields, [left, right])

    @classmethod
    def _make_nullable(cls, full_type: 'FullType') -> 'FullType':
        return full_type if full_type.nullable else FullType(full_type.data_type, True)

    def generate_code(self, h: 'DataFrameCodeHelper') -> None:
        code = h.code.line(f"-- join/{self.how}")
        code.line(f"SELECT {', '.join(self.selects)}")
        code.line(f"FROM {h.refs[0]} l")
        code.line(f"{self.how.upper()} JOIN {h.refs[1]} r")
        if self.condition:
            code.text("ON ")
            ExpressionGenerator(code, join=True).generate(self.condition)
            code.newline()


class TypeResolver:
    def __init__(
        self,
        dfs: Optional[list['BaseDataFrame']] = None,
        column_by_name: Optional[dict[str, 'BaseColumn']] = None,
    ):
        self._dfs = dfs
        self._column_by_name = column_by_name

    def typed(self, column: 'BaseColumn', finish: bool = True) -> 'BaseColumn':
        alias = column.alias
        created: BaseColumn
        if isinstance(column, Lit):
            created = copy(column)
        elif isinstance(column, ColBy):
            if self._column_by_name:
                created = copy(self._column_by_name[column.name])
            else:
                assert self._dfs
                di, fi, fs = get_di_fi_fs(self._dfs, column)
                field = self._dfs[di].fields[fi]
                created = Col(field.name, field.type, (di, fi))
                f = None
                for f in fs:
                    gi = GetItem(created, Lit.of(f))
                    gi.set_type()
                    created = gi
                alias = alias or f
        elif isinstance(column, Over):
            over = copy(column)
            agg_column = self.typed(column.agg_column, finish=False)
            assert isinstance(agg_column, Agg)
            over.agg_column = agg_column
            over.partition_columns = [self.typed(c) for c in column.partition_columns]
            over.order_columns = [self.typed(c) for c in column.order_columns]
            over.set_type()
            created = column.agg_column.finish(over)
        elif isinstance(column, WhenBasic):
            created = self._typed_when(column, self.typed(Lit.of(None)))
            created.name = adjust_when_name(column.name)
        elif isinstance(column, WhenComposite):
            created = self._typed_when(column.parent, self.typed(column.otherwise))
            created.name = adjust_when_name(column.name)
        elif isinstance(column, Fn):
            fn = copy(column)
            fn.args = [self.typed(arg) for arg in fn.args]
            fn.set_type()
            if finish and isinstance(column, Agg):
                created = column.finish(fn)
                alias = column.get_alias()
            else:
                created = fn
        else:
            raise RuntimeError(column)
        created.alias = alias
        return created

    def _typed_when(
        self,
        parent: Union['WhenBasic', 'WhenComposite'],
        typed_otherwise: 'BaseColumn',
    ) -> 'When':
        if isinstance(parent, WhenBasic):
            condition = self.typed(parent.condition)
            value = self.typed(parent.value)
            when = When(condition, value, typed_otherwise)
            when.set_type()
            return when
        else:
            assert isinstance(parent, WhenComposite)
            assert isinstance(parent.otherwise, WhenBasic)
            return self._typed_when(
                parent.parent, self._typed_when(parent.otherwise, typed_otherwise)
            )


class ColumnCodeHelper:
    def __init__(self, column: 'BaseColumn', generator: 'ExpressionGenerator'):
        self.column = column
        self.generator = generator

    def text(self, text: str) -> 'ColumnCodeHelper':
        self.generator.code.text(text)
        return self

    def arg(self, i: int) -> 'ColumnCodeHelper':
        arg = self.column.args[i]
        arg.generate_code(ColumnCodeHelper(arg, self.generator))
        return self

    def arg_column(self, column: 'BaseColumn') -> 'ColumnCodeHelper':
        column.generate_code(ColumnCodeHelper(column, self.generator))
        return self

    def window(self, sql: str) -> 'ColumnCodeHelper':
        name_by_sql = self.generator.window_name_by_sql
        name = name_by_sql.get(sql)
        if name is None:
            name = f"w{len(name_by_sql)}"
            name_by_sql[sql] = name
        return self.text(name)


class ExpressionGenerator:
    def __init__(self, code: 'Code', join: bool = False) -> None:
        self.code = code
        self.window_name_by_sql: dict[str, str] = {}
        self.join = join

    def generate(self, column: 'BaseColumn') -> None:
        column.generate_code(ColumnCodeHelper(column, self))


class DataFrameCodeHelper:
    def __init__(self, code: 'Code', params: list, refs: list[str], number: int):
        self.code = code
        self.params = params
        self.refs = refs
        self.number = number


class CodeGenerator:
    def __init__(self, df: 'BaseDataFrame', purpose: str):
        self.df = df
        self.purpose = purpose
        self.code = Code()
        self.params: list = []
        self.number_by_df: dict['BaseDataFrame', int] = {}

    def generate(self) -> tuple['Code', list]:
        self.code.line('WITH')
        last_ref = f"t{self._traverse(self.df)}"
        self.code.text('\n')

        if self.purpose == 'cache':
            self.code.line(f"SELECT * FROM {last_ref}")
        elif self.purpose == 'collect':
            self.code.line("SELECT")
            self.code.increase_indent()
            for i, f in enumerate(self.df.fields):
                self.code.line(f'{f.ref()} AS "{escape_name(f.name)}", -- {f.type_comment()}')
            self.code.decrease_indent()
            self.code.line(f"FROM {last_ref}")
        elif self.purpose == 'count':
            self.code.line(f"SELECT COUNT(*) FROM {last_ref}")
        elif self.purpose == 'isEmpty':
            self.code.line(f"SELECT EXISTS(FROM {last_ref})")
        else:
            raise RuntimeError(self.purpose)

        return self.code, self.params

    def _traverse(self, df: 'BaseDataFrame') -> int:
        refs = [f"t{self._traverse(d)}" for d in df.dependencies]
        number = self.number_by_df.get(df)
        if number is None:
            number = len(self.number_by_df)
            self.number_by_df[df] = number
            if number != 0:
                self.code.text(",\n")
            self.code.line(f"t{number} AS (")
            df.generate_code(DataFrameCodeHelper(self.code, self.params, refs, number))
            self.code.text(")")
        return number
