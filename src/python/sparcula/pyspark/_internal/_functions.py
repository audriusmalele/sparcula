from typing import TYPE_CHECKING, Optional

from ._column import Agg, Col, Fn, Lit, Param
from ._types import FullType, escape_name, get_common_type, is_nullable_cast, type_literal
from ..sql import types as T

if TYPE_CHECKING:
    from ._column import BaseColumn
    from ._dataframe import Code, ColumnCodeHelper


def _cast(
    columns: list['BaseColumn'], data_type: Optional['T.DataType'] = None
) -> tuple['FullType', list['BaseColumn']]:
    nullable = False
    if data_type is None:
        common_type = get_common_type([c.get_type() for c in columns])
        data_type = common_type.data_type
        nullable = common_type.nullable
    common_type = get_common_type([
        Cast.internal(c, data_type, nullable).get_type() for c in columns
    ])
    assert common_type.data_type == data_type
    cs = [Cast.internal(c, common_type.data_type, common_type.nullable) for c in columns]
    return common_type, cs


class When(Fn):
    def __init__(self, condition: 'BaseColumn', value: 'BaseColumn', otherwise: 'BaseColumn'):
        super().__init__('IGNORED', [condition, value, otherwise])

    def set_type(self) -> None:
        assert isinstance(self.args[0].get_type().data_type, T.BooleanType)
        self.full_type, self.args[1:] = _cast(self.args[1:])

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("CASE WHEN ").arg(0).text(" THEN ").arg(1).text(" ELSE ").arg(2).text(" END")


class Null(Fn):
    def __init__(self, is_null: bool, column: 'BaseColumn'):
        prefix = "" if is_null else "NOT "
        super().__init__(f'({column.as_string()} IS {prefix}NULL)', [column])
        self._is_null = is_null

    def set_type(self) -> None:
        self.full_type = FullType(T.BooleanType(), False)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        prefix = "" if self._is_null else "NOT "
        h.text("(").arg(0).text(f" IS {prefix}NULL)")


def get_comparison_type(args: list['BaseColumn']) -> 'T.DataType':
    a = args[0].get_type().data_type
    b = args[1].get_type().data_type
    if isinstance(a, T.StringType) and not isinstance(b, T.NullType):
        return b
    if isinstance(b, T.StringType) and not isinstance(a, T.NullType):
        return a
    return get_common_type([FullType(a, False), FullType(b, False)]).data_type


class EqNullSafe(Fn):
    def __init__(self, left: 'BaseColumn', right: 'BaseColumn'):
        super().__init__(f'({left.as_string()} <=> {right.as_string()})', [left, right])

    def set_type(self) -> None:
        comparison_type = get_comparison_type(self.args)
        _, self.args = _cast(self.args, comparison_type)
        self.full_type = FullType(T.BooleanType(), False)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("(").arg(0).text(" IS NOT DISTINCT FROM ").arg(1).text(")")


class Equal(Fn):
    def __init__(self, left: 'BaseColumn', right: 'BaseColumn'):
        super().__init__(f'({left.as_string()} = {right.as_string()})', [left, right])

    def set_type(self) -> None:
        comparison_type = get_comparison_type(self.args)
        common_type, self.args = _cast(self.args, comparison_type)
        self.full_type = FullType(T.BooleanType(), common_type.nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("(").arg(0).text(" = ").arg(1).text(")")


class Compare(Fn):
    def __init__(self, op: str, left: 'BaseColumn', right: 'BaseColumn'):
        super().__init__(f'({left.as_string()} {op} {right.as_string()})', [left, right])
        self._op = op

    def set_type(self) -> None:
        comparison_type = get_comparison_type(self.args)
        common_type, self.args = _cast(self.args, comparison_type)
        self.full_type = FullType(T.BooleanType(), common_type.nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("(").arg(0).text(" ").text(self._op).text(" ").arg(1).text(")")


class Not(Fn):
    def __init__(self, column: 'BaseColumn'):
        super().__init__(f'(NOT {column.as_string()})', [column])

    def set_type(self) -> None:
        arg_type = self.args[0].get_type()
        assert isinstance(arg_type.data_type, T.BooleanType)
        self.full_type = FullType(T.BooleanType(), arg_type.nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("(NOT ").arg(0).text(")")


class LogicalBinOp(Fn):
    def __init__(self, op: str, left: 'BaseColumn', right: 'BaseColumn'):
        super().__init__(f'({left.as_string()} {op} {right.as_string()})', [left, right])
        self._op = op

    def set_type(self) -> None:
        left_type = self.args[0].get_type()
        right_type = self.args[1].get_type()
        assert isinstance(left_type.data_type, T.BooleanType)
        assert isinstance(right_type.data_type, T.BooleanType)
        self.full_type = FullType(T.BooleanType(), left_type.nullable or right_type.nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("(").arg(0).text(" ").text(self._op).text(" ").arg(1).text(")")


def double_on_string(args: list['BaseColumn']) -> tuple['FullType', list['BaseColumn']]:
    if any([isinstance(arg.get_type().data_type, T.StringType) for arg in args]):
        return _cast(args, T.DoubleType())
    return _cast(args)


class Neg(Fn):
    def __init__(self, column: 'BaseColumn'):
        super().__init__(f'(- {column.as_string()})', [column])

    def set_type(self) -> None:
        self.full_type, self.args = double_on_string(self.args)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("(- ").arg(0).text(")")


class ArithmeticBinOp(Fn):
    def __init__(self, op: str, left: 'BaseColumn', right: 'BaseColumn'):
        super().__init__(f'({left.as_string()} {op} {right.as_string()})', [left, right])
        self._op = op

    def set_type(self) -> None:
        self.full_type, self.args = double_on_string(self.args)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("(").arg(0).text(" ").text(self._op).text(" ").arg(1).text(")")


class Div(Fn):
    def __init__(self, left: 'BaseColumn', right: 'BaseColumn'):
        super().__init__(f'({left.as_string()} / {right.as_string()})', [left, right])

    def set_type(self) -> None:
        self.full_type, self.args = _cast(self.args, T.DoubleType())

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("(").arg(0).text(" / ").arg(1).text(")")


class Cast(Fn):
    @staticmethod
    def internal(
        column: 'BaseColumn', data_type: 'T.DataType', nullable: bool = False
    ) -> 'BaseColumn':
        type = column.get_type()
        if type.data_type == data_type and (not nullable or type.nullable):
            return column
        fn = Cast(column, data_type, nullable)
        fn.set_type()
        fn.alias = column.get_alias()
        return fn

    def __init__(
        self, column: 'BaseColumn', data_type: 'T.DataType', nullable: bool = False
    ):
        if isinstance(data_type, (T.StructType, T.ArrayType, T.MapType)) and not nullable:
            name = column.as_string()
        else:
            suffix = '?' if nullable else ''
            name = f'CAST({column.as_string()} AS {data_type.simpleString().upper() + suffix})'
        super().__init__(name, [column])
        self._data_type = data_type
        self._nullable = nullable

    def set_type(self) -> None:
        arg_type = self.args[0].get_type()
        if arg_type.data_type == T.NullType():
            lit = Lit.of(None).with_type(self._data_type)
            self.args[0] = lit
            self.full_type = lit.type
        else:
            nullable = (
                arg_type.nullable
                or self._nullable
                or is_nullable_cast(arg_type.data_type, self._data_type)
            )
            self.full_type = FullType(self._data_type, nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        arg = self.args[0]
        arg_data_type = arg.get_type().data_type
        both_integral = (
            isinstance(arg_data_type, T.IntegralType)
            and isinstance(self._data_type, T.IntegralType)
        )
        if arg_data_type == self._data_type or both_integral:
            h.arg(0)
        elif arg_data_type == T.StringType() and self._data_type == T.DateType():
            h.text("try_strptime(").arg(0).text(", '%Y-%m-%d')::DATE")
        elif is_nullable_cast(arg_data_type, self._data_type):
            h.text("TRY_CAST(").arg(0).text(f" AS {type_literal(self.get_type().data_type)})")
        else:
            h.arg(0).text(f"::{type_literal(self.get_type().data_type)}")


class Struct(Fn):
    def __init__(self, columns: list['BaseColumn']):
        super().__init__(f'struct({", ".join([c.as_string() for c in columns])})', columns)

    def set_type(self) -> None:
        fields = []
        for i, arg in enumerate(self.args):
            if arg.alias is not None:
                name = arg.alias
            else:
                name = arg.name if isinstance(arg, Col) else f'col{i + 1}'
            full_type = arg.get_type()
            fields.append(T.StructField(name, full_type.data_type, full_type.nullable))
        self.full_type = FullType(T.StructType(fields), False)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        data_type = self.get_type().data_type
        assert isinstance(data_type, T.StructType)
        h.text("{")
        for i, f in enumerate(data_type.fields):
            if i != 0:
                h.text(", ")
            h.arg_column(Lit.of(f.name)).text(": ").arg(i)
        h.text("}")


class Array(Fn):
    def __init__(self, columns: list['BaseColumn']):
        super().__init__(f'array({", ".join([c.as_string() for c in columns])})', columns)

    def set_type(self) -> None:
        if self.args:
            common_type, self.args = _cast(self.args)
            array_type = T.ArrayType(common_type.data_type, common_type.nullable)
            self.full_type = FullType(array_type, False)
        else:
            self.full_type = FullType(T.ArrayType(T.NullType(), False), False)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("[")
        for i in range(len(self.args)):
            if i != 0:
                h.text(", ")
            h.arg(i)
        h.text("]")


class Like(Fn):
    def __init__(self, column: 'BaseColumn', pattern: str):
        super().__init__(f'{column.as_string()} LIKE {pattern}', [column])
        self._pattern = pattern

    def set_type(self) -> None:
        _, self.args = _cast(self.args, T.StringType())
        self.full_type = FullType(T.BooleanType(), self.args[0].get_type().nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("(").arg(0).text(" LIKE ").arg_column(Lit.of(self._pattern)).text(")")


class RLike(Fn):
    def __init__(self, column: 'BaseColumn', pattern: str):
        super().__init__(f'RLIKE({column.as_string()}, {pattern})', [column])
        self._pattern = pattern

    def set_type(self) -> None:
        _, self.args = _cast(self.args, T.StringType())
        self.full_type = FullType(T.BooleanType(), self.args[0].get_type().nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("regexp_matches(").arg(0).text(f", ").arg_column(Lit.of(self._pattern)).text(")")


class RegexpReplace(Fn):
    def __init__(self, string: 'BaseColumn', pattern: 'BaseColumn', replacement: 'BaseColumn'):
        args = [string, pattern, replacement]
        name = f'regexp_replace({", ".join([c.as_string() for c in args])}, 1)'
        super().__init__(name, args)

    def set_type(self) -> None:
        self.full_type, self.args = _cast(self.args, T.StringType())

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("regexp_replace(").arg(0).text(", ").arg(1).text(", ").arg(2).text(", 'g')")


class RegexpExtract(Fn):
    def __init__(self, column: 'BaseColumn', pattern: str, group: int):
        super().__init__(f'regexp_extract({column.as_string()}, {pattern}, {group})', [column])
        self._pattern = pattern
        self._group = group

    def set_type(self) -> None:
        self.full_type, self.args = _cast(self.args, T.StringType())

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("regexp_extract(").arg(0).text(", ").arg_column(Lit.of(self._pattern)).text(f", {self._group})")


class Split(Fn):
    def __init__(self, column: 'BaseColumn', pattern: str, limit: int):
        super().__init__(f'split({column.as_string()}, {pattern}, {limit})', [column])
        self._pattern = pattern

    def set_type(self) -> None:
        arg_type = self.args[0].get_type()
        assert isinstance(arg_type.data_type, T.StringType)
        self.full_type = FullType(T.ArrayType(T.StringType(), False), arg_type.nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("regexp_split_to_array(").arg(0).text(", ").arg_column(Lit.of(self._pattern)).text(")")


class SplitPart(Fn):
    def __init__(self, string: 'BaseColumn', delimiter: 'BaseColumn', number: 'BaseColumn'):
        args = [string, delimiter, number]
        super().__init__(f'split_part({", ".join([a.as_string() for a in args])})', args)

    def set_type(self) -> None:
        _, self.args[:1] = _cast(self.args[:1], T.StringType())
        _, self.args[1:2] = _cast(self.args[1:2], T.StringType())
        _, self.args[2:] = _cast(self.args[2:], T.IntegerType())
        self.full_type = FullType(T.StringType(), any([a.get_type().nullable for a in self.args]))

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("split_part(").arg(0).text(", ").arg(1).text(", ").arg(2).text(")")


class ElementAt(Fn):
    def __init__(self, column: 'BaseColumn', extraction: 'BaseColumn'):
        name = f'element_at({column.as_string()}, {extraction.as_string()})'
        super().__init__(name, [column, extraction])

    def set_type(self) -> None:
        column_type = self.args[0].get_type()
        extraction_type = self.args[1].get_type()
        assert isinstance(column_type.data_type, T.ArrayType)
        assert isinstance(extraction_type.data_type, T.IntegralType)
        self.full_type = FullType(column_type.data_type.elementType, True)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.arg(0).text("[").arg(1).text("]")


class Concat(Fn):
    def __init__(self, columns: list['BaseColumn']):
        super().__init__(f'concat({", ".join([c.as_string() for c in columns])})', columns)

    def set_type(self) -> None:
        arg_types = [arg.get_type() for arg in self.args]
        array = any([isinstance(t.data_type, T.ArrayType) for t in arg_types])
        if array:
            self.full_type = get_common_type(arg_types)
        else:
            if self.args:
                self.full_type, self.args = _cast(self.args, T.StringType())
            else:
                self.full_type = FullType(T.StringType(), False)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        if self.args:
            h.text("(")
            for i in range(len(self.args)):
                if i != 0:
                    h.text(" || ")
                h.arg(i)
            h.text(")")
        else:
            h.text("''")


class ConcatWs(Fn):
    def __init__(self, separator: str, columns: list['BaseColumn']):
        name = f'concat_ws({", ".join([separator] + [c.as_string() for c in columns])})'
        super().__init__(name, columns)
        self._separator = separator

    def set_type(self) -> None:
        self.full_type = FullType(T.StringType(), False)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        separator = Lit.of(self._separator)
        h.text("concat_ws(").arg_column(separator)
        for i, arg in enumerate(self.args):
            h.text(", ")
            if isinstance(arg.get_type().data_type, T.ArrayType):
                h.text("array_to_string(").arg(i).text(", ").arg_column(separator).text(")")
            else:
                h.arg(i)
        h.text(")")


class DateAddSub(Fn):
    def __init__(self, name: str, sign: str, start: 'BaseColumn', days: 'BaseColumn'):
        name = f'{name}({start.as_string()}, {days.as_string()})'
        super().__init__(name, [start, days])
        self._sign = sign

    def set_type(self) -> None:
        _, self.args[:1] = _cast(self.args[:1], T.DateType())
        _, self.args[1:] = _cast(self.args[1:], T.IntegerType())
        self.full_type = FullType(T.DateType(), any([a.get_type().nullable for a in self.args]))

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("(").arg(0).text(f" {self._sign} ").arg(1).text(")")


class DayOfWeek(Fn):
    def __init__(self, column: 'BaseColumn'):
        super().__init__(f'dayofweek({column.as_string()})', [column])

    def set_type(self) -> None:
        _, self.args = _cast(self.args, T.DateType())
        self.full_type = FullType(T.IntegerType(), self.args[0].get_type().nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("(1 + dayofweek(").arg(0).text("))")


class DateFormat(Fn):
    def __init__(self, column: 'BaseColumn', format: str):
        assert format == 'yyyy-MM-dd'
        super().__init__(f'date_format({column.as_string()}, {format})', [column])

    def set_type(self) -> None:
        _, self.args = _cast(self.args, T.DateType())
        self.full_type = FullType(T.StringType(), self.args[0].get_type().nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("strftime(").arg(0).text(", '%Y-%m-%d')")


class ToDate(Fn):
    def __init__(self, column: 'BaseColumn', format: Optional[str]):
        if format is None:
            format = 'yyyy-MM-dd'
            name = f'to_date({column.as_string()})'
        else:
            name = f'to_date({column.as_string()}, {format})'
        assert 'yyyy' in format or 'yy' in format
        assert 'MMM' in format or 'MM' in format or 'M' in format
        assert 'dd' in format or 'd' in format

        if 'yyyy' in format:
            format = format.replace('yyyy', '%Y')
        elif 'yy' in format:
            format = format.replace('yy', '%y')
        else:
            raise

        if 'MMM' in format:
            format = format.replace('MMM', '%b')
        elif 'MM' in format:
            format = format.replace('MM', '%m')
        elif 'M' in format:
            format = format.replace('M', '%-m')
        else:
            raise

        if 'dd' in format:
            format = format.replace('dd', '%d')
        elif 'd' in format:
            format = format.replace('d', '%-d')
        else:
            raise

        super().__init__(name, [column])
        self._format = format

    def set_type(self) -> None:
        _, self.args = _cast(self.args, T.StringType())
        self.full_type = FullType(T.DateType(), True)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("try_strptime(").arg(0).text(", ").arg_column(Lit.of(self._format)).text(")::DATE")


class ToTimestamp(Fn):
    _FORMATS = (
        'yyyy-MM-dd HH:mm:ss',
    )

    def __init__(self, column: 'BaseColumn', format: Optional[str]):
        if format is None:
            name = f'to_timestamp({column.as_string()})'
        else:
            assert format in self._FORMATS
            name = f'to_timestamp({column.as_string()}, {format})'
        super().__init__(name, [column])

    def set_type(self) -> None:
        _, self.args = _cast(self.args, T.StringType())
        self.full_type = FullType(T.TimestampType(), True)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("try_strptime(").arg(0).text(", '%Y-%m-%d %H:%M:%S')")


class StartsEndsWith(Fn):
    def __init__(self, type: str, string: 'BaseColumn', prefix: 'BaseColumn'):
        name = f'{type}with({string.as_string()}, {prefix.as_string()})'
        super().__init__(name, [string, prefix])
        self._function = type + '_with'

    def set_type(self) -> None:
        common_type, self.args = _cast(self.args, T.StringType())
        self.full_type = FullType(T.BooleanType(), common_type.nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text(self._function).text("(").arg(0).text(", ").arg(1).text(")")


class Contains(Fn):
    def __init__(self, string: 'BaseColumn', substring: 'BaseColumn'):
        name = f'contains({string.as_string()}, {substring.as_string()})'
        super().__init__(name, [string, substring])

    def set_type(self) -> None:
        common_type, self.args = _cast(self.args, T.StringType())
        self.full_type = FullType(T.BooleanType(), common_type.nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("contains(").arg(0).text(", ").arg(1).text(")")


class IsIn(Fn):
    def __init__(self, column: 'BaseColumn', columns: list['BaseColumn']):
        name = f'({column.as_string()} IN ({", ".join([c.as_string() for c in columns])}))'
        super().__init__(name, [column] + columns)

    def set_type(self) -> None:
        _, self.args = _cast(self.args)
        self.full_type = FullType(T.BooleanType(), self.args[0].get_type().nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("(").arg(0).text(" IN (")
        for i in range(1, len(self.args)):
            if i != 1:
                h.text(", ")
            h.arg(i)
        h.text("))")


class LPad(Fn):
    def __init__(self, column: 'BaseColumn', length: int, padding: str):
        super().__init__(f'lpad({column.as_string()}, {length}, {padding})', [column])
        self._length = length
        self._padding = padding

    def set_type(self) -> None:
        common_type, self.args = _cast(self.args, T.StringType())
        self.full_type = FullType(T.StringType(), common_type.nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("lpad(").arg(0).text(f", {self._length}, ").arg_column(Lit.of(self._padding)).text(")")


class Length(Fn):
    def __init__(self, column: 'BaseColumn'):
        super().__init__(f'length({column.as_string()})', [column])

    def set_type(self) -> None:
        common_type, self.args = _cast(self.args, T.StringType())
        self.full_type = FullType(T.IntegerType(), common_type.nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("length(").arg(0).text(")")


class Size(Fn):
    def __init__(self, column: 'BaseColumn'):
        super().__init__(f'size({column.as_string()})', [column])
        self._function: Optional[str] = None

    def set_type(self) -> None:
        self.full_type = FullType(T.IntegerType(), False)
        data_type = self.args[0].get_type().data_type
        if isinstance(data_type, T.ArrayType):
            self._function = 'len'
        elif isinstance(data_type, T.MapType):
            self._function = 'cardinality'
        else:
            raise RuntimeError(self)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        assert self._function
        h.text("coalesce(").text(self._function).text("(").arg(0).text("), -1)")


class ArrayContains(Fn):
    def __init__(self, array: 'BaseColumn', value: 'BaseColumn'):
        super().__init__(f'array_contains({array.as_string()}, {value.as_string()})', [array, value])

    def set_type(self) -> None:
        array_arg = self.args[0]
        value_arg = self.args[1]
        array_type = array_arg.get_type()
        value_type = value_arg.get_type()
        assert isinstance(array_type.data_type, T.ArrayType)
        assert value_type.data_type == array_type.data_type.elementType
        self.full_type = FullType(T.BooleanType(), array_type.nullable or value_type.nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("list_contains(").arg(0).text(", ").arg(1).text(")")


class ArrayExcept(Fn):
    def __init__(self, left: 'BaseColumn', right: 'BaseColumn'):
        super().__init__(f'array_except({left.as_string()}, {right.as_string()})', [left, right])

    def set_type(self) -> None:
        self.full_type, self.args = _cast(self.args)
        left_type = self.args[0].get_type()
        right_type = self.args[1].get_type()
        assert isinstance(left_type.data_type, T.ArrayType)
        assert isinstance(right_type.data_type, T.ArrayType)
        assert left_type.data_type.elementType == right_type.data_type.elementType
        assert left_type.data_type.containsNull == right_type.data_type.containsNull

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("CASE WHEN ").arg(0).text(" IS NULL").text(" OR ").arg(1).text(" IS NULL THEN NULL ")
        h.text("ELSE array(SELECT unnest(").arg(0).text(") EXCEPT SELECT unnest(").arg(1).text(")) END")


class ArraySort(Fn):
    def __init__(self, array: 'BaseColumn'):
        super().__init__(f'array_sort({array.as_string()})', [array])

    def set_type(self) -> None:
        arg_type = self.args[0].get_type()
        assert isinstance(arg_type.data_type, T.ArrayType)
        self.full_type = arg_type

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("list_sort(").arg(0).text(", 'ASC', 'NULLS LAST')")


class ArraysZip(Fn):
    def __init__(self, columns: list['BaseColumn']):
        super().__init__(f'arrays_zip({", ".join([c.as_string() for c in columns])})', columns)

    def set_type(self) -> None:
        fields = []
        nullable = False
        for i, arg in enumerate(self.args):
            full_type = arg.get_type()
            data_type = full_type.data_type
            assert isinstance(data_type, T.ArrayType)
            name = arg.name if isinstance(arg, Col) else arg.alias or str(i)
            fields.append(T.StructField(name, data_type.elementType))
            nullable = nullable or full_type.nullable
        self.full_type = FullType(T.ArrayType(T.StructType(fields), False), nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("CASE WHEN ")
        for i in range(len(self.args)):
            if i != 0:
                h.text(" OR ")
            h.arg(i).text(" IS NULL")
        h.text(" THEN NULL ELSE list_zip(")
        for i in range(len(self.args)):
            if i != 0:
                h.text(", ")
            h.arg(i)
        h.text(f") END::{type_literal(self.get_type().data_type)}")


class _Reduce(Fn):
    def __init__(self, array: 'BaseColumn', initial_value: 'BaseColumn', merge: 'BaseColumn'):
        super().__init__('_reduce', [array, initial_value])
        self._merge = merge
        self._lambda_code: Optional['Code'] = None

    def set_type(self) -> None:
        from ._dataframe import TypeResolver, Code, ExpressionGenerator

        array_type = self.args[0].get_type()
        initial_value_type = self.args[1].get_type()

        assert isinstance(array_type.data_type, T.ArrayType)
        element_type = FullType(array_type.data_type.elementType,
                                array_type.data_type.containsNull)

        typed_merge = TypeResolver(column_by_name={
            Aggregate.ACC: Param(Aggregate.ACC, initial_value_type),
            Aggregate.X: Param(Aggregate.X, element_type),
        }).typed(self._merge)
        merge_type = typed_merge.get_type()

        assert merge_type == initial_value_type

        lambda_code = Code().text(f"({Aggregate.ACC}, {Aggregate.X}) -> ")
        ExpressionGenerator(lambda_code).generate(typed_merge)

        self.full_type = FullType(
            merge_type.data_type,
            merge_type.nullable or array_type.nullable or initial_value_type.nullable,
        )
        self._lambda_code = lambda_code

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        assert self._lambda_code
        (
            h.text("list_reduce(CASE WHEN ")
            .arg(0)
            .text(" IS NULL THEN NULL ELSE list_prepend(")
            .arg(1)
            .text(", ")
            .arg(0)
            .text(") END, ")
            .text(self._lambda_code.to_string())
            .text(")")
        )


class Aggregate(Fn):
    VAR = 'namedlambdavariable()'
    ACC = 'acc'
    X = 'x'

    def __init__(
        self,
        array: 'BaseColumn', initial_value: 'BaseColumn',
        merge: 'BaseColumn', finish: 'BaseColumn',
        merge_name: str, finish_name: str,
    ):
        merge_lambda = f'lambdafunction({merge_name}, {self.VAR}, {self.VAR})'
        finish_lambda = f'lambdafunction({finish_name}, {self.VAR})'
        arg_names = [array.as_string(), initial_value.as_string(), merge_lambda, finish_lambda]
        reduce_column = _Reduce(array, initial_value, merge)
        super().__init__(f'aggregate({", ".join(arg_names)})', [reduce_column])
        self._finish = finish

    def set_type(self) -> None:
        from ._dataframe import TypeResolver

        typed_finish = TypeResolver(column_by_name={self.ACC: self.args[0]}).typed(self._finish)
        self.args = [typed_finish]
        self.full_type = typed_finish.get_type()

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.arg(0)


class CreateMap(Fn):
    def __init__(self, columns: list['BaseColumn']):
        assert len(columns) % 2 == 0
        super().__init__(f'map({", ".join([c.as_string() for c in columns])})', columns)

    def set_type(self) -> None:
        if self.args:
            key_type, key_args = _cast(self.args[::2])
            value_type, value_args = _cast(self.args[1::2])
            self.args = [e for p in zip(key_args, value_args) for e in p]
        else:
            key_type = FullType(T.NullType(), False)
            value_type = FullType(T.NullType(), False)
        map_type = T.MapType(key_type.data_type, value_type.data_type, value_type.nullable)
        self.full_type = FullType(map_type, False)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("MAP {")
        for i in range(0, len(self.args), 2):
            if i != 0:
                h.text(", ")
            h.arg(i).text(": ").arg(i + 1)
        h.text("}")


class MapKeys(Fn):
    def __init__(self, column: 'BaseColumn'):
        super().__init__(f'map_keys({column.as_string()})', [column])

    def set_type(self) -> None:
        arg_type = self.args[0].get_type()
        assert isinstance(arg_type.data_type, T.MapType)
        self.full_type = FullType(T.ArrayType(arg_type.data_type.keyType, False), arg_type.nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("map_keys(").arg(0).text(")")


class MapValues(Fn):
    def __init__(self, column: 'BaseColumn'):
        super().__init__(f'map_values({column.as_string()})', [column])

    def set_type(self) -> None:
        arg_type = self.args[0].get_type()
        map_type = arg_type.data_type
        assert isinstance(map_type, T.MapType)
        array_type = T.ArrayType(map_type.valueType, map_type.valueContainsNull)
        self.full_type = FullType(array_type, arg_type.nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("map_values(").arg(0).text(")")


class MapConcat(Fn):
    def __init__(self, columns: list['BaseColumn']):
        super().__init__(f'map_concat({", ".join([c.as_string() for c in columns])})', columns)

    def set_type(self) -> None:
        arg_types = [arg.get_type() for arg in self.args]
        if arg_types:
            self.full_type = get_common_type(arg_types)
            assert isinstance(self.full_type.data_type, T.MapType)
        else:
            self.full_type = FullType(T.MapType(T.StringType(), T.StringType(), False), False)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        if self.args:
            h.text("map_from_entries(list_concat(")
            for i in range(len(self.args)):
                if i != 0:
                    h.text(", ")
                h.text("map_entries(").arg(i).text(")")
            h.text("))")
        else:
            h.text("MAP {}")


class FromJson(Fn):
    def __init__(self, column: 'BaseColumn', data_type: 'T.DataType'):
        super().__init__(f'entries', [column])
        self._data_type = data_type

    def set_type(self) -> None:
        _, self.args = _cast(self.args, T.StringType())
        self.full_type = FullType(self._data_type, True)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.arg(0).text("::JSON::").text(type_literal(self._data_type))


class Trim(Fn):
    def __init__(self, column: 'BaseColumn'):
        super().__init__(f'trim({column.as_string()})', [column])

    def set_type(self) -> None:
        arg_type = self.args[0].get_type()
        assert isinstance(arg_type.data_type, T.StringType)
        self.full_type = FullType(T.StringType(), arg_type.nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("trim(").arg(0).text(")")


class Substring(Fn):
    def __init__(self, column: 'BaseColumn', position: int, length: int):
        super().__init__(f'substring({column.as_string()}, {position}, {length})', [column])
        self._position = 1 if position == 0 else position
        self._length = max(length, 0)

    def set_type(self) -> None:
        self.full_type, self.args = _cast(self.args, T.StringType())

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("substring(").arg(0).text(f", {self._position}, {self._length})")


class Coalesce(Fn):
    def __init__(self, columns: list['BaseColumn']):
        super().__init__(f'coalesce({", ".join([c.as_string() for c in columns])})', columns)

    def set_type(self) -> None:
        arg_types = [FullType(arg.get_type().data_type, False) for arg in self.args]
        common_type = get_common_type(arg_types)

        args = []
        nullable = True
        for arg in self.args:
            arg = Cast.internal(arg, common_type.data_type)
            args.append(arg)
            nullable = arg.get_type().nullable
            if not nullable:
                break

        self.args = args
        self.full_type = FullType(common_type.data_type, nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("coalesce(")
        for i in range(len(self.args)):
            if i != 0:
                h.text(", ")
            h.arg(i)
        h.text(")")


class Hash(Fn):
    def __init__(self, columns: list['BaseColumn']):
        super().__init__(f'hash({", ".join([c.as_string() for c in columns])})', columns)

    def set_type(self) -> None:
        self.full_type = FullType(T.IntegerType(), False)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("hash(")
        for i in range(len(self.args)):
            if i != 0:
                h.text(", ")
            h.arg(i)
        h.text(")")


class GetItem(Fn):
    def __init__(self, column: 'BaseColumn', key: 'BaseColumn'):
        super().__init__(f'{column.as_string()}[{key.as_string()}]', [column, key])

    def set_type(self) -> None:
        col_type = self.args[0].get_type()
        key_arg = self.args[1]
        if isinstance(col_type.data_type, T.StructType):
            assert isinstance(key_arg, Lit)
            assert isinstance(key_arg.value, str)
            self.name = f'{self.args[0].as_string()}.{key_arg.value}'
            field = col_type.data_type[key_arg.value]
            self.full_type = FullType(field.dataType, col_type.nullable or field.nullable)
        elif isinstance(col_type.data_type, T.ArrayType):
            assert isinstance(key_arg.get_type().data_type, T.NumericType)
            self.full_type = FullType(col_type.data_type.elementType, True)
        else:
            assert isinstance(col_type.data_type, T.MapType)
            self.full_type = FullType(col_type.data_type.valueType, True)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        data_type = self.args[0].get_type().data_type
        if isinstance(data_type, T.StructType):
            key_arg = self.args[1]
            assert isinstance(key_arg, Lit)
            assert isinstance(key_arg.value, str)
            h.arg(0).text(f'."{escape_name(key_arg.value)}"')
        elif isinstance(data_type, T.ArrayType):
            h.arg(0).text("[").arg(1).text(" + 1]")
        else:
            h.arg(0).text("[").arg(1).text("]")


class CurrentTimestamp(Fn):
    def __init__(self):
        super().__init__('current_timestamp()', [])

    def set_type(self) -> None:
        self.full_type = FullType(T.TimestampType(), False)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("current_localtimestamp()")


class MonotonicallyIncreasingId(Fn):
    def __init__(self):
        super().__init__('monotonically_increasing_id()', [])

    def set_type(self) -> None:
        self.full_type = FullType(T.LongType(), False)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("row_number() OVER ()")


class UpperLower(Fn):
    def __init__(self, name: str, column: 'BaseColumn'):
        super().__init__(f'{name}({column.as_string()})', [column])
        self._function = name

    def set_type(self) -> None:
        self.full_type, self.args = _cast(self.args, T.StringType())

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text(self._function).text("(").arg(0).text(")")


class Abs(Fn):
    def __init__(self, column: 'BaseColumn'):
        super().__init__(f'abs({column.as_string()})', [column])

    def set_type(self) -> None:
        self.full_type, self.args = double_on_string(self.args)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("abs(").arg(0).text(")")


class Round(Fn):
    def __init__(self, column: 'BaseColumn', scale: int):
        super().__init__(f'round({column.as_string()}, {scale})', [column])
        self._scale = scale

    def set_type(self) -> None:
        self.full_type, self.args = double_on_string(self.args)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("round(").arg(0).text(f", {self._scale})")


# aggregations

class RowNumber(Agg):
    def __init__(self):
        super().__init__('row_number()', [])

    def set_type(self) -> None:
        self.full_type = FullType(T.IntegerType(), False)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("row_number()")


class Rank(Agg):
    def __init__(self):
        super().__init__('rank()', [])

    def set_type(self) -> None:
        self.full_type = FullType(T.IntegerType(), False)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("rank()")


class DenseRank(Agg):
    def __init__(self):
        super().__init__('dense_rank()', [])

    def set_type(self) -> None:
        self.full_type = FullType(T.IntegerType(), False)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("dense_rank()")


class LagLead(Agg):
    def __init__(self, name: str, column: 'BaseColumn', offset: int, default: 'BaseColumn'):
        self._name = name
        self._offset = offset
        args = [column, default]
        super().__init__(f'{name}({column.as_string()}, {offset}, {default.as_string()})', args)

    def set_type(self) -> None:
        self.full_type, self.args = _cast(self.args)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text(f"{self._name}(").arg(0).text(f", {self._offset}, ").arg(1).text(")")


class Count(Agg):
    def __init__(self, column: Optional['BaseColumn']):
        name = f'count({column.as_string() if column else 1})'
        super().__init__(name, [column] if column else [])

    def set_type(self) -> None:
        self.full_type = FullType(T.LongType(), False)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("count(")
        if self.args:
            h.arg(0)
        else:
            h.text("*")
        h.text(")")


class ApproxCountDistinct(Agg):
    def __init__(self, column: 'BaseColumn'):
        super().__init__(f'approx_count_distinct({column.as_string()})', [column])

    def set_type(self) -> None:
        self.full_type = FullType(T.LongType(), False)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("approx_count_distinct(").arg(0).text(")")


class FirstLast(Agg):
    def __init__(self, name: str, column: 'BaseColumn', ignore_nulls: bool):
        super().__init__(f'{name}({column.as_string()})', [column])
        self._function = name
        self._ignore_nulls = ignore_nulls

    def set_type(self) -> None:
        self.full_type = self.args[0].get_type()

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        suffix = " IGNORE NULLS" if self._ignore_nulls else ""
        h.text(f"{self._function}(").arg(0).text(f"{suffix})")


class MinMax(Agg):
    def __init__(self, name: str, column: 'BaseColumn'):
        super().__init__(f'{name}({column.as_string()})', [column])
        self._function = name

    def set_type(self) -> None:
        self.full_type = self.args[0].get_type()

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text(self._function).text("(").arg(0).text(")")


class MinMaxBy(Agg):
    def __init__(self, name: str, value: 'BaseColumn', order: 'BaseColumn'):
        super().__init__(f'{name}({value.as_string()}, {order.as_string()})', [value, order])
        if name == 'min_by':
            self._function = 'first'
            self._order_spec = 'NULLS LAST'
        else:
            self._function = 'last'
            self._order_spec = 'NULLS FIRST'

    def set_type(self) -> None:
        value_type = self.args[0].get_type()
        order_type = self.args[1].get_type()
        self.full_type = FullType(value_type.data_type, value_type.nullable or order_type.nullable)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text(f"{self._function}(").arg(0).text(" ORDER BY ").arg(1).text(f" {self._order_spec})")


class Sum(Agg):
    def __init__(self, column: 'BaseColumn'):
        super().__init__(f'sum({column.as_string()})', [column])

    def set_type(self) -> None:
        sum_type = (
            T.LongType()
            if isinstance(self.args[0].get_type().data_type, T.IntegralType)
            else T.DoubleType()
        )
        self.full_type, self.args = _cast(self.args, sum_type)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text("sum(").arg(0).text(")")


class Collect(Agg):
    def __init__(self, collection: str, column: 'BaseColumn'):
        super().__init__(f'collect_{collection}({column.as_string()})', [column])
        self._collection = collection

    def set_type(self) -> None:
        self.full_type = FullType(T.ArrayType(self.args[0].get_type().data_type, False), True)

    def finish(self, column: 'BaseColumn') -> 'BaseColumn':
        a = Array([])
        a.set_type()
        c = Coalesce([column, a])
        c.set_type()
        return c

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        prefix = "DISTINCT " if self._collection == "set" else ""
        h.text(f"list({prefix}").arg(0).text(") FILTER (").arg(0).text(" IS NOT NULL)")
