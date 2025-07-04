import math
from datetime import date, datetime
from typing import Callable, Optional, TYPE_CHECKING

from ..sql import types as T

if TYPE_CHECKING:
    from .._typing import CellType, LiteralType


class FullType:
    def __init__(self, data_type: 'T.DataType', nullable: bool):
        self.data_type = data_type
        self.nullable = nullable

    def __repr__(self) -> str:
        return f'{self.data_type}{"?" if self.nullable else ""}'

    def __eq__(self, other: object) -> bool:
        return str(other) == str(self)

    def __hash__(self) -> int:
        return hash(str(self))


_PRIORITIZED_TYPES: list['T.DataType'] = [
    T.IntegerType(),
    T.LongType(),
    T.DoubleType(),
    T.StringType(),
]


# source type, target type, nullable
_ALLOWED_CASTS: dict[tuple['T.DataType', 'T.DataType'], bool] = {
    (T.StringType(), T.BooleanType()): False,
    (T.StringType(), T.IntegerType()): True,
    (T.StringType(), T.LongType()): True,
    (T.StringType(), T.DoubleType()): True,
    (T.StringType(), T.DateType()): True,

    (T.BooleanType(), T.StringType()): False,
    (T.BooleanType(), T.IntegerType()): False,

    (T.IntegerType(), T.LongType()): False,
    (T.IntegerType(), T.DoubleType()): False,
    (T.IntegerType(), T.StringType()): False,
    (T.LongType(), T.IntegerType()): False,
    (T.LongType(), T.DoubleType()): False,
    (T.LongType(), T.StringType()): False,

    (T.FloatType(), T.DoubleType()): False,
    (T.DoubleType(), T.StringType()): False,

    (T.DateType(), T.StringType()): False,
    (T.TimestampType(), T.StringType()): False,
    (T.TimestampType(), T.DateType()): False,
}


def is_nullable_cast(from_type: 'T.DataType', to_type: 'T.DataType') -> bool:
    if from_type == to_type:
        return False
    if from_type == T.NullType():
        return True
    if isinstance(from_type, T.ArrayType) and isinstance(to_type, T.ArrayType):
        return False
    if isinstance(from_type, T.MapType) and isinstance(to_type, T.MapType):
        return False
    return _ALLOWED_CASTS[(from_type, to_type)]


def get_common_type(types: list['FullType']) -> 'FullType':
    common_type = types[0]
    for type in types[1:]:
        current_type = _common_type(common_type, type)
        assert current_type, f'No common type between {[common_type, type]}'
        common_type = current_type
    return common_type


def _common_type(a: 'FullType', b: 'FullType') -> Optional['FullType']:
    nullable = a.nullable or b.nullable

    if isinstance(a.data_type, T.NullType):
        return FullType(b.data_type, nullable)
    if isinstance(b.data_type, T.NullType):
        return FullType(a.data_type, nullable)

    if isinstance(a.data_type, T.StringType) and isinstance(b.data_type, T.AtomicType):
        return FullType(T.StringType(), nullable)
    if isinstance(b.data_type, T.StringType) and isinstance(a.data_type, T.AtomicType):
        return FullType(T.StringType(), nullable)

    if a.data_type in _PRIORITIZED_TYPES and b.data_type in _PRIORITIZED_TYPES:
        data_type = max(a.data_type, b.data_type, key=lambda t: _PRIORITIZED_TYPES.index(t))
        return FullType(data_type, nullable)

    if isinstance(a.data_type, T.ArrayType) and isinstance(b.data_type, T.ArrayType):
        element_type = _common_type(
            FullType(a.data_type.elementType, a.data_type.containsNull),
            FullType(b.data_type.elementType, b.data_type.containsNull),
        )
        if element_type is None:
            return None
        array_type = T.ArrayType(element_type.data_type, element_type.nullable)
        return FullType(array_type, nullable)

    if isinstance(a.data_type, T.MapType) and isinstance(b.data_type, T.MapType):
        key_type = _common_type(
            FullType(a.data_type.keyType, False),
            FullType(b.data_type.keyType, False),
        )
        value_type = _common_type(
            FullType(a.data_type.valueType, a.data_type.valueContainsNull),
            FullType(b.data_type.valueType, b.data_type.valueContainsNull),
        )
        if key_type is None or value_type is None:
            return None
        map_type = T.MapType(key_type.data_type, value_type.data_type, value_type.nullable)
        return FullType(map_type, nullable)

    return FullType(a.data_type, nullable) if a.data_type == b.data_type else None


_FULL_TYPE_BY_TYPE = {
    type(None): FullType(T.NullType(), True),
    bool: FullType(T.BooleanType(), False),
    float: FullType(T.DoubleType(), False),
    str: FullType(T.StringType(), False),
    date: FullType(T.DateType(), False),
    datetime: FullType(T.TimestampType(), False),
}


def full_type_from_value(value: 'CellType') -> 'FullType':
    full_type = _FULL_TYPE_BY_TYPE.get(type(value))
    if full_type:
        return full_type
    if isinstance(value, int):
        if T.IntegerType.MIN <= value <= T.IntegerType.MAX:
            return FullType(T.IntegerType(), False)
        else:
            return FullType(T.LongType(), False)
    assert isinstance(value, list), type(value)
    if len(value) == 0:
        return FullType(T.ArrayType(T.NullType(), False), False)
    common_type = get_common_type([full_type_from_value(e) for e in value])
    return FullType(T.ArrayType(common_type.data_type, common_type.nullable), False)


_TYPE_LITERAL_BY_DATA_TYPE: dict['T.DataType', str] = {
    T.NullType(): 'INT',
    T.BooleanType(): 'BOOLEAN',
    T.ByteType(): 'TINYINT',
    T.ShortType(): 'SMALLINT',
    T.IntegerType(): 'INT',
    T.LongType(): 'BIGINT',
    T.FloatType(): 'FLOAT',
    T.DoubleType(): 'DOUBLE',
    T.DateType(): 'DATE',
    T.TimestampType(): 'TIMESTAMP',
    T.StringType(): 'STRING',
}


def type_literal(data_type: 'T.DataType') -> str:
    if isinstance(data_type, T.StructType):
        fields = [f'"{escape_name(f.name)}" {type_literal(f.dataType)}' for f in data_type.fields]
        return f"STRUCT({', '.join(fields)})"
    if isinstance(data_type, T.ArrayType):
        return f"{type_literal(data_type.elementType)}[]"
    if isinstance(data_type, T.MapType):
        return f"MAP({type_literal(data_type.keyType)}, {type_literal(data_type.valueType)})"
    return _TYPE_LITERAL_BY_DATA_TYPE[data_type]


def value_literal(data_type: 'T.DataType', value: 'LiteralType') -> str:
    if value is None:
        return f"NULL::{type_literal(data_type)}"
    if isinstance(data_type, T.StringType):
        assert isinstance(value, str)
        return f"'{escape_string(value)}'"
    if isinstance(data_type, T.DateType):
        assert isinstance(value, date)
        return f"DATE '{value.isoformat()}'"
    if isinstance(data_type, T.TimestampType):
        assert isinstance(value, datetime)
        return f"TIMESTAMP '{value.isoformat().replace('T', ' ')}'"
    if isinstance(data_type, T.BooleanType):
        literal = 'true' if value is True else 'false'
    elif isinstance(data_type, T.IntegralType):
        assert isinstance(value, int)
        literal = str(value)
    elif isinstance(data_type, T.FractionalType):
        assert isinstance(value, (int, float))
        if math.isnan(value) or math.isinf(value):
            value = f"'{value}'"
        literal = f"{value}::{type_literal(data_type)}"
    elif isinstance(data_type, T.StructType):
        if isinstance(value, dict):
            value = tuple(value.get(f) for f in data_type.names)
        assert isinstance(value, tuple)
        entries = ', '.join([
            f'"{escape_name(f.name)}": {value_literal(f.dataType, v)}'
            for f, v in zip(data_type.fields, value)
        ])
        literal = f"{{{entries}}}"
    elif isinstance(data_type, T.ArrayType):
        assert isinstance(value, list)
        literal = f"[{', '.join([value_literal(data_type.elementType, v) for v in value])}]"
    elif isinstance(data_type, T.MapType):
        assert isinstance(value, dict)
        entries = ', '.join([
            f"{value_literal(data_type.keyType, k)}: {value_literal(data_type.valueType, v)}"
            for k, v in value.items()
        ])
        literal = f"MAP {{{entries}}}"
    else:
        raise RuntimeError(data_type, value)
    return literal


def escape_string(value: str) -> str:
    return value.replace("'", "''")


def escape_name(name: str) -> str:
    return name.replace('"', '""')


def data_type_from_string(string: str) -> 'T.DataType':
    return TypeParser(string.lower()).parse()


class TypeParser:
    _TYPES: dict[str, tuple[int, Callable[[list[T.DataType]], 'T.DataType']]] = {
        'void': (0, lambda data_types: T.NullType()),
        'boolean': (0, lambda data_types: T.BooleanType()),
        'tinyint': (0, lambda data_types: T.ByteType()),
        'smallint': (0, lambda data_types: T.ShortType()),
        'int': (0, lambda data_types: T.IntegerType()),
        'bigint': (0, lambda data_types: T.LongType()),
        'float': (0, lambda data_types: T.FloatType()),
        'double': (0, lambda data_types: T.DoubleType()),
        'date': (0, lambda data_types: T.DateType()),
        'timestamp': (0, lambda data_types: T.TimestampType()),
        'string': (0, lambda data_types: T.StringType()),
        'array': (1, lambda data_types: T.ArrayType(data_types[0], True)),
        'map': (2, lambda data_types: T.MapType(data_types[0], data_types[1], True)),
    }

    def __init__(self, string: str):
        self._string = string
        self._i = 0

    def parse(self) -> 'T.DataType':
        data_type = self._parse_type()
        if self._i != len(self._string):
            raise RuntimeError(f'Failed to parse {self._string} - '
                               f'did not consume the entire string '
                               f'({self._i} != {len(self._string)})')
        return data_type

    def _parse_type(self) -> 'T.DataType':
        data_type_name = self._parse_identifier()
        if data_type_name == 'struct':
            fields = []
            self._consume('<')
            i = 0
            while self._string[self._i] != '>':
                if i != 0:
                    self._consume(',')
                field_name = self._parse_identifier()
                self._consume(':')
                data_type = self._parse_type()
                fields.append(T.StructField(field_name, data_type))
                i += 1
            self._consume('>')
            return T.StructType(fields)
        else:
            param_count, factory = self._TYPES[data_type_name]
            data_types = []
            if param_count != 0:
                self._consume('<')
                for i in range(param_count):
                    if i != 0:
                        self._consume(',')
                    data_type = self._parse_type()
                    data_types.append(data_type)
                self._consume('>')
            return factory(data_types)

    def _parse_identifier(self) -> str:
        cs = []
        while self._i != len(self._string):
            c = self._string[self._i]
            if c == '_' or c.isalnum():
                cs.append(c)
            else:
                break
            self._i += 1
        return ''.join(cs)

    def _consume(self, c: str) -> None:
        if self._string[self._i] != c:
            raise RuntimeError(f'Failed to parse {self._string} - expected {c} at {self._i}')
        self._i += 1
