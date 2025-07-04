from typing import Any, Optional, Union


class DataType:
    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and other.__dict__ == self.__dict__

    def __hash__(self) -> int:
        return hash(str(self))

    def simpleString(self) -> str:
        return self.typeName()

    @classmethod
    def typeName(cls) -> str:
        return cls.__name__[:-4].lower()


class NullType(DataType):
    @classmethod
    def typeName(cls) -> str:
        return "void"


class AtomicType(DataType):
    pass


class BooleanType(AtomicType):
    pass


class BinaryType(AtomicType):
    pass


class StringType(AtomicType):
    pass


class NumericType(AtomicType):
    pass


class IntegralType(NumericType):
    pass


class FractionalType(NumericType):
    pass


class ByteType(IntegralType):
    def simpleString(self) -> str:
        return 'tinyint'


class ShortType(IntegralType):
    def simpleString(self) -> str:
        return 'smallint'


class IntegerType(IntegralType):
    MIN = -2 ** 31
    MAX = 2 ** 31 - 1

    def simpleString(self) -> str:
        return 'int'


class LongType(IntegralType):
    def simpleString(self) -> str:
        return 'bigint'


class DecimalType(FractionalType):
    pass


class FloatType(FractionalType):
    pass


class DoubleType(FractionalType):
    pass


class DateType(AtomicType):
    pass


class TimestampType(AtomicType):
    pass


class StructField(DataType):
    def __init__(self, name: str, dataType: 'DataType', nullable: bool = True):
        self.name = name
        self.dataType = dataType
        self.nullable = nullable

    def __repr__(self) -> str:
        return 'StructField(%r, %r, %r)' % (self.name, self.dataType, self.nullable)

    def simpleString(self) -> str:
        return '%s:%s' % (self.name, self.dataType.simpleString())


class StructType(DataType):
    def __init__(self, fields: Optional[list['StructField']] = None):
        self.fields: list['StructField'] = [] if fields is None else fields
        self.names = [f.name for f in self.fields]

    def __repr__(self) -> str:
        return 'StructType(%r)' % self.fields

    def __getitem__(self, key: Union[str, int]) -> 'StructField':
        if isinstance(key, str):
            key = self.names.index(key)
        return self.fields[key]

    def simpleString(self) -> str:
        return 'struct<%s>' % (','.join(f.simpleString() for f in self.fields))

    def add(
        self,
        field: Union[str, 'StructField'],
        data_type: Optional['DataType'] = None,
        nullable: bool = True,
    ) -> 'StructType':
        if isinstance(field, str):
            assert data_type is not None
            field = StructField(field, data_type, nullable)
        self.fields.append(field)
        self.names.append(field.name)
        return self


class ArrayType(DataType):
    def __init__(self, elementType: 'DataType', containsNull: bool = True):
        self.elementType = elementType
        self.containsNull = containsNull

    def __repr__(self) -> str:
        return 'ArrayType(%r, %r)' % (self.elementType, self.containsNull)

    def simpleString(self) -> str:
        return 'array<%s>' % self.elementType.simpleString()


class MapType(DataType):
    def __init__(self, keyType: 'DataType', valueType: 'DataType', valueContainsNull: bool = True):
        self.keyType = keyType
        self.valueType = valueType
        self.valueContainsNull = valueContainsNull

    def __repr__(self) -> str:
        return "MapType(%s, %s, %s)" % (self.keyType, self.valueType, self.valueContainsNull)

    def simpleString(self) -> str:
        return "map<%s,%s>" % (self.keyType.simpleString(), self.valueType.simpleString())


class Row(tuple):
    def __new__(cls, *args: tuple[str, object], **kwargs: object):
        assert not (args and kwargs)
        if args:
            row = tuple.__new__(cls, (v for _, v in args))
            row.__fields__ = tuple(k for k, _ in args)  # type: ignore[attr-defined]
        else:
            row = tuple.__new__(cls, kwargs.values())
            row.__fields__ = tuple(kwargs.keys())  # type: ignore[attr-defined]
        return row

    def asDict(self) -> dict[str, object]:
        return dict(zip(self.__fields__, self))

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            item = self.__fields__.index(item)
        return super().__getitem__(item)

    def __getattr__(self, item: str) -> Any:
        return self[item]

    def __repr__(self) -> str:
        return 'Row(%s)' % ', '.join('%s=%r' % (k, v) for k, v in zip(self.__fields__, self))
