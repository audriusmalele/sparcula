import math
from copy import copy
from datetime import date, datetime
from typing import TYPE_CHECKING, Callable, Optional, Union

from ._types import FullType, escape_name, full_type_from_value, value_literal
from ..sql.window import Window

if TYPE_CHECKING:
    from ._dataframe import Code, ColumnCodeHelper, BaseDataFrame
    from .._typing import LiteralType
    from ..sql.types import DataType


def adjust_when_name(name: str) -> str:
    return f'CASE {name} END'


class BaseColumn:
    def __init__(
        self,
        name: str,
        full_type: Optional['FullType'] = None,
        args: Optional[list['BaseColumn']] = None,
    ):
        self.name = name
        self.full_type = full_type
        self.alias: Optional[str] = None
        self.args: list['BaseColumn'] = args or []

    def with_alias(self, alias: str) -> 'BaseColumn':
        copied = copy(self)
        copied.alias = alias
        return copied

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        raise NotImplementedError()

    def get_alias(self) -> str:
        return self.alias or self.name

    def get_type(self) -> 'FullType':
        assert self.full_type
        return self.full_type

    def as_string(self) -> str:
        name = self.name
        if isinstance(self, (WhenBasic, WhenComposite)):
            name = adjust_when_name(name)
        if self.alias is not None:
            name = f'{name} AS {self.alias}'
        return name

    def __repr__(self) -> str:
        return f'{self.as_string()}: {self.full_type}'


class Param(BaseColumn):
    def __init__(self, name: str, full_type: 'FullType'):
        super().__init__(name, full_type)

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text(self.name)


class Lit(BaseColumn):
    @staticmethod
    def of(value: 'LiteralType'):
        if isinstance(value, list):
            from ._functions import Array
            return Array([Lit.of(e) for e in value])
        return Lit(value)

    def __init__(self, value: 'LiteralType'):
        if value is None:
            name = 'NULL'
        elif isinstance(value, bool):
            name = 'true' if value else 'false'
        elif isinstance(value, datetime):
            name = f"TIMESTAMP '{value.isoformat().replace('T', ' ')}'"
        elif isinstance(value, date):
            name = f"DATE '{value.isoformat()}'"
        elif isinstance(value, float):
            if math.isnan(value):
                name = 'NaN'
            elif math.isinf(value):
                name = f'{"-" if value < 0 else ""}Infinity'
            else:
                name = str(value)
        else:
            name = str(value)
        full_type = full_type_from_value(value)
        super().__init__(name, full_type)
        self.value = value
        self.type = full_type

    def with_type(self, data_type: 'DataType') -> 'Lit':
        assert self.value is None
        full_type = FullType(data_type, True)
        copied = copy(self)
        copied.name = f'CAST({self.name} AS {data_type.simpleString().upper()})'
        copied.full_type = full_type
        copied.type = full_type
        return copied

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.text(value_literal(self.type.data_type, self.value))


class Col(BaseColumn):
    def __init__(self, name: str, full_type: 'FullType', di_fi: tuple[int, int]):
        super().__init__(name, full_type)
        self.di_fi = di_fi

    def __repr__(self) -> str:
        return f'{self.di_fi} -> {super().__repr__()}'

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        di, fi = self.di_fi
        prefix = ["l.", "r."][di] if h.generator.join else ""
        h.text(f'{prefix}"{fi + 1}:{escape_name(self.name)}"')


class ColBy(BaseColumn):
    def __init__(self, name: str, df: Optional['BaseDataFrame'] = None, fi: Optional[int] = None):
        super().__init__(name)
        self.df = df
        self.fi = fi


class Fn(BaseColumn):
    def __init__(self, name: str, args: list['BaseColumn']):
        super().__init__(name, None, args)

    def set_type(self) -> None:
        raise NotImplementedError()


class Agg(Fn):
    def finish(self, column: 'BaseColumn') -> 'BaseColumn':
        return column


class Over(BaseColumn):
    def __init__(
        self,
        agg_column: 'Agg',
        partition_columns: list['BaseColumn'],
        order_columns: list['BaseColumn'],
        order_specs: list[str],
        rows_start_end: tuple[bool, int, int],
    ):
        from ._dataframe import Code

        self.agg_column = agg_column
        self.partition_columns = partition_columns
        self.order_columns = order_columns
        self.order_specs = order_specs
        self.rows_start_end = rows_start_end

        code = Code()
        self._window(code, lambda c: code.text(c.name))

        super().__init__(f'{agg_column.name} OVER ({code.to_string()})')

    def _window(self, code: 'Code', f: Callable[['BaseColumn'], object]) -> None:
        if self.partition_columns:
            code.text("PARTITION BY ")
            for i, c in enumerate(self.partition_columns):
                if i != 0:
                    code.text(", ")
                f(c)
        if self.order_columns:
            if self.partition_columns:
                code.text(" ")
            code.text("ORDER BY ")
            for i, (c, s) in enumerate(zip(self.order_columns, self.order_specs)):
                if i != 0:
                    code.text(", ")
                f(c)
                code.text(" ").text(s)
        if self.partition_columns or self.order_columns:
            code.text(" ")
        rows, start, end = self.rows_start_end
        (
            code.text("ROWS" if rows else "RANGE")
            .text(" BETWEEN ")
            .text(self._bound(start))
            .text(" AND ")
            .text(self._bound(end))
        )

    @classmethod
    def _bound(cls, bound: int) -> str:
        if bound == Window.unboundedPreceding:
            return 'UNBOUNDED PRECEDING'
        elif bound == Window.unboundedFollowing:
            return 'UNBOUNDED FOLLOWING'
        elif bound == Window.currentRow:
            return 'CURRENT ROW'
        else:
            return f'{bound} FOLLOWING'

    def _window_sql(self) -> str:
        from ._dataframe import Code, ExpressionGenerator

        code = Code()
        generator = ExpressionGenerator(code)
        self._window(code, lambda c: generator.generate(c))
        return code.to_string()

    def set_type(self) -> None:
        self.full_type = self.agg_column.get_type()

    def generate_code(self, h: 'ColumnCodeHelper') -> None:
        h.arg_column(self.agg_column).text(" OVER ").window(self._window_sql())


class Explode(BaseColumn):
    def __init__(self, array_column: 'BaseColumn', outer: bool):
        name = f'explode({array_column.name})'
        super().__init__(name)
        self.alias = 'col'
        self.array_column = array_column
        self.outer = outer


class WhenBasic(BaseColumn):
    def __init__(self, condition: 'BaseColumn', value: 'BaseColumn'):
        name = 'WHEN %s THEN %s' % (condition.name, value.name)
        super().__init__(name)
        self.condition = condition
        self.value = value


class WhenComposite(BaseColumn):
    def __init__(self, parent: Union['WhenBasic', 'WhenComposite'], otherwise: 'BaseColumn'):
        if isinstance(otherwise, WhenBasic):
            name = '%s %s' % (parent.name, otherwise.name)
        else:
            name = '%s ELSE %s' % (parent.name, otherwise.name)
        super().__init__(name)
        self.parent = parent
        self.otherwise = otherwise


class Order(BaseColumn):
    def __init__(self, column: 'BaseColumn', descending: bool, nulls_last: bool):
        self.column = column
        self.descending = descending
        self.nulls_last = nulls_last
        super().__init__(f'{column.name} {self.spec()}')

    def spec(self) -> str:
        order = 'DESC' if self.descending else 'ASC'
        nulls = 'NULLS LAST' if self.nulls_last else 'NULLS FIRST'
        return f'{order} {nulls}'
