from typing import TYPE_CHECKING, Union

from .._internal import _functions as F
from .._internal._column import Agg, ColBy, Lit, Order, Over, WhenBasic, WhenComposite
from .._internal._types import data_type_from_string
from .window import Window

if TYPE_CHECKING:
    from .._internal._column import BaseColumn
    from . import types as T
    from ._typing import ColumnOrName, ColumnOrValue
    from .window import WindowSpec


class Column:
    def __init__(self, column: 'BaseColumn'):
        self._column = column

    def __repr__(self) -> str:
        return "Column<'%s'>" % self._column.as_string()

    @classmethod
    def _get_column_from_name(cls, name: 'ColumnOrName') -> 'BaseColumn':
        return name._column if isinstance(name, Column) else ColBy(name)

    @classmethod
    def _get_column_from_value(cls, value: 'ColumnOrValue') -> 'BaseColumn':
        return value._column if isinstance(value, Column) else Lit.of(value)

    def alias(self, alias: str) -> 'Column':
        return Column(self._column.with_alias(alias))

    def over(self, window: 'WindowSpec') -> 'Column':
        assert isinstance(self._column, Agg)
        partition_columns = [self._get_column_from_name(c) for c in window._partition_columns]
        order_columns = [self._get_column_from_name(c) for c in window._order_columns]

        rows_start_end = window._rows_start_end
        if rows_start_end is None:
            if order_columns:
                rows_start_end = False, Window.unboundedPreceding, Window.currentRow
            else:
                rows_start_end = True, Window.unboundedPreceding, Window.unboundedFollowing

        columns = []
        specs = []
        for c in order_columns:
            if isinstance(c, Order):
                columns.append(c.column)
                specs.append(c.spec())
            else:
                columns.append(c)
                specs.append('ASC NULLS FIRST')

        return Column(Over(self._column, partition_columns, columns, specs, rows_start_end))

    def when(self, condition: 'Column', value: 'ColumnOrValue') -> 'Column':
        assert (
                isinstance(self._column, WhenBasic) or
                isinstance(self._column, WhenComposite) and isinstance(self._column.otherwise, WhenBasic)
        )
        otherwise = WhenBasic(condition._column, self._get_column_from_value(value))
        return Column(WhenComposite(self._column, otherwise))

    def otherwise(self, value: 'ColumnOrValue') -> 'Column':
        assert (
                isinstance(self._column, WhenBasic) or
                isinstance(self._column, WhenComposite) and isinstance(self._column.otherwise, WhenBasic)
        )
        return Column(WhenComposite(self._column, self._get_column_from_value(value)))

    def eqNullSafe(self, other: 'ColumnOrValue') -> 'Column':
        return Column(F.EqNullSafe(self._column, self._get_column_from_value(other)))

    def __eq__(self, other: 'ColumnOrValue') -> 'Column':  # type: ignore[override]
        return Column(F.Equal(self._column, self._get_column_from_value(other)))

    def __ne__(self, other: 'ColumnOrValue') -> 'Column':  # type: ignore[override]
        return ~(self == other)

    def __le__(self, other: 'ColumnOrValue') -> 'Column':
        return Column(F.Compare('<=', self._column, self._get_column_from_value(other)))

    def __ge__(self, other: 'ColumnOrValue') -> 'Column':
        return Column(F.Compare('>=', self._column, self._get_column_from_value(other)))

    def __lt__(self, other: 'ColumnOrValue') -> 'Column':
        return Column(F.Compare('<', self._column, self._get_column_from_value(other)))

    def __gt__(self, other: 'ColumnOrValue') -> 'Column':
        return Column(F.Compare('>', self._column, self._get_column_from_value(other)))

    def __invert__(self) -> 'Column':
        return Column(F.Not(self._column))

    def __and__(self, other: 'ColumnOrValue') -> 'Column':
        return Column(F.LogicalBinOp('AND', self._column, self._get_column_from_value(other)))

    def __or__(self, other: 'ColumnOrValue') -> 'Column':
        return Column(F.LogicalBinOp('OR', self._column, self._get_column_from_value(other)))

    def __neg__(self) -> 'Column':
        return Column(F.Neg(self._column))

    def __add__(self, other: 'ColumnOrValue') -> 'Column':
        return Column(F.ArithmeticBinOp('+', self._column, self._get_column_from_value(other)))

    def __radd__(self, other: 'ColumnOrValue') -> 'Column':
        return self + other

    def __sub__(self, other: 'ColumnOrValue') -> 'Column':
        return Column(F.ArithmeticBinOp('-', self._column, self._get_column_from_value(other)))

    def __rsub__(self, other: 'ColumnOrValue') -> 'Column':
        return Column(F.ArithmeticBinOp('-', self._get_column_from_value(other), self._column))

    def __mul__(self, other: 'ColumnOrValue') -> 'Column':
        return Column(F.ArithmeticBinOp('*', self._column, self._get_column_from_value(other)))

    def __rmul__(self, other: 'ColumnOrValue') -> 'Column':
        return self * other

    def __truediv__(self, other: 'ColumnOrValue') -> 'Column':
        return Column(F.Div(self._column, self._get_column_from_value(other)))

    def __rtruediv__(self, other: 'ColumnOrValue') -> 'Column':
        return Column(F.Div(self._get_column_from_value(other), self._column))

    def isNull(self) -> 'Column':
        return Column(F.Null(True, self._column))

    def isNotNull(self) -> 'Column':
        return Column(F.Null(False, self._column))

    def between(self, lowerBound: 'ColumnOrValue', upperBound: 'ColumnOrValue') -> 'Column':
        return (self >= lowerBound) & (self <= upperBound)

    def cast(self, dataType: Union['T.DataType', str]) -> 'Column':
        data_type = data_type_from_string(dataType) if isinstance(dataType, str) else dataType
        return Column(F.Cast(self._column, data_type))

    def isin(self, *cols: Union['ColumnOrValue', set]) -> 'Column':
        cs: list['ColumnOrValue'] = []
        for c in cols:
            if isinstance(c, (list, set)):
                cs.extend(c)
            else:
                cs.append(c)
        return Column(F.IsIn(self._column, [self._get_column_from_value(c) for c in cs]))

    def startswith(self, other: Union['Column', str]) -> 'Column':
        return Column(F.StartsEndsWith('starts', self._column, Column._get_column_from_value(other)))

    def endswith(self, other: Union['Column', str]) -> 'Column':
        return Column(F.StartsEndsWith('ends', self._column, Column._get_column_from_value(other)))

    def contains(self, other: Union['Column', str]) -> 'Column':
        return Column(F.Contains(self._column, Column._get_column_from_value(other)))

    def like(self, other: str) -> 'Column':
        return Column(F.Like(self._column, other))

    def rlike(self, other: str) -> 'Column':
        return Column(F.RLike(self._column, other))

    def asc(self) -> 'Column':
        return Column(Order(self._column, False, False))

    def desc(self) -> 'Column':
        return Column(Order(self._column, True, True))

    def getField(self, name: str) -> 'Column':
        return self[name]

    def getItem(self, key: Union[str, int, 'Column']) -> 'Column':
        return self[key]

    def __getitem__(self, item: Union[str, int, 'Column']) -> 'Column':
        return Column(F.GetItem(self._column, Column._get_column_from_value(item)))
