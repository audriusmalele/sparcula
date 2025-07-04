from copy import copy
from typing import Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ._typing import ColumnOrName


def _flatten(*cols: Union['ColumnOrName', list['ColumnOrName']]) -> list['ColumnOrName']:
    flat = []
    for col in cols:
        if isinstance(col, list):
            flat.extend(col)
        else:
            flat.append(col)
    return flat


class Window:
    unboundedPreceding: int = -(1 << 63)
    unboundedFollowing: int = (1 << 63) - 1
    currentRow: int = 0

    @staticmethod
    def partitionBy(*cols: Union['ColumnOrName', list['ColumnOrName']]) -> 'WindowSpec':
        return WindowSpec().partitionBy(*cols)

    @staticmethod
    def orderBy(*cols: Union['ColumnOrName', list['ColumnOrName']]) -> 'WindowSpec':
        return WindowSpec().orderBy(*cols)

    @staticmethod
    def rowsBetween(start: int, end: int) -> 'WindowSpec':
        return WindowSpec().rowsBetween(start, end)

    @staticmethod
    def rangeBetween(start: int, end: int) -> 'WindowSpec':
        return WindowSpec().rangeBetween(start, end)


class WindowSpec:
    def __init__(self) -> None:
        self._partition_columns: list['ColumnOrName'] = []
        self._order_columns: list['ColumnOrName'] = []
        self._rows_start_end: Optional[tuple[bool, int, int]] = None

    def partitionBy(self, *cols: Union['ColumnOrName', list['ColumnOrName']]) -> 'WindowSpec':
        copied = copy(self)
        copied._partition_columns = _flatten(*cols)
        return copied

    def orderBy(self, *cols: Union['ColumnOrName', list['ColumnOrName']]) -> 'WindowSpec':
        copied = copy(self)
        copied._order_columns = _flatten(*cols)
        return copied

    def rowsBetween(self, start: int, end: int) -> 'WindowSpec':
        copied = copy(self)
        copied._rows_start_end = True, start, end
        return copied

    def rangeBetween(self, start: int, end: int) -> 'WindowSpec':
        copied = copy(self)
        copied._rows_start_end = False, start, end
        return copied
