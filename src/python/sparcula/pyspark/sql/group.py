from typing import TYPE_CHECKING

from .._internal._dataframe import GroupDataFrame
from .dataframe import DataFrame

if TYPE_CHECKING:
    from .._internal._column import BaseColumn
    from .._internal._dataframe import BaseDataFrame
    from .column import Column


class GroupedData:
    def __init__(self, base_df: 'BaseDataFrame', key_columns: list['BaseColumn']):
        self._base_df = base_df
        self._key_columns = key_columns

    def agg(self, *columns: 'Column') -> 'DataFrame':
        return DataFrame(GroupDataFrame(self._base_df, self._key_columns, [c._column for c in columns]))
