from typing import Union

from .._typing import LiteralType
from .column import Column

ColumnOrName = Union[Column, str]
ColumnOrValue = Union[Column, LiteralType]
