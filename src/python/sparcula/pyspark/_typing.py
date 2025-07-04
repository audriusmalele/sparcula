from datetime import date, datetime
from typing import Union

ScalarType = Union[None, bool, int, float, str, date, datetime]
LiteralType = Union[ScalarType, list]
CellType = Union[ScalarType, tuple, list]
