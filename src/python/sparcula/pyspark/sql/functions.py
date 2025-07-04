from typing import Callable, Optional, TYPE_CHECKING, Union

from .._internal import _functions as F
from .._internal._column import ColBy, Explode, Lit, WhenBasic
from .._internal._types import data_type_from_string
from .column import Column
from .window import _flatten

if TYPE_CHECKING:
    from ._typing import ColumnOrName, ColumnOrValue, LiteralType
    from .dataframe import DataFrame
    from . import types as T


def broadcast(df: 'DataFrame') -> 'DataFrame':
    # from .session import SparkSession
    # spark = SparkSession.builder.getOrCreate()
    # rows = df.collect()
    # df = spark.createDataFrame(rows, df.schema)
    return df


def col(col: str) -> 'Column':
    return Column(ColBy(col))


def lit(col: 'LiteralType') -> 'Column':
    if isinstance(col, Column):
        return col
    return Column(Lit.of(col))


def when(condition: 'Column', value: 'ColumnOrValue') -> 'Column':
    return Column(WhenBasic(condition._column, Column._get_column_from_value(value)))


def struct(*cols: 'ColumnOrName') -> 'Column':
    return Column(F.Struct([Column._get_column_from_name(c) for c in cols]))


def array(*cols: 'ColumnOrName') -> 'Column':
    return Column(F.Array([Column._get_column_from_name(c) for c in cols]))


def explode(col: 'ColumnOrName') -> 'Column':
    return Column(Explode(Column._get_column_from_name(col), False))


def explode_outer(col: 'ColumnOrName') -> 'Column':
    return Column(Explode(Column._get_column_from_name(col), True))


def isnull(col: 'ColumnOrName') -> 'Column':
    return Column(F.Null(True, Column._get_column_from_name(col)))


def isnotnull(col: 'ColumnOrName') -> 'Column':
    return Column(F.Null(False, Column._get_column_from_name(col)))


def regexp_replace(
    string: 'ColumnOrName', pattern: Union[str, 'Column'], replacement: Union[str, 'Column']
) -> 'Column':
    return Column(F.RegexpReplace(
        Column._get_column_from_name(string),
        Column._get_column_from_value(pattern),
        Column._get_column_from_value(replacement),
    ))


def regexp_extract(str: 'ColumnOrName', pattern: str, idx: int) -> 'Column':
    return Column(F.RegexpExtract(Column._get_column_from_name(str), pattern, idx))


def split(str: 'ColumnOrName', pattern: str, limit: int = -1) -> 'Column':
    return Column(F.Split(Column._get_column_from_name(str), pattern, limit))


def split_part(src: 'ColumnOrName', delimiter: 'ColumnOrName', partNum: 'ColumnOrName') -> 'Column':
    return Column(F.SplitPart(
        Column._get_column_from_name(src),
        Column._get_column_from_name(delimiter),
        Column._get_column_from_name(partNum),
    ))


def element_at(col: 'ColumnOrName', extraction: Union[int, Column]) -> 'Column':
    return Column(F.ElementAt(
        Column._get_column_from_name(col),
        Column._get_column_from_value(extraction),
    ))


def concat(*cols: 'ColumnOrName') -> 'Column':
    return Column(F.Concat([Column._get_column_from_name(c) for c in cols]))


def concat_ws(sep: str, *cols: 'ColumnOrName') -> 'Column':
    return Column(F.ConcatWs(sep, [Column._get_column_from_name(c) for c in cols]))


def _date_add_sub(
    name: str, sign: str, start: 'ColumnOrName', days: Union['ColumnOrName', int]
) -> 'Column':
    return Column(F.DateAddSub(
        name,
        sign,
        Column._get_column_from_name(start),
        Column._get_column_from_name(lit(days) if isinstance(days, int) else days),
    ))


def date_add(start: 'ColumnOrName', days: Union['ColumnOrName', int]) -> 'Column':
    return _date_add_sub('date_add', '+', start, days)


def date_sub(start: 'ColumnOrName', days: Union['ColumnOrName', int]) -> 'Column':
    return _date_add_sub('date_sub', '-', start, days)


def dayofweek(col: 'ColumnOrName') -> 'Column':
    return Column(F.DayOfWeek(Column._get_column_from_name(col)))


def date_format(date: 'ColumnOrName', format: str) -> 'Column':
    return Column(F.DateFormat(Column._get_column_from_name(date), format))


def to_date(col: 'ColumnOrName', format: Optional[str] = None) -> 'Column':
    return Column(F.ToDate(Column._get_column_from_name(col), format))


def to_timestamp(col: 'ColumnOrName', format: Optional[str] = None) -> 'Column':
    return Column(F.ToTimestamp(Column._get_column_from_name(col), format))


def lpad(col: 'ColumnOrName', length: int, padding: str) -> 'Column':
    assert length > 0
    assert len(padding) == 1
    return Column(F.LPad(Column._get_column_from_name(col), length, padding))


def length(col: 'ColumnOrName') -> 'Column':
    return Column(F.Length(Column._get_column_from_name(col)))


def size(col: 'ColumnOrName') -> 'Column':
    return Column(F.Size(Column._get_column_from_name(col)))


def array_contains(col: 'ColumnOrName', value: 'ColumnOrValue') -> 'Column':
    return Column(F.ArrayContains(
        Column._get_column_from_name(col), Column._get_column_from_value(value),
    ))


def array_except(col1: 'ColumnOrName', col2: 'ColumnOrName') -> 'Column':
    return Column(F.ArrayExcept(
        Column._get_column_from_name(col1), Column._get_column_from_name(col2),
    ))


def array_sort(col: 'ColumnOrName') -> 'Column':
    return Column(F.ArraySort(Column._get_column_from_name(col)))


def arrays_zip(*cols: 'ColumnOrName') -> 'Column':
    return Column(F.ArraysZip([Column._get_column_from_name(c) for c in cols]))


def aggregate(
    col: 'ColumnOrName',
    initialValue: 'ColumnOrName',
    merge: Callable[['Column', 'Column'], 'Column'],
    finish: Callable[['Column'], 'Column'] = lambda acc: acc,
) -> 'Column':
    var = Column(ColBy(F.Aggregate.VAR))
    acc = Column(ColBy(F.Aggregate.ACC))
    x = Column(ColBy(F.Aggregate.X))
    return Column(F.Aggregate(
        Column._get_column_from_name(col),
        Column._get_column_from_name(initialValue),
        merge(acc, x)._column,
        finish(acc)._column,
        merge(var, var)._column.name,
        finish(var)._column.name,
    ))


def create_map(*cols: Union['ColumnOrName', list['ColumnOrName']]) -> 'Column':
    return Column(F.CreateMap([Column._get_column_from_name(c) for c in _flatten(*cols)]))


def map_keys(col: 'ColumnOrName') -> 'Column':
    return Column(F.MapKeys(Column._get_column_from_name(col)))


def map_values(col: 'ColumnOrName') -> 'Column':
    return Column(F.MapValues(Column._get_column_from_name(col)))


def map_concat(*cols: 'ColumnOrName') -> 'Column':
    return Column(F.MapConcat([Column._get_column_from_name(c) for c in cols]))


def from_json(col: 'ColumnOrName', schema: Union['T.DataType', str]) -> 'Column':
    data_type = data_type_from_string(schema) if isinstance(schema, str) else schema
    return Column(F.FromJson(Column._get_column_from_name(col), data_type))


def trim(col: 'ColumnOrName') -> 'Column':
    return Column(F.Trim(Column._get_column_from_name(col)))


def upper(col: 'ColumnOrName') -> 'Column':
    return Column(F.UpperLower('upper', Column._get_column_from_name(col)))


def lower(col: 'ColumnOrName') -> 'Column':
    return Column(F.UpperLower('lower', Column._get_column_from_name(col)))


def substring(str: 'ColumnOrName', pos: int, len: int) -> 'Column':
    return Column(F.Substring(Column._get_column_from_name(str), pos, len))


def coalesce(*cols: 'ColumnOrName') -> 'Column':
    return Column(F.Coalesce([Column._get_column_from_name(c) for c in cols]))


def abs(col: 'ColumnOrName') -> 'Column':
    return Column(F.Abs(Column._get_column_from_name(col)))


def round(col: 'ColumnOrName', scale: int = 0) -> 'Column':
    return Column(F.Round(Column._get_column_from_name(col), scale))


# aggregations

def row_number() -> 'Column':
    return Column(F.RowNumber())


def rank() -> 'Column':
    return Column(F.Rank())


def dense_rank() -> 'Column':
    return Column(F.DenseRank())


def count(col: 'ColumnOrName') -> 'Column':
    if isinstance(col, str) and col == '*':
        return Column(F.Count(None))
    return Column(F.Count(Column._get_column_from_name(col)))


def approx_count_distinct(col: 'ColumnOrName') -> 'Column':
    return Column(F.ApproxCountDistinct(Column._get_column_from_name(col)))


def first(col: 'ColumnOrName', ignorenulls: bool = False) -> 'Column':
    return Column(F.FirstLast('first', Column._get_column_from_name(col), ignorenulls))


def last(col: 'ColumnOrName', ignorenulls: bool = False) -> 'Column':
    return Column(F.FirstLast('last', Column._get_column_from_name(col), ignorenulls))


def min(col: 'ColumnOrName') -> 'Column':
    return Column(F.MinMax('min', Column._get_column_from_name(col)))


def max(col: 'ColumnOrName') -> 'Column':
    return Column(F.MinMax('max', Column._get_column_from_name(col)))


def min_by(col: 'ColumnOrName', ord: 'ColumnOrName') -> 'Column':
    return Column(F.MinMaxBy(
        'min_by', Column._get_column_from_name(col), Column._get_column_from_name(ord)
    ))


def max_by(col: 'ColumnOrName', ord: 'ColumnOrName') -> 'Column':
    return Column(F.MinMaxBy(
        'max_by', Column._get_column_from_name(col), Column._get_column_from_name(ord)
    ))


def sum(col: 'ColumnOrName') -> 'Column':
    return Column(F.Sum(Column._get_column_from_name(col)))


def collect_list(col: 'ColumnOrName') -> 'Column':
    return Column(F.Collect('list', Column._get_column_from_name(col)))


def collect_set(col: 'ColumnOrName') -> 'Column':
    return Column(F.Collect('set', Column._get_column_from_name(col)))


def lag(col: 'ColumnOrName', offset: int = 1, default: Optional['LiteralType'] = None) -> 'Column':
    return Column(F.LagLead(
        'lag', Column._get_column_from_name(col), offset, Column._get_column_from_value(default)
    ))


def lead(col: 'ColumnOrName', offset: int = 1, default: Optional['LiteralType'] = None) -> 'Column':
    return Column(F.LagLead(
        'lead', Column._get_column_from_name(col), offset, Column._get_column_from_value(default)
    ))

# aggregations end


def asc(col: 'ColumnOrName') -> 'Column':
    return Column(Column._get_column_from_name(col)).asc()


def desc(col: 'ColumnOrName') -> 'Column':
    return Column(Column._get_column_from_name(col)).desc()


def current_timestamp() -> 'Column':
    return Column(F.CurrentTimestamp())


def monotonically_increasing_id() -> 'Column':
    return Column(F.MonotonicallyIncreasingId())


def hash(*cols: 'ColumnOrName') -> 'Column':
    return Column(F.Hash([Column._get_column_from_name(c) for c in cols]))


def udf(f=None, returnType=None):
    pass
