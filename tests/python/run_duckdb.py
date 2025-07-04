import duckdb
from duckdb.value.constant import Value

if __name__ == '__main__':
    connection = duckdb.default_connection()

    struct_type = connection.type('struct(b int)')
    map_type = connection.type('map(varchar, int)')
    list_type = connection.type('int[]')

    rows = [
        [1, Value([2], list_type), Value((1,), struct_type), Value({'b': 123}, map_type)],
        [10, Value([20], list_type), Value((30,), struct_type), Value({'a': 246}, map_type)],
    ]
    rel = connection.sql(
        query='SELECT * FROM (VALUES (?, ?, ?, ?), (?, ?, ?, ?)) t(a, b, c, d)',
        params=[v for row in rows for v in row],
    )
    out = rel.fetchall()
    print(out)
    rel = connection.sql(f"select a as c0, array_agg(distinct b) as c1, count(distinct b) as c2 from rel group by all")
    rel = connection.sql(f'select list_reduce(list_prepend(0, []), (acc, x) -> acc + x)')
    rel = connection.sql(f"select p, o, any_value(v) over (partition by p order by o ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as v from (values ('a', 1, NULL), ('a', 2, 2)) t(p, o, v)")
    rel = connection.sql("select array(select unnest([1, 2, 3, 4]) except select unnest([2, 3, 5]))")
    rel = connection.sql("select array(select * from (values ({'a':1, 'b':3}), ({'a':2, 'b':4})))")
    print(type(rel), rel.columns, rel.types)
    print(rel)
    print(rel.fetchall())
    print(rel.sql_query())
