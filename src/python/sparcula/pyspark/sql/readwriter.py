from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union

from . import types as T
from .._internal._dataframe import (
    CodeGenerator, CsvDataFrame, ParquetDataFrame, TableDataFrame
)
from .._internal._types import data_type_from_string

if TYPE_CHECKING:
    from logging import Logger

    from duckdb import DuckDBPyRelation

    from .._internal._dataframe import BaseDataFrame
    from .._internal._session import Session
    from .dataframe import DataFrame


class DataFrameReader:
    def __init__(self, session: 'Session'):
        self._session = session
        self._schema: Optional['T.StructType'] = None

    def _df(self, df: 'BaseDataFrame') -> 'DataFrame':
        from .dataframe import DataFrame

        return DataFrame(df)

    def schema(self, schema: str) -> 'DataFrameReader':
        data_type = data_type_from_string(schema)
        assert isinstance(data_type, T.StructType)
        self._schema = data_type
        return self

    def csv(
        self,
        path: Union[str, 'Path'],
        schema: Optional[Union['T.StructType', str]] = None,
        escape: str = '"',
        header: bool = False,
    ) -> 'DataFrame':
        assert escape == '"'
        if isinstance(schema, str):
            data_type = data_type_from_string(schema)
            assert isinstance(data_type, T.StructType)
            schema = data_type
        return self._df(CsvDataFrame(self._session, str(path), schema, header))

    def parquet(self, path: str) -> 'DataFrame':
        assert self._schema
        if Path(path.removeprefix('file:')).is_dir():
            path += '/*.parquet'
        return self._df(ParquetDataFrame(self._session, path, self._schema))

    def table(self, tableName: str) -> 'DataFrame':
        return self._df(TableDataFrame(self._session, tableName))


class DataFrameWriter:
    def __init__(self, df: 'BaseDataFrame'):
        self._df = df
        self._table: Optional[str] = None

    def partitionBy(self, *cols: Union[str, list[str]]) -> 'DataFrameWriter':
        return self

    def format(self, source: str) -> 'DataFrameWriter':
        return self

    def mode(self, saveMode: str) -> 'DataFrameWriter':
        return self

    def option(self, key: str, value: str) -> 'DataFrameWriter':
        return self

    def saveAsTable(self, name: str) -> None:
        raise NotImplementedError()

    def csv(self, path: str) -> None:
        pass

    def parquet(self, path: str, compression: Optional[str] = None) -> None:
        def terminate(relation, _p, _t):
            relation.write_parquet(path, compression=compression, per_thread_output=True)

        _execute(self._df, terminate)


def _log_call_stack(logger: 'Logger', call_stack_depth: int) -> None:
    if call_stack_depth:
        import inspect

        lines = ['Terminal operation called at:']
        for f in inspect.stack()[1:1 + call_stack_depth]:
            lines.append(f'{f.filename}:{f.lineno} - {f.function}')

        logger.info('\n'.join(lines))


_T = TypeVar('_T')


def _execute(
    df: 'BaseDataFrame',
    terminate: Callable[['DuckDBPyRelation', 'Logger', datetime], _T],
    purpose: str = 'collect',
) -> _T:
    t0 = datetime.now()
    session = df.session
    logger = session.logger
    logger.info(f'JOB START')

    _log_call_stack(logger, session.call_stack_depth)

    from .._internal._dataframe import total_calls, max_depth
    logger.info(f'distance function - total_calls: {total_calls}, max_depth: {max_depth}')

    code, params = CodeGenerator(df, purpose).generate()
    sql = code.to_string()

    _open_params(params, session)
    try:
        relation = session.connection.sql(sql)

        t1 = datetime.now()
        logger.info(f'Relation created in {t1 - t0}')

        result = terminate(relation, logger, t1)
    except Exception as e:
        a = 1  # for debugging
        raise e
    finally:
        _close_params(params, session)

    t2 = datetime.now()
    logger.info(f'JOB END - duration: {t2 - t0}')
    return result


def _open_params(params: list, session: 'Session') -> None:
    table_info_by_name = {}
    for p in params:
        if len(p) == 2:
            session.connection.register(*p)
        elif len(p) == 3:
            table_info_by_name[p[0]] = p[1:]
        else:
            raise RuntimeError(p)
    if table_info_by_name:
        import requests
        from databricks.sdk import WorkspaceClient
        client = WorkspaceClient()
        region = client.metastores.summary().region
        entry_point: Any = client.dbutils.notebook.entry_point
        context = entry_point.getDbutils().notebook().getContext()
        api_url = context.apiUrl().get()
        api_token = context.apiToken().get()
        url = f"{api_url}/api/2.0/unity-catalog/temporary-table-credentials"
        headers = {"Authorization": "Bearer " + api_token}
        for table_name, (table_id, storage_location) in table_info_by_name.items():
            body = {"table_id": table_id, "operation": "READ"}
            response = requests.post(url, headers=headers, json=body)
            creds = response.json()['aws_temp_credentials']
            session.connection.sql(f"""CREATE SECRET {table_name.replace('.', '__')} (
                TYPE s3,
                REGION '{region}',
                KEY_ID '{creds['access_key_id']}',
                SECRET '{creds['secret_access_key']}',
                SESSION_TOKEN '{creds['session_token']}',
                SCOPE '{storage_location}'
            )""")


def _close_params(params: list, session: 'Session') -> None:
    table_names = set()
    for p in params:
        if len(p) == 2:
            session.connection.unregister(p[0])
        elif len(p) == 3:
            table_names.add(p[0])
        else:
            raise RuntimeError(p)
    for table_name in table_names:
        session.connection.sql(f"DROP SECRET {table_name.replace('.', '__')}")
