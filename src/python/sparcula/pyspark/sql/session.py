import json
import logging
import sys
from typing import Optional, TYPE_CHECKING, Union

import duckdb

from .._internal._dataframe import CreateDataFrame
from .._internal._session import Session
from ..context import SparkContext
from . import types as T
from .catalog import Catalog
from .conf import RuntimeConfig
from .dataframe import DataFrame
from .readwriter import DataFrameReader

if TYPE_CHECKING:
    from .._typing import CellType


import atexit
atexit.register(lambda: print(duckdb.sql("SHOW TABLES")))


class SparkSession:
    class Builder:
        def __init__(self, options: dict[str, str]):
            self._options: dict[str, str] = options

        def master(self, master: str) -> 'SparkSession.Builder':
            return self.config('spark.master', master)

        def appName(self, name: str) -> 'SparkSession.Builder':
            return self.config('spark.app.name', name)

        def enableHiveSupport(self) -> 'SparkSession.Builder':
            return self.config('spark.sql.catalogImplementation', 'hive')

        def config(self, key: str, value: Union[str, int, bool]) -> 'SparkSession.Builder':
            value_as_string = value if isinstance(value, str) else json.dumps(value)
            return SparkSession.Builder(self._options | {key: value_as_string})

        def getOrCreate(self) -> 'SparkSession':
            global _SESSION
            if _SESSION is None:
                _SESSION = SparkSession(self._options)
            else:
                _SESSION.conf._options.update(self._options)
            return _SESSION

    builder: 'Builder' = Builder({})

    def __init__(self, options: dict[str, str]):
        self.sparkContext = SparkContext()
        self.catalog = Catalog()
        self.conf = RuntimeConfig(options)

        logger = logging.getLogger('sparcula')
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        self._logger = logger
        self._connection = duckdb.default_connection()

    def _get_session(self) -> 'Session':
        options = self.conf._options
        return Session(
            logger=self._logger,
            connection=self._connection,
            call_stack_depth=int(options.get('call_stack_depth', '0')),
            cache_path=options.get('cache_path'),
        )

    @classmethod
    def getActiveSession(cls) -> Optional['SparkSession']:
        global _SESSION
        return _SESSION

    def table(self, tableName: str) -> 'DataFrame':
        return self.read.table(tableName)

    @property
    def read(self) -> 'DataFrameReader':
        return DataFrameReader(self._get_session())

    def createDataFrame(
        self,
        data: list[Union[list['CellType'], dict[str, 'CellType']]],
        schema: Optional[Union['T.StructType', list[str], str]] = None,
    ) -> 'DataFrame':
        return DataFrame(CreateDataFrame(self._get_session(), data, schema))

    def sql(self, sqlQuery: str) -> None:
        pass


_SESSION: Optional['SparkSession'] = None
