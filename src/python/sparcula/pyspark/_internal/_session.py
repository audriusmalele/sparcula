from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from logging import Logger

    from duckdb import DuckDBPyConnection


class Session:
    def __init__(
        self,
        logger: 'Logger',
        connection: 'DuckDBPyConnection',
        call_stack_depth: int,
        cache_path: Optional[str],
    ):
        self.logger = logger
        self.connection = connection
        self.call_stack_depth = call_stack_depth
        self.cache_path = cache_path
