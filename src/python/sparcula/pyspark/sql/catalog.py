from typing import NamedTuple, Optional


class Catalog:
    def clearCache(self) -> None:
        pass


class Table(NamedTuple):
    name: str
    catalog: Optional[str]
    namespace: Optional[list[str]]
    description: Optional[str]
    tableType: str
    isTemporary: bool

    @property
    def database(self) -> Optional[str]:
        if self.namespace is not None and len(self.namespace) == 1:
            return self.namespace[0]
        else:
            return None
