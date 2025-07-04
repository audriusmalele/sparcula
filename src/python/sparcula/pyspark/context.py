from typing import Optional


class SparkContext:
    def __init__(self) -> None:
        self._properties: dict[str, str] = {}

    def setLocalProperty(self, key: str, value: str) -> None:
        self._properties[key] = value

    def getLocalProperty(self, key: str) -> Optional[str]:
        return self._properties.get(key)
