import json
from typing import Optional, Union


class RuntimeConfig:
    def __init__(self, options: dict[str, str]):
        self._options = options

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._options.get(key, default)

    def set(self, key: str, value: Union[str, int, bool]) -> None:
        self._options[key] = value if isinstance(value, str) else json.dumps(value)
