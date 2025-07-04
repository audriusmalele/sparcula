from typing import Generic, TypeVar


T_co = TypeVar('T_co', covariant=True)


class RDD(Generic[T_co]):
    pass
