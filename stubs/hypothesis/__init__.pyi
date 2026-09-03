from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

def given(
    *args: Any, **kwargs: Any
) -> Callable[[Callable[..., T]], Callable[..., T]]: ...
def settings(
    *,
    max_examples: int | None = None,
    deadline: int | None = None,
    print_blob: bool = False,
    **kwargs: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]: ...

class Strategies:
    @staticmethod
    def text(
        min_size: int = 0,
        max_size: int | None = None,
        alphabet: str | None = None,
    ) -> Any: ...
    @staticmethod
    def integers(
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> Any: ...
    @staticmethod
    def lists(
        elements: Any,
        min_size: int = 0,
        max_size: int | None = None,
    ) -> Any: ...
    @staticmethod
    def floats(
        min_value: float | None = None,
        max_value: float | None = None,
        allow_nan: bool = True,
        allow_infinity: bool = True,
    ) -> Any: ...
    @staticmethod
    def booleans() -> Any: ...
    @staticmethod
    def none() -> Any: ...
    @staticmethod
    def one_of(*args: Any) -> Any: ...
    @staticmethod
    def sampled_from(elements: list[Any]) -> Any: ...


strategies = Strategies
