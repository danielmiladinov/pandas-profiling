import itertools
from typing import Tuple, TypeVar, Iterable, Callable

T = TypeVar('T')


def partition(ts: Iterable[T], p: Callable[[T], bool] = bool) -> Tuple[Iterable[T], Iterable[T]]:
    """
    Partition an iterable into a tuple of iterables, based on each element's result returned from a predicate function.
    The first element of the returned tuple contains all of the input iterable's elements that passed the predicate,
    and the second element contains all of the input iterable's elements that failed the predicate.
    The default predicate is the bool constructor if none is provided.

    :param ts: Iterable[T]
    :param p: Callable[[T], bool]
    :return: Tuple[Iterable[T], Iterable[T]]
    """
    a, b = itertools.tee((p(t), t) for t in ts)
    return (t for b, t in a if b), (t for b, t in b if not b)
