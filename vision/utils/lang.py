from __future__ import annotations
from time import time
from typing import Callable, Generator, Iterable, TypeVar

def float_rounded (input, ndigits=2):
    return round(float(input), ndigits)

def format_point (x, y, ndigits=2):
    return [round(float(x), ndigits), round(float(y), ndigits)]

def format_points (coords, ndigits=2):
    for coord in coords:
        yield format_point(coord[0], coord[1], ndigits)

T = TypeVar('T')
TGenerator = TypeVar('TGenerator', Generator, object)

class lazy ():
    def __init__(self, fn: Callable[[], T]):
        self.fn = fn
        self.data = None
        self.done = False

    def __call__(self):
        if not self.done:
            self.data = self.fn()
            self.done = True
        return self.data

def lazy_timed (target: dict[str, float], name: str, fn: Callable[[], T]) -> Callable[[], T]:
    def run ():
        start = time()
        res = fn()
        target[name] = time() - start
        return res
    return lazy(run)

def timed (target: dict[str, float], name: str, fn: Callable[[], T]) -> T:
    return lazy_timed(target, name, fn)()

class TimedIter(Iterable):
    def __init__(self, timings: dict[str, float], name: str, iter: Generator):
        self.timings = timings
        self.name = name
        self.iter = iter
        self.total = 0.0

    def __iter__(self):
        return self

    def __next__(self):
        start = time()
        res = next(self.iter)
        self.total += time() - start
        self.timings[self.name] = self.total
        return res

def timed_iter (timings: dict[str, float], name: str, iter: TGenerator) -> TGenerator:
    return TimedIter(timings, name, iter)

def format_timings_str (timings: dict[str, float]) -> str:
    return ' '.join((f'{key}={float_rounded(value)}' for key, value in timings.items()))

def format_timings (timings: dict[str, float]):
    formatted = {}
    for key, value in timings.items():
        # seconds -> milliseconds
        value *= 1000
        # In order to have smaller transferred data we use only as much
        # digits as make sense
        if value > 100:
            value = int(round(value))
        else:
            if value < 1:
                rounding = 3
            elif value < 10:
                rounding = 2
            else:
                rounding = 1
            value = round(value, rounding)
        formatted[key] = value
    return formatted

class to_tuple:
    def __init__(self, iter):
        self.iter = iter

    def __iter__(self):
        return self

    def __next__(self):
        return [next(self.iter), next(self.iter)]

def optional_bool_value(dict: dict[str, any], key: str):
    if key in dict:
        value = dict[key]
        if value == True:
            return True
        if value == 1:
            return True
    return False
