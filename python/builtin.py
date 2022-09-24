from dataclasses import dataclass
import typing

@dataclass
class More:
    first: typing.Any 
    rest: typing.Any
    def __iter__(self):
        while True:
            match self:
                case Done():
                    return 
                case More(first, rest):
                    yield first
                    self = rest()

class Done:
    pass

def range(a, b):
    if a == b:
        return Done()
    else:
        return More(a, lambda: range(a + 1,  b))

def map(stream, f):
    match next(stream):
        case More(first, rest):
            return More(f(first), lambda: map(rest, f))
        case Done():
            return Done()

def filter(stream, f):
    match next(stream):
        case More(first, rest):
            if f(first):
                return More(first, lambda: filter(rest, f))
            else:
                return filter(rest, f)
        case Done():
            return Done()

def reduce(stream, init, f):
    match next(stream):
        case More(first, rest):
            return reduce(rest, f(init, first), f)
        case Done():
            return init

for el in range(0, 1000000):
    print(el)

