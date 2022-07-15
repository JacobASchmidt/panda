from enum import Enum
from functools import reduce
from string import whitespace
from typing import Dict, Iterable, List, Tuple
import typing

class Name(str):
    pass
class String(str):
    pass 
class Int(int):
    pass 
class Bool(Enum):
    true = 0
    false = 1
class NoneT():
    pass

NoneV = NoneT()
class OpenParen(NoneT):
    pass
class CloseParen(NoneT):
    pass 


Value = Name | String | Int | Bool | NoneT 
AST = List | Value
Token = OpenParen | CloseParen | Value

class Done():
    pass

def matchOn(inp: Iterable):
    try:
        inp = iter(inp)
        val = next(inp)
        return val, inp
    except:
        return Done(), None

def cons(a, b):
    yield a 
    yield from b

def Tokenize(inp: Iterable[str]) -> Iterable[Token]:
    def tokenizeAlphanum(word: str) -> Token:
        match word:
            case "true":
                return Bool.true
            case "false":
                return Bool.false
            case "none":
                return NoneV
            case n if all(el in "0123456789" for el in n):  
                return Int(int(n))
            case _:
                return Name(word)
    def parseString(inp: Iterable[str]) -> typing.Tuple[str, Iterable[str]]:
        def escape(inp: Iterable[str]) -> typing.Tuple[str, Iterable[str]]:
            first, rest = matchOn(inp)
            match first:
                case 'n':
                    return "\n", rest 
                case 'r':
                    return "\r", rest 
                case 't':
                    return "\t", rest 
                case _:
                    return first, rest 
        def impl(inp: Iterable[str], out: str) -> typing.Tuple[str, Iterable[str]]:
            first, rest = matchOn(inp)
            match first:
                case '\\':
                    r, rest = escape(rest)
                    return impl(rest, out + r)
                case '"':
                    return out, rest 
                case letter:
                    return impl(rest, out + letter)
        return impl(inp, "")
    first, rest = matchOn(inp)
    match first:
        case Done():
            return
        case '"':
            string, rest = parseString(rest)
            yield String(string) 
        case '(':
            yield OpenParen()
        case ')':
            yield CloseParen()
        case el if el in whitespace:
            pass
        case letter:
            s = letter
            while True:
                first, rest = matchOn(rest)
                if first == Done():
                    raise ValueError("bad input")
                if first not in whitespace + ")":
                    s += first 
                else:
                    break 
            yield tokenizeAlphanum(s)
            rest = cons(first, rest)
    yield from Tokenize(rest)
            

def Lex(inp: typing.Iterable[Token]) -> List[AST]:
    def LexList(inp: Iterable[Token]) -> Tuple[AST, Token]:
        def impl(inp: Iterable[Token], ast: List[AST]) -> Tuple[AST, Iterable[Token]]:
            first, rest = matchOn(inp)
            match first:
                case OpenParen():
                    l, rest = LexList(rest)
                    ast.append(l)
                    return impl(rest, ast)
                case CloseParen():
                    return ast, rest
                case val if isinstance(val, Value):
                    ast.append(val)
                    return impl(rest, ast)
        return impl(inp, [])
    first, rest = matchOn(inp)
    match first:
        case Done():
            return
        case OpenParen():
            first, rest = LexList(rest)
            yield first 
        case CloseParen():
            raise ValueError("invalid syntax")
        case val if isinstance(val, Value):
            yield val
        case _:
            print(f"{val, type(val)=}")
            raise ValueError("invalid syntax")
    yield from Lex(rest)
    
def Eval(ast: AST) -> Value:
    def add(l: List[AST]) -> Value:
        val = Eval(l[1])
        if isinstance(val, Int):
            l = l[2:]
            assert all(isinstance(el, Int) for el in l)
            l = map(Eval, l)
            return reduce(lambda a, b: a + b, l, val)
        if isinstance(val, String):
            l = l[2:]
            assert all(isinstance(el, String) for el in l)
            l = map(Eval, l)
            return reduce(lambda a, b: a + b, l, val)
        raise ValueError("bad input")
    def mul(l: List[AST]) -> Value:
        val = Eval(l[1])
        if isinstance(val, Int):
            l = l[2:]
            assert all(isinstance(el, Int) for el in l)
            l = map(Eval, l)
            return reduce(lambda a, b: a * b, l, val)
        raise ValueError("bad input")
            
    def impl(ast: AST, defines: Dict[str, Value]) -> Value:
        def call(l: List[AST], defines: Dict[str, Value]) -> Value:
            match l[0]:
                case "+":
                    return add(l)
                case _:
                    raise ValueError("unsupported")
        match ast:
            case value if isinstance(value, Value):
                return value 
            case l if isinstance(l, List):
                assert l != []
                return call(l, defines)
    return impl(ast, {})

def printAll(l):
    print('[', end="")
    for el in l:
        if isinstance(el, Name):
            print(f"Name({el})", end="")
        elif isinstance(el, String):
            print(f"String({el})", end="")
        elif isinstance(el, Int):
            print(f"Int({el})", end="")
        elif isinstance(el, Iterable):
            printAll(el)
        else:
            print(el, end="")
    print(']')

printAll(Lex(Tokenize("(is true true) (+ 1 2) (- 2 1) (+ \"wowwww\" \" cool\") (define x 124)")))

print([Eval(el) for el in Lex(Tokenize("(+ 1 2) (+ \"wowwww\" \"cool\\\"\")"))])