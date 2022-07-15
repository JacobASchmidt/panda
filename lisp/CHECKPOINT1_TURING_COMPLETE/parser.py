from enum import Enum
from functools import reduce
from string import whitespace
from typing import Dict, Iterable, List, Tuple
import typing
from dataclasses import dataclass



class Identifier(str):
    pass

class NoneT:
    pass

NoneV = NoneT()

class OpenParen(NoneT):
    pass
class CloseParen(NoneT):
    pass 

@dataclass
class Lambda:
    Unbound: List[str]
    Body: typing.Any


Value = NoneT | Identifier | Lambda
AST = List | Value



def betaReduce(l: Lambda, args: List[Value]) -> AST:
    d = {var: value for var, value in zip(l.Unbound, args)}
    
    def impl(ast: AST, bound: Dict[str, Value]) -> AST:
        if bound == {}:
            return ast 
        match ast:
            case vals if isinstance(vals, List):
                return [impl(el, bound) for el in vals]
            case value if isinstance(value, Value):
                match value:
                    case NoneT(_):
                        return NoneV 
                    case Identifier(el):
                        if el in bound:
                            return bound[el]
                        else:
                            return el 
                    case Lambda(Unbound=unbound, Body=body):
                        bound = {key: value for key, value in bound.items() if key not in unbound}
                        body = impl(body, bound)
                        return Lambda(unbound, body)
    return impl(l.Body, d)


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
            case "none":
                return NoneV
            case _:
                return Identifier(word)
    
    first, rest = matchOn(inp)
    match first:
        case Done():
            return
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
    def makeLambda(l: List[AST]) -> Lambda:
        if len(l) != 3:
            raise ValueError(f"expected three parts to lambda statement, got {len(l)=}")
        if not isinstance(l[1], List):
            raise ValueError(f"expected list of names, got {l[1]=}")
    
        def getId(el): 
            match el:
                case Identifier(id):
                    return id
                case _:
                    raise ValueError("Expected all to be ids, got {}", el)
        unbound = [getId(el) for el in l[1]]
        body = l[2]
        return Lambda(Unbound=unbound, Body=body)
    print(ast)
    match ast:
        case v if isinstance(v, Value):
            return v 
        case l if isinstance(l, List):
            if isinstance(l[0], Identifier) and l[0] == "lambda":
                return makeLambda(l)
            
            l = [Eval(el) for el in l]
            match l[0]:
                case Identifier(id):
                    raise ValueError(f"Unknown identifier {id}")
                case lmbda if isinstance(lmbda, Lambda):
                    return Eval(betaReduce(lmbda, l[1:]))
                case _: 
                    raise ValueError("expected lambda function call, got {}", l[0])


s = "((lambda (a b) a) ((lambda (b c) b) none (lambda (a b) b)) (lambda (c d) d))"

print([Eval(el) for el in (Lex(Tokenize(s)))])
