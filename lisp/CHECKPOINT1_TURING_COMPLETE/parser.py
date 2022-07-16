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


def Substitute(ast: AST, bound: Dict[str, Value]) -> AST:
    if bound == {}:
            return ast 
    match ast:
        case vals if isinstance(vals, List):
            return [Substitute(el, bound) for el in vals]
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
                    body = Substitute(body, bound)
                    return Lambda(unbound, body)


def betaReduce(l: Lambda, args: Iterable[Value]) -> AST:
    d = {var: value for var, value in zip(l.Unbound, args)}
    return Substitute(l.Body, d)

class LambdaToken:
    pass

Token = OpenParen | CloseParen | Value | Lambda

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
            case "lambda":
                return LambdaToken()
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
    def expect(inp: Iterable[Token], token: Token) -> Iterable[Token]:
        first, rest = matchOn(inp)
        match first:
            case el if el == first:
                return rest
            case Done():
                raise ValueError(f"unexpexted eof")
            case el:
                raise ValueError(f"unexpected token {el}")
            
    def lambdaArgs(inp: Iterable[Token]) -> Tuple[List[str], Iterable[Token]]:
        def impl(inp: Iterable[Token], out: List[str]) -> Tuple[List[str], Iterable[Token]]:
            first, rest = matchOn(inp)
            match first:
                case CloseParen():
                    return (out, rest)
                case id if isinstance(id, Identifier):
                    if id in out:
                        raise ValueError("multiple arguments of same name in lambda")
                    out.append(id)
                    return impl(rest, out)
                case Done():
                    raise ValueError(f"unexpected eof")
                case el:
                    raise ValueError(f"expected identifier or closeparen, got {first}")
        return impl(inp, [])
    def lambdaBody(inp: Iterable[Token]) -> Tuple[AST, Iterable[Token]]:
        first, rest = matchOn(inp)
        match first:
            case OpenParen():
                return LexList(inp)
            case CloseParen():
                raise ValueError("unexpected close paren in lambda body")
            case val if isinstance(val, Value):
                return val, rest
            case LambdaToken():
                raise ValueError("found 'lambda' in middle of list")
            case Done():
                raise ValueError("unexpecetd eof")
            case rest:
                raise f"FAIL {rest} in lambdaBody"
    def LexList(inp: Iterable[Token]) -> Tuple[AST, Iterable[Token]]:
        
        def impl(inp: Iterable[Token], ast: List[AST]) -> Tuple[AST, Iterable[Token]]:
            first, rest = matchOn(inp)
            match first:
                case OpenParen():
                    l, rest = LexList(rest)
                    ast.append(l)
                    return impl(rest, ast)
                case CloseParen():
                    return ast, rest
                case LambdaToken():
                    if len(ast) != 0:
                        raise ValueError("lambda found in middle of arguments")
                    rest = expect(rest, OpenParen())
                    args, rest = lambdaArgs(rest)
                    body, rest = lambdaBody(rest)
                    rest = expect(rest, CloseParen())
                    return Lambda(Unbound=args, Body=body), rest
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
    match ast:
        case v if isinstance(v, Value):
            return v 
        case l if isinstance(l, List):
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
