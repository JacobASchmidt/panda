from string import whitespace
from typing import Dict, Iterable, List, Set, Tuple
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
Expression = List | Value


def Substitute(expr: Expression, bound: Dict[str, Value]) -> Expression:
    if bound == {}:
            return expr 
    match expr:
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


def betaReduce(l: Lambda, args: Iterable[Value]) -> Expression:
    d = {var: value for var, value in zip(l.Unbound, args)}
    assert len(l.Unbound) == len(d), f"unexpected number of args, expected {len(l.Unbound)} got {len(d)} in {l} with args={[*args]}"
    return Substitute(l.Body, d)

class LambdaToken:
    pass

class Define:
    pass

Token = OpenParen | CloseParen | Value | LambdaToken | Define

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
            
def expect(inp: Iterable[Token], token: Token) -> Iterable[Token]:
    first, rest = matchOn(inp)
    match first:
        case el if el == first:
            return rest
        case Done():
            raise ValueError(f"unexpexted eof")
        case el:
            raise ValueError(f"unexpected token {el}")
        
def LexExpr(inp: typing.Iterable[Token]) -> Tuple[Expression, Iterable[Token]] | None:
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
                    assert False, f"expected identifier or closeparen, got {first}"
        return impl(inp, [])
    def lambdaBody(inp: Iterable[Token]) -> Tuple[Expression, Iterable[Token]]:
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
                raise ValueError(f"FAIL {rest} in lambdaBody")
    def LexList(inp: Iterable[Token]) -> Tuple[Expression, Iterable[Token]]:
        
        def impl(inp: Iterable[Token], expr: List[Expression]) -> Tuple[Expression, Iterable[Token]]:
            first, rest = matchOn(inp)
            match first:
                case OpenParen():
                    l, rest = LexList(rest)
                    expr.append(l)
                    return impl(rest, expr)
                case CloseParen():
                    return expr, rest
                case LambdaToken():
                    if len(expr) != 0:
                        raise ValueError("lambda found in middle of arguments")
                    rest = expect(rest, OpenParen())
                    args, rest = lambdaArgs(rest)
                    body, rest = lambdaBody(rest)
                    rest = expect(rest, CloseParen())
                    return Lambda(Unbound=args, Body=body), rest
                case val if isinstance(val, Value):
                    expr.append(val)
                    return impl(rest, expr)
                case Done():
                    raise ValueError("unexpected EOF")
                case el:
                    raise ValueError(f"got {el}")
        return impl(inp, [])
    first, rest = matchOn(inp)
    match first:
        case Done():
            return
        case OpenParen():
            return LexList(rest)
        case CloseParen():
            raise ValueError("invalid syntax")
        case val if isinstance(val, Value):
            return val, rest
        case _:
            print(f"{val, type(val)=}")
            raise ValueError("invalid syntax")
    
def Lex(inp: Iterable[Token]) -> Iterable[Tuple[str, Expression]]:
    def lexIdentifier(inp: Iterable[Token]) -> Tuple[str, Iterable[Token]]:
        first, rest = matchOn(inp)
        match first:
            case id if isinstance(id, Identifier):
                return id, rest
            case tok:
                raise ValueError(f"expected identifier, got {tok}")
            
    first, rest = matchOn(inp)
    print(first)
    match first:
        case OpenParen():
            rest = expect(rest, Define())
            name, rest = lexIdentifier(rest)
            body, rest = LexExpr(rest)
            rest = expect(rest, CloseParen())
            yield (name, body)
        case Done():
            return
        case tok:
            raise ValueError(f"unexpected token, expected OpenParen, got {tok}")
    yield from Lex(rest)

def Eval(expr: Expression) -> Value:
    match expr:
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

def Exec(program: Iterable[Tuple[str, Expression]]) -> Dict[str, Value]:
    def undefined(expr: Expression) -> Set[str]:
        def impl(expr: Expression, lambdaBounds: Set[str], globals: Set[str]) -> Tuple[Set[str], Set[str]]:
            match expr:
                case id if isinstance(expr, Identifier):
                    if id not in lambdaBounds:
                        globals.add(id)
                    return lambdaBounds, globals
                case l if isinstance(l, list):
                    for el in l:
                        lambdaBounds, globals = impl(el, lambdaBounds, globals)
                    return lambdaBounds, globals
                case Lambda(Unbound=unbound, Body=body):
                    ub = set(el for el in unbound)
                    toAdd = ub - lambdaBounds

                    lambdaBounds |= toAdd
                    lambdaBounds, globals = impl(body, lambdaBounds, globals)

                    lambdaBounds -= toAdd
                    return lambdaBounds, globals
                case _:
                    return lambdaBounds, globals 

        _, globals = impl(expr, set(), set())
        return globals
    for name, expr in program:
        print(name)
        print(undefined(expr))
        print(expr)
s = """
((lambda (a b c) a) 
    ((lambda (b c) b) 
        none 
        (lambda (a b) b)) 
    (lambda (c d) d) 
    (lambda () (lambda (a) a)))"
"""
s = """
(def pair (lambda (a b) (lambda (f) (f a b)) ))
(def first  (lambda (a b) a))
(def second  (lambda (a b) b))
(def a (first (pair none (pair none none))))
"""

# print([Eval(el) for el in (Lex(Tokenize(s)))])
print(Exec(Lex(Tokenize(s))))
