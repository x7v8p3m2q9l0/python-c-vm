import ast
from enum import IntEnum
import py_compile
import dis
import sys
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import secrets
import marshal
import types
import random
from dataclasses import dataclass

sys.setrecursionlimit(50_000)


class MainBreaker:
    def __init__(self):
        self.counter = 0
        self.seed = secrets.randbits(32)
    
    def _var(self, prefix: str = "v") -> str:
        prefix_hash = hash(f"{prefix}{self.seed}{self.counter}") & 0xFFFF
        self.counter += 1
        return f"_{hex(prefix_hash)[2:]}_{self.counter}"

    
    def ld_glob_err(self, count: int = 200) -> List[ast.stmt]:
        stmts = []
        
        for i in range(count):

            stmts.append(ast.Assign(
                targets=[ast.Name(id=self._var('abs'), ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='abs', ctx=ast.Load()),
                    args=[ast.Constant(value=i * 42)],
                    keywords=[]
                )
            ))
            

            stmts.append(ast.Assign(
                targets=[ast.Name(id=self._var('int'), ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='int', ctx=ast.Load()),
                    args=[ast.Constant(value=3.14 * i)],
                    keywords=[]
                )
            ))
            

            stmts.append(ast.Assign(
                targets=[ast.Name(id=self._var('float'), ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='float', ctx=ast.Load()),
                    args=[ast.Constant(value=i)],
                    keywords=[]
                )
            ))
        
        return stmts
    
    def call_chain_err(self, length: int = 100) -> ast.stmt:
        expr = ast.Constant(value=1)
        
        for i in range(length):
            func = ['abs', 'int', 'float'][i % 3]
            expr = ast.Call(
                func=ast.Name(id=func, ctx=ast.Load()),
                args=[expr],
                keywords=[]
            )
        
        return ast.Assign(
            targets=[ast.Name(id=self._var('chain'), ctx=ast.Store())],
            value=expr
        )
    



    
    def try_nest_timeout(self, depth: int = 18, count: int = 15) -> List[ast.stmt]:
        stmts = []
        
        for _ in range(count):
            body = [ast.Assign(
                targets=[ast.Name(id=self._var(), ctx=ast.Store())],
                value=ast.Constant(value=42)
            )]
            

            for i in range(depth):
                body = [ast.Try(
                    body=[
                        ast.Assign(
                            targets=[ast.Name(id=self._var(f"t{i}"), ctx=ast.Store())],
                            value=ast.BinOp(
                                left=ast.BinOp(
                                    left=ast.Constant(value=i),
                                    op=ast.Mult(),
                                    right=ast.Constant(value=2)
                                ),
                                op=ast.Add(),
                                right=ast.Constant(value=1)
                            )
                        )
                    ] + body,
                    handlers=[ast.ExceptHandler(
                        type=ast.Name(id='Exception', ctx=ast.Load()),
                        name=None,
                        body=[ast.Assign(
                            targets=[ast.Name(id=self._var(f"e{i}"), ctx=ast.Store())],
                            value=ast.Constant(value=0)
                        )]
                    )],
                    orelse=[],
                    finalbody=[]
                )]
            
            stmts.extend(body)
        
        return stmts
    
    def bool_chain_timeout(self, count: int = 100) -> List[ast.stmt]:
        stmts = []
        

        var_names = []
        for i in range(count):
            var_name = self._var('b')
            var_names.append(var_name)
            stmts.append(ast.Assign(
                targets=[ast.Name(id=var_name, ctx=ast.Store())],
                value=ast.Constant(value=i)
            ))
        

        comparisons = []
        for i, var_name in enumerate(var_names):
            comparisons.append(ast.Compare(
                left=ast.BinOp(
                    left=ast.Name(id=var_name, ctx=ast.Load()),
                    op=ast.Add(),
                    right=ast.Constant(value=i)
                ),
                ops=[ast.Lt() if i % 2 else ast.Gt()],
                comparators=[ast.Constant(value=i * 2)]
            ))
        

        result = comparisons[0]
        for comp in comparisons[1:]:
            result = ast.BoolOp(op=ast.And(), values=[result, comp])
        
        stmts.append(ast.Assign(
            targets=[ast.Name(id=self._var('bool_result'), ctx=ast.Store())],
            value=result
        ))
        
        return stmts
    



    
    def comp_nest(self, depth: int = 5) -> ast.stmt:
        comp = ast.ListComp(
            elt=ast.Name(id='x', ctx=ast.Load()),
            generators=[
                ast.comprehension(
                    target=ast.Name(id='x', ctx=ast.Store()),
                    iter=ast.Call(
                        func=ast.Name(id='range', ctx=ast.Load()),
                        args=[ast.Constant(value=3)],
                        keywords=[]
                    ),
                    ifs=[
                        ast.Compare(
                            left=ast.BinOp(
                                left=ast.Name(id='x', ctx=ast.Load()),
                                op=ast.Mod(),
                                right=ast.Constant(value=2)
                            ),
                            ops=[ast.Eq()],
                            comparators=[ast.Constant(value=1)]
                        )
                    ],
                    is_async=0
                )
            ]
        )
        

        for level in range(depth):
            var = f'_lv{level}'
            comp = ast.ListComp(
                elt=ast.Tuple(
                    elts=[ast.Name(id=var, ctx=ast.Load()), comp],
                    ctx=ast.Load()
                ),
                generators=[
                    ast.comprehension(
                        target=ast.Name(id=var, ctx=ast.Store()),
                        iter=ast.Call(
                            func=ast.Name(id='range', ctx=ast.Load()),
                            args=[ast.Constant(value=3)],
                            keywords=[]
                        ),
                        ifs=[
                            ast.Compare(
                                left=ast.BinOp(
                                    left=ast.Name(id=var, ctx=ast.Load()),
                                    op=ast.Mod(),
                                    right=ast.Constant(value=2)
                                ),
                                ops=[ast.NotEq()],
                                comparators=[ast.Constant(value=0)]
                            )
                        ],
                        is_async=0
                    )
                ]
            )
        
        return ast.Assign(
            targets=[ast.Name(id=self._var('comp'), ctx=ast.Store())],
            value=comp
        )
    
    def lambda_nest(self, depth: int = 12) -> ast.stmt:
        expr = ast.Constant(value=42)
        
        for i in range(depth):
            expr = ast.Call(
                func=ast.Lambda(
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg=f'_x{i}')],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[ast.Constant(value=None)]
                    ),
                    body=ast.BinOp(
                        left=expr,
                        op=ast.Add(),
                        right=ast.Constant(value=i)
                    )
                ),
                args=[ast.Constant(value=i)],
                keywords=[]
            )
        
        return ast.Assign(
            targets=[ast.Name(id=self._var('lambda'), ctx=ast.Store())],
            value=expr
        )
    
    def walrus_chain(self, count: int = 50) -> ast.stmt:
        exprs = []
        for i in range(count):
            exprs.append(ast.NamedExpr(
                target=ast.Name(id=self._var(f'w'), ctx=ast.Store()),
                value=ast.BinOp(
                    left=ast.Constant(value=i),
                    op=ast.Mult(),
                    right=ast.Constant(value=2)
                )
            ))
        

        result = exprs[0]
        for expr in exprs[1:]:
            result = ast.BoolOp(op=ast.And(), values=[result, expr])
        
        return ast.Expr(value=result)
    
    def match_cases(self) -> List[ast.stmt]:
        if sys.version_info < (3, 10):
            return []
        
        funcs = []
        
        for idx in range(10):
            func_name = self._var('match')
            
            match_stmt = ast.Match(
                subject=ast.Name(id='val', ctx=ast.Load()),
                cases=[
                    ast.match_case(
                        pattern=ast.MatchSequence(
                            patterns=[
                                ast.MatchAs(name='x', pattern=None),
                                ast.MatchAs(name='y', pattern=None),
                                ast.MatchStar(name='rest')
                            ]
                        ),
                        guard=ast.Compare(
                            left=ast.Name(id='x', ctx=ast.Load()),
                            ops=[ast.Gt()],
                            comparators=[ast.Constant(value=10)]
                        ),
                        body=[ast.Return(value=ast.Tuple(
                            elts=[
                                ast.Name(id='x', ctx=ast.Load()),
                                ast.Name(id='y', ctx=ast.Load()),
                                ast.Name(id='rest', ctx=ast.Load())
                            ],
                            ctx=ast.Load()
                        ))]
                    ),
                    ast.match_case(
                        pattern=ast.MatchMapping(
                            keys=[ast.Constant(value='a'), ast.Constant(value='b')],
                            patterns=[
                                ast.MatchAs(name='a', pattern=None),
                                ast.MatchAs(name='b', pattern=None)
                            ],
                            rest='rest'
                        ),
                        guard=None,
                        body=[ast.Return(value=ast.Dict(
                            keys=[
                                ast.Constant(value='a'),
                                ast.Constant(value='b'),
                                ast.Constant(value='rest')
                            ],
                            values=[
                                ast.Name(id='a', ctx=ast.Load()),
                                ast.Name(id='b', ctx=ast.Load()),
                                ast.Name(id='rest', ctx=ast.Load())
                            ]
                        ))]
                    ),
                    ast.match_case(
                        pattern=ast.MatchOr(
                            patterns=[
                                ast.MatchValue(value=ast.Constant(value=1)),
                                ast.MatchValue(value=ast.Constant(value=2)),
                                ast.MatchValue(value=ast.Constant(value=3))
                            ]
                        ),
                        guard=None,
                        body=[ast.Return(value=ast.Constant(value='number'))]
                    ),
                    ast.match_case(
                        pattern=ast.MatchAs(name=None, pattern=None),
                        guard=None,
                        body=[ast.Return(value=ast.Constant(value=None))]
                    )
                ]
            )
            
            func = ast.FunctionDef(
                name=func_name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg='val')],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[]
                ),
                body=[match_stmt],
                decorator_list=[],
                returns=None
            )
            
            funcs.append(func)
        
        return funcs
    
    def mem_bomb_oom(self, count: int = 100) -> List[ast.stmt]:
        stmts = []
        
        for i in range(count):


            huge_tuple = ast.Tuple(
                elts=[ast.Constant(value=j) for j in range(1000)],
                ctx=ast.Load()
            )
            
            stmts.append(ast.Assign(
                targets=[ast.Name(id=self._var('mem'), ctx=ast.Store())],
                value=huge_tuple
            ))
            

            nested_list = ast.List(
                elts=[
                    ast.List(
                        elts=[ast.Constant(value=k) for k in range(50)],
                        ctx=ast.Load()
                    )
                    for j in range(50)
                ],
                ctx=ast.Load()
            )
            
            stmts.append(ast.Assign(
                targets=[ast.Name(id=self._var('nest'), ctx=ast.Store())],
                value=nested_list
            ))
            

            huge_dict = ast.Dict(
                keys=[ast.Constant(value=f'k{j}') for j in range(200)],
                values=[ast.Constant(value=j * 2) for j in range(200)]
            )
            
            stmts.append(ast.Assign(
                targets=[ast.Name(id=self._var('dict'), ctx=ast.Store())],
                value=huge_dict
            ))
        
        return stmts
    
    def const_flood(self, count: int = 500) -> List[ast.stmt]:
        stmts = []
        
        for i in range(count):

            stmts.append(ast.Assign(
                targets=[ast.Name(id=self._var('s'), ctx=ast.Store())],
                value=ast.Constant(value=f'constant_string_{i}_{secrets.token_hex(8)}')
            ))
            

            stmts.append(ast.Assign(
                targets=[ast.Name(id=self._var('n'), ctx=ast.Store())],
                value=ast.Constant(value=i * 1000000 + 123456789)
            ))
            

            stmts.append(ast.Assign(
                targets=[ast.Name(id=self._var('t'), ctx=ast.Store())],
                value=ast.Tuple(
                    elts=[ast.Constant(value=f'tuple_{i}_{j}') for j in range(5)],
                    ctx=ast.Load()
                )
            ))
        
        return stmts
    
    def decorator_chaos(self, count: int = 20) -> List[ast.FunctionDef]:
        funcs = []
        
        for i in range(count):

            decorators = []
            for j in range(3):

                decorator_name = ['property', 'staticmethod', 'classmethod'][j % 3]
                decorators.append(ast.Name(id=decorator_name, ctx=ast.Load()))
            
            func = ast.FunctionDef(
                name=self._var('dec'),
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg='self')] if i % 2 else [ast.arg(arg='cls')],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[]
                ),
                body=[ast.Return(value=ast.Constant(value=None))],
                decorator_list=decorators,
                returns=None
            )
            
            funcs.append(func)
        
        return funcs
    
    def class_nest_bomb(self, depth: int = 10, count: int = 5) -> List[ast.ClassDef]:
        classes = []
        
        for i in range(count):

            inner = ast.ClassDef(
                name=self._var('Inner'),
                bases=[],
                keywords=[],
                body=[ast.Pass()],
                decorator_list=[]
            )
            

            for level in range(depth):
                inner = ast.ClassDef(
                    name=self._var(f'L{level}'),
                    bases=[],
                    keywords=[],
                    body=[
                        inner,
                        ast.Assign(
                            targets=[ast.Name(id=self._var('attr'), ctx=ast.Store())],
                            value=ast.Constant(value=level)
                        )
                    ],
                    decorator_list=[]
                )
            
            classes.append(inner)
        
        return classes
    
    def async_bomb(self, count: int = 15) -> List[ast.stmt]:
        funcs = []
        
        for i in range(count):

            func_name = self._var('async')
            

            body = [
                ast.Expr(value=ast.Await(
                    value=ast.Call(
                        func=ast.Name(id='__import__', ctx=ast.Load()),
                        args=[ast.Constant(value='asyncio')],
                        keywords=[]
                    )
                ))
            ]
            

            async_for = ast.AsyncFor(
                target=ast.Name(id=self._var('i'), ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id='range', ctx=ast.Load()),
                    args=[ast.Constant(value=3)],
                    keywords=[]
                ),
                body=[
                    ast.Expr(value=ast.Await(
                        value=ast.Call(
                            func=ast.Name(id='__import__', ctx=ast.Load()),
                            args=[ast.Constant(value='asyncio')],
                            keywords=[]
                        )
                    ))
                ],
                orelse=[]
            )
            
            body.append(async_for)
            

            async_with = ast.AsyncWith(
                items=[
                    ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Name(id='__import__', ctx=ast.Load()),
                            args=[ast.Constant(value='asyncio')],
                            keywords=[]
                        ),
                        optional_vars=ast.Name(id=self._var('ctx'), ctx=ast.Store())
                    )
                ],
                body=[ast.Pass()]
            )
            
            body.append(async_with)
            
            func = ast.AsyncFunctionDef(
                name=func_name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[]
                ),
                body=body,
                decorator_list=[],
                returns=None
            )
            
            funcs.append(func)
        
        return funcs
    
    def slice_chaos(self, count: int = 50) -> List[ast.stmt]:
        stmts = []
        
        for i in range(count):

            var = self._var('lst')
            stmts.append(ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.List(
                    elts=[ast.Constant(value=j) for j in range(20)],
                    ctx=ast.Load()
                )
            ))
            

            stmts.append(ast.Assign(
                targets=[ast.Name(id=self._var('sl'), ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name(id=var, ctx=ast.Load()),
                    slice=ast.Slice(
                        lower=ast.Constant(value=i % 5),
                        upper=ast.Constant(value=15 - (i % 5)),
                        step=ast.Constant(value=2)
                    ),
                    ctx=ast.Load()
                )
            ))
            

            stmts.append(ast.Assign(
                targets=[ast.Name(id=self._var('nsl'), ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Subscript(
                        value=ast.Name(id=var, ctx=ast.Load()),
                        slice=ast.Slice(
                            lower=ast.Constant(value=0),
                            upper=ast.Constant(value=10),
                            step=None
                        ),
                        ctx=ast.Load()
                    ),
                    slice=ast.Slice(
                        lower=ast.Constant(value=2),
                        upper=ast.Constant(value=8),
                        step=ast.Constant(value=2)
                    ),
                    ctx=ast.Load()
                )
            ))
        
        return stmts
    
    def attr_chain_bomb(self, count: int = 30) -> List[ast.stmt]:
        stmts = []
        
        for i in range(count):

            var = self._var('obj')
            stmts.append(ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='object', ctx=ast.Load()),
                    args=[],
                    keywords=[]
                )
            ))
            


            expr = ast.Call(
                func=ast.Name(id='hasattr', ctx=ast.Load()),
                args=[
                    ast.Name(id=var, ctx=ast.Load()),
                    ast.Constant(value='__class__')
                ],
                keywords=[]
            )
            

            for j in range(15):
                expr = ast.BoolOp(
                    op=ast.And(),
                    values=[
                        expr,
                        ast.Call(
                            func=ast.Name(id='hasattr', ctx=ast.Load()),
                            args=[
                                ast.Name(id=var, ctx=ast.Load()),
                                ast.Constant(value=f'__attr_{j}__')
                            ],
                            keywords=[]
                        )
                    ]
                )
            
            stmts.append(ast.Assign(
                targets=[ast.Name(id=self._var('attr'), ctx=ast.Store())],
                value=expr
            ))
        
        return stmts
    
    def starred_chaos(self, count: int = 40) -> List[ast.stmt]:
        stmts = []
        
        for i in range(count):

            var = self._var('tup')
            stmts.append(ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.Tuple(
                    elts=[ast.Constant(value=j) for j in range(20)],
                    ctx=ast.Load()
                )
            ))
            

            stmts.append(ast.Assign(
                targets=[ast.Tuple(
                    elts=[
                        ast.Name(id=self._var('a'), ctx=ast.Store()),
                        ast.Starred(
                            value=ast.Name(id=self._var('mid'), ctx=ast.Store()),
                            ctx=ast.Store()
                        ),
                        ast.Name(id=self._var('z'), ctx=ast.Store())
                    ],
                    ctx=ast.Store()
                )],
                value=ast.Name(id=var, ctx=ast.Load())
            ))
            

            stmts.append(ast.Assign(
                targets=[ast.Name(id=self._var('starred'), ctx=ast.Store())],
                value=ast.Tuple(
                    elts=[
                        ast.Starred(
                            value=ast.Name(id=var, ctx=ast.Load()),
                            ctx=ast.Load()
                        )
                    ],
                    ctx=ast.Load()
                )
            ))
        
        return stmts
    
    def yield_bomb(self, count: int = 20) -> List[ast.FunctionDef]:
        funcs = []
        
        for i in range(count):
            body = []
            

            for j in range(10):
                body.append(ast.Expr(value=ast.Yield(
                    value=ast.BinOp(
                        left=ast.Constant(value=j),
                        op=ast.Mult(),
                        right=ast.Constant(value=i)
                    )
                )))
            

            body.append(ast.Expr(value=ast.YieldFrom(
                value=ast.Call(
                    func=ast.Name(id='range', ctx=ast.Load()),
                    args=[ast.Constant(value=5)],
                    keywords=[]
                )
            )))
            

            body.append(ast.Try(
                body=[
                    ast.Expr(value=ast.Yield(value=ast.Constant(value=42)))
                ],
                handlers=[
                    ast.ExceptHandler(
                        type=None,
                        name=None,
                        body=[ast.Expr(value=ast.Yield(value=ast.Constant(value=0)))]
                    )
                ],
                orelse=[],
                finalbody=[]
            ))
            
            func = ast.FunctionDef(
                name=self._var('gen'),
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[]
                ),
                body=body,
                decorator_list=[],
                returns=None
            )
            
            funcs.append(func)
        
        return funcs

PY_VERSION = sys.version_info[:2]
PY38_PLUS = PY_VERSION >= (3, 8)
PY310_PLUS = PY_VERSION >= (3, 10)
PY311_PLUS = PY_VERSION >= (3, 11)


class TransformIntensity(IntEnum):
    NONE = 0
    LIGHT = 1
    MODERATE = 2
    AGGRESSIVE = 3
    EXTREME = 4  
@dataclass
class InstructionInfo:
    offset: int
    opcode: int
    opname: str
    arg: Optional[int]
    argval: any
    is_jump: bool
    jump_target: Optional[int]
    size: int  # Total size including CACHE entries

class BytecodeTransformer:    
    def __init__(self, verbose: bool = False, version: Optional[Tuple[int, int]] = None):
        self.verbose = verbose
        self.py_version = version or PY_VERSION
        
        # Opcode information
        self.nop_opcode = dis.opmap.get('NOP', 9)
        self.extended_arg_opcode = dis.opmap.get('EXTENDED_ARG', 144)
        self.return_value_opcode = dis.opmap.get('RETURN_VALUE', 83)
        
        # Jump opcodes
        self.jump_opcodes = set(dis.hasjrel) | set(dis.hasjabs)
        self.rel_jump_opcodes = set(dis.hasjrel)
        self.abs_jump_opcodes = set(dis.hasjabs)
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'files_succeeded': 0,
            'files_failed': 0,
            'bytes_added': 0,
            'constants_added': 0,
        }
    
    def log(self, message: str, level: str = "INFO"):
        if self.verbose:
            prefix = {
                "INFO": "ℹ️ ",
                "WARN": "⚠️ ",
                "ERROR": "❌",
                "SUCCESS": "✅"
            }.get(level, "")
            print(f"{prefix} {message}")
    
    def read_pyc(self, pyc_path: Path) -> Tuple[bytes, bytes, bytes, types.CodeType]:
        """
        Returns: (magic, bit_field, timestamp, code_object)
        """
        try:
            with open(pyc_path, 'rb') as f:
                # Magic number (4 bytes)
                magic = f.read(4)
                if len(magic) != 4:
                    raise ValueError("Invalid .pyc file: truncated magic number")
                
                # Validate magic number
                if not self._validate_magic(magic):
                    raise ValueError(f"Invalid magic number: {magic.hex()}")
                
                # Bit field (4 bytes) - Python 3.7+
                bit_field = f.read(4)
                if len(bit_field) != 4:
                    raise ValueError("Invalid .pyc file: truncated bit field")
                
                # Timestamp/hash (4 bytes)
                timestamp = f.read(4)
                if len(timestamp) != 4:
                    raise ValueError("Invalid .pyc file: truncated timestamp")
                
                # Size (4 bytes) - Python 3.3+
                size = f.read(4)
                if len(size) != 4:
                    # Try to continue without size
                    size = b'\x00\x00\x00\x00'
                
                # Code object
                code_obj = marshal.load(f)
                
                if not isinstance(code_obj, types.CodeType):
                    raise ValueError("Invalid .pyc file: code object is not CodeType")
                
                self.log(f"Successfully read {pyc_path}", "SUCCESS")
                return magic, bit_field, timestamp, code_obj
                
        except Exception as e:
            self.log(f"Failed to read {pyc_path}: {e}", "ERROR")
            raise
    
    def write_pyc(self, pyc_path: Path, magic: bytes, bit_field: bytes,
                  timestamp: bytes, code_obj: types.CodeType) -> None:
        try:
            # Pre-write validation
            is_valid, message = self.verify_code_object(code_obj)
            if not is_valid:
                raise ValueError(f"Code object validation failed: {message}")
            
            # Write to temporary file first
            temp_path = pyc_path.with_suffix('.pyc.tmp')
            
            with open(temp_path, 'wb') as f:
                f.write(magic)
                f.write(bit_field)
                f.write(timestamp)
                f.write(b'\x00\x00\x00\x00')  # Size (placeholder)
                marshal.dump(code_obj, f)
            
            # Verify the written file can be read
            try:
                _, _, _, test_code = self.read_pyc(temp_path)
                is_valid, message = self.verify_code_object(test_code)
                if not is_valid:
                    raise ValueError(f"Written file validation failed: {message}")
            except Exception as e:
                temp_path.unlink()
                raise ValueError(f"Written file is invalid: {e}")
            
            # Atomic replace
            temp_path.replace(pyc_path)
            
            self.log(f"Successfully wrote {pyc_path}", "SUCCESS")
            
        except Exception as e:
            self.log(f"Failed to write {pyc_path}: {e}", "ERROR")
            raise
    
    def _validate_magic(self, magic: bytes) -> bool:
        if len(magic) != 4:
            return False
        
        # Python 3.x magic numbers typically have patterns
        # We won't be too strict, just check it's not zeros or garbage
        if magic == b'\x00\x00\x00\x00':
            return False
        
        return True
    
    # ========================================================================
    # VALIDATION - COMPREHENSIVE
    # ========================================================================
    
    def verify_code_object(self, code_obj: types.CodeType, 
                          depth: int = 0) -> Tuple[bool, str]:
        if depth > 10:
            return False, "Code object nesting too deep"
        
        try:
            # Check 1: Basic attributes exist
            required_attrs = ['co_code', 'co_consts', 'co_names', 'co_varnames']
            for attr in required_attrs:
                if not hasattr(code_obj, attr):
                    return False, f"Missing attribute: {attr}"
            
            # Check 2: Bytecode is not empty
            if len(code_obj.co_code) == 0:
                return False, "Empty bytecode"
            
            # Check 3: Bytecode length is even (2-byte instructions)
            if len(code_obj.co_code) % 2 != 0:
                return False, "Bytecode length is odd (should be even)"
            
            # Check 4: Can disassemble without errors
            try:
                list(dis.get_instructions(code_obj))
            except Exception as e:
                return False, f"Disassembly failed: {e}"
            
            # Check 5: All jump targets are valid instruction boundaries
            jump_targets = self._get_all_jump_targets(code_obj)
            valid_offsets = self._get_valid_instruction_offsets(code_obj)
            
            invalid_targets = jump_targets - valid_offsets
            if invalid_targets:
                return False, f"Invalid jump targets: {invalid_targets}"
            
            # Check 6: Validate nested code objects
            for const in code_obj.co_consts:
                if isinstance(const, types.CodeType):
                    is_valid, message = self.verify_code_object(const, depth + 1)
                    if not is_valid:
                        return False, f"Nested code object invalid: {message}"
            
            return True, "OK"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def _get_all_jump_targets(self, code_obj: types.CodeType) -> Set[int]:
        targets = set()
        
        for instr in dis.get_instructions(code_obj):
            if instr.opcode in self.jump_opcodes:
                if isinstance(instr.argval, int):
                    targets.add(instr.argval)
        
        return targets
    
    def _get_valid_instruction_offsets(self, code_obj: types.CodeType) -> Set[int]:
        return {instr.offset for instr in dis.get_instructions(code_obj)}
    
    # ========================================================================
    # SAFE CONSTANT POLLUTION
    # ========================================================================
    
    def add_constant_pollution(self, code_obj: types.CodeType, 
                               count: int = 50) -> types.CodeType:
        try:
            original_consts = list(code_obj.co_consts)
            
            # Generate diverse fake constants
            fake_consts = []
            for i in range(count):
                const_type = random.choice([
                    'int', 'float', 'str', 'bytes', 
                    'bool', 'none', 'tuple', 'frozenset'
                ])
                
                if const_type == 'int':
                    fake_consts.append(random.randint(-999999, 999999))
                elif const_type == 'float':
                    fake_consts.append(random.uniform(-1000.0, 1000.0))
                elif const_type == 'str':
                    fake_consts.append(secrets.token_hex(random.randint(4, 20)))
                elif const_type == 'bytes':
                    fake_consts.append(secrets.token_bytes(random.randint(4, 20)))
                elif const_type == 'bool':
                    fake_consts.append(random.choice([True, False]))
                elif const_type == 'none':
                    fake_consts.append(None)
                elif const_type == 'tuple':
                    fake_consts.append(tuple(random.randint(0, 100) for _ in range(random.randint(0, 5))))
                elif const_type == 'frozenset':
                    fake_consts.append(frozenset(random.randint(0, 100) for _ in range(random.randint(0, 3))))
            
            # Shuffle to avoid patterns
            random.shuffle(fake_consts)
            
            # Combine
            new_consts = tuple(original_consts + fake_consts)
            
            # Update stats
            self.stats['constants_added'] += len(fake_consts)
            
            result = code_obj.replace(co_consts=new_consts)
            
            self.log(f"Added {len(fake_consts)} fake constants", "INFO")
            
            return result
            
        except Exception as e:
            self.log(f"Constant pollution failed: {e}", "WARN")
            return code_obj  # Return unchanged on error
    
    # ========================================================================
    # SAFE LINE NUMBER OBFUSCATION
    # ========================================================================
    
    def obfuscate_line_numbers(self, code_obj: types.CodeType) -> types.CodeType:
        try:
            # Obfuscate firstlineno
            offset = random.randint(100, 5000)
            new_firstlineno = code_obj.co_firstlineno + offset
            
            result = code_obj.replace(co_firstlineno=new_firstlineno)
            
            # Python 3.10+ uses co_linetable
            if PY310_PLUS and hasattr(result, 'co_linetable'):
                original = result.co_linetable
                if len(original) > 0:
                    # Apply transformation to line table
                    # This is a delta encoding, so we modify deltas
                    scrambled = bytearray(original)
                    for i in range(len(scrambled)):
                        if i % 2 == 0:  # Line deltas
                            scrambled[i] = (scrambled[i] + random.randint(1, 10)) % 256
                    result = result.replace(co_linetable=bytes(scrambled))
            
            # Python 3.6-3.9 uses co_lnotab
            elif hasattr(result, 'co_lnotab'):
                original = result.co_lnotab
                if len(original) > 0:
                    scrambled = bytearray(original)
                    for i in range(len(scrambled)):
                        if i % 2 == 1:  # Line increments
                            scrambled[i] = (scrambled[i] + random.randint(1, 10)) % 256
                    result = result.replace(co_lnotab=bytes(scrambled))
            
            self.log("Obfuscated line numbers", "INFO")
            
            return result
            
        except Exception as e:
            self.log(f"Line obfuscation failed: {e}", "WARN")
            return code_obj
    
    # ========================================================================
    # SAFE NOP INJECTION - WITH COMPREHENSIVE EDGE CASE HANDLING
    # ========================================================================
    
    def inject_nops_ultra_safe(self, code_obj: types.CodeType,
                               min_nops: int = 1, max_nops: int = 3) -> types.CodeType:
        # Validate parameters
        if min_nops < 0 or max_nops < min_nops or max_nops > 5:
            self.log(f"Invalid NOP range: [{min_nops}, {max_nops}], skipping", "WARN")
            return code_obj
        
        try:
            # Parse instructions
            instructions = list(dis.get_instructions(code_obj))
            
            if len(instructions) == 0:
                self.log("Empty bytecode, skipping NOP injection", "INFO")
                return code_obj
            
            # Check if code is too complex
            if len(instructions) > 500:
                self.log(f"Code too large ({len(instructions)} instructions), skipping NOP injection", "WARN")
                return code_obj
            
            # Build instruction info with jump targets
            jump_targets = self._get_all_jump_targets(code_obj)
            
            instr_info = []
            for instr in instructions:
                info = InstructionInfo(
                    offset=instr.offset,
                    opcode=instr.opcode,
                    opname=instr.opname,
                    arg=instr.arg,
                    argval=instr.argval,
                    is_jump_target=(instr.offset in jump_targets),
                    size=2  # All instructions are 2 bytes in modern Python
                )
                instr_info.append(info)
            
            # Build new bytecode with NOPs
            old_code = bytearray(code_obj.co_code)
            new_code = bytearray()
            offset_map: Dict[int, int] = {}
            
            for idx, info in enumerate(instr_info):
                old_offset = info.offset
                new_offset = len(new_code)
                offset_map[old_offset] = new_offset
                
                # Copy original instruction (2 bytes)
                new_code.extend(old_code[old_offset:old_offset + 2])
                
                # Decide whether to add NOPs
                should_add_nops = (
                    idx < len(instr_info) - 1 and  # Not last instruction
                    info.opcode != self.extended_arg_opcode and  # Not EXTENDED_ARG
                    info.opcode != self.return_value_opcode  # Not RETURN (risky)
                )
                
                if should_add_nops:
                    nop_count = random.randint(min_nops, max_nops)
                    for _ in range(nop_count):
                        new_code.extend([self.nop_opcode, 0])
            
            # Recalculate jump offsets with safety checks
            new_code = self._patch_jumps_safe(
                new_code, instr_info, offset_map
            )
            
            # Create new code object
            result = code_obj.replace(co_code=bytes(new_code))
            
            # CRITICAL: Validate before returning
            is_valid, message = self.verify_code_object(result)
            
            if not is_valid:
                self.log(f"NOP injection produced invalid code: {message}", "WARN")
                return code_obj  # Return original
            
            # Update stats
            old_size = len(code_obj.co_code)
            new_size = len(new_code)
            self.stats['bytes_added'] += (new_size - old_size)
            
            self.log(f"NOP injection: {old_size} → {new_size} bytes", "SUCCESS")
            
            return result
            
        except Exception as e:
            self.log(f"NOP injection failed: {e}", "WARN")
            return code_obj  # Always return original on error
    
    def _patch_jumps_safe(self, bytecode: bytearray,
                         instructions: List[InstructionInfo],
                         offset_map: Dict[int, int]) -> bytearray:
        for info in instructions:
            # Skip non-jumps
            if info.opcode not in self.jump_opcodes:
                continue
            
            # Skip if no target
            if not isinstance(info.argval, int):
                continue
            
            old_target = info.argval
            
            # Check target is mapped
            if old_target not in offset_map:
                self.log(f"Jump target {old_target} not in offset map, skipping", "WARN")
                continue
            
            new_offset = offset_map[info.offset]
            new_target = offset_map[old_target]
            
            # Calculate new jump argument
            try:
                new_arg = self._calculate_jump_arg_safe(
                    info, new_offset, new_target
                )
                
                # Clamp to valid range
                if new_arg < 0:
                    self.log(f"Negative jump arg {new_arg}, clamping to 0", "WARN")
                    new_arg = 0
                elif new_arg > 255:
                    self.log(f"Jump arg {new_arg} > 255, clamping", "WARN")
                    new_arg = 255
                
                # Patch the argument byte
                bytecode[new_offset + 1] = new_arg
                
            except Exception as e:
                self.log(f"Failed to patch jump at {info.offset}: {e}", "WARN")
                continue
        
        return bytecode
    
    def _calculate_jump_arg_safe(self, info: InstructionInfo,
                                 new_offset: int, new_target: int) -> int:
        # Absolute jumps
        if info.opcode in self.abs_jump_opcodes:
            # Python 3.10+ uses byte offsets directly
            if PY310_PLUS:
                return new_target // 2
            else:
                return new_target // 2
        
        # Relative jumps
        elif info.opcode in self.rel_jump_opcodes:
            instr_size = info.size  # 2 bytes
            
            # Python 3.11+ has JUMP_BACKWARD
            if PY311_PLUS and info.opname == 'JUMP_BACKWARD':
                # Backward jump: distance from end of instruction to target
                distance = (new_offset + instr_size) - new_target
                return distance // 2
            else:
                # Forward jump: distance from end of instruction to target
                distance = new_target - (new_offset + instr_size)
                return distance // 2
        
        return 0
    
    # ========================================================================
    # RECURSIVE TRANSFORMATION
    # ========================================================================
    
    def transform_recursive(self, code_obj: types.CodeType,
                          transform_func) -> types.CodeType:
        try:
            # Transform this level
            result = transform_func(code_obj)
            
            # Transform nested code objects in constants
            new_consts = []
            for const in result.co_consts:
                if isinstance(const, types.CodeType):
                    try:
                        const = self.transform_recursive(const, transform_func)
                    except Exception as e:
                        self.log(f"Failed to transform nested code: {e}", "WARN")
                        # Keep original nested code on error
                new_consts.append(const)
            
            result = result.replace(co_consts=tuple(new_consts))
            
            return result
            
        except Exception as e:
            self.log(f"Recursive transformation failed: {e}", "WARN")
            return code_obj
    
    # ========================================================================
    # HIGH-LEVEL API
    # ========================================================================
    
    def transform_file(self, pyc_path: Path,
                      intensity: TransformIntensity = TransformIntensity.EXTREME,
                      min_nops: int = 1, max_nops: int = 3,
                      backup: bool = True) -> bool:
        """
        Args:
            pyc_path: Path to .pyc file
            intensity: Transformation intensity level
            min_nops: Minimum NOPs (for AGGRESSIVE+)
            max_nops: Maximum NOPs (for AGGRESSIVE+)
            backup: Create backup before transformation
        
        Returns:
            True if successful, False otherwise
        """
        
        self.log(f"Transforming {pyc_path} with intensity={intensity.name}", "INFO")
        
        try:
            # Create backup
            if backup:
                backup_path = pyc_path.with_suffix('.pyc.bak')
                backup_path.write_bytes(pyc_path.read_bytes())
                self.log(f"Created backup: {backup_path}", "INFO")
            
            # Read original
            magic, bit_field, timestamp, code_obj = self.read_pyc(pyc_path)
            
            # Validate original
            is_valid, message = self.verify_code_object(code_obj)
            if not is_valid:
                self.log(f"Original file invalid: {message}", "ERROR")
                return False
            
            original_size = len(code_obj.co_code)
            
            # Apply transformations based on intensity
            if intensity >= TransformIntensity.LIGHT:
                # Always safe - just constants
                code_obj = self.transform_recursive(
                    code_obj,
                    lambda c: self.add_constant_pollution(c, count=30)
                )
            
            if intensity >= TransformIntensity.MODERATE:
                # Safe - line number obfuscation
                code_obj = self.transform_recursive(
                    code_obj,
                    self.obfuscate_line_numbers
                )
            
            if intensity >= TransformIntensity.AGGRESSIVE:
                # Potentially risky - NOP injection
                code_obj = self.transform_recursive(
                    code_obj,
                    lambda c: self.inject_nops_ultra_safe(c, min_nops, max_nops)
                )
            
            if intensity >= TransformIntensity.EXTREME:
                # Extra aggressive
                code_obj = self.transform_recursive(
                    code_obj,
                    lambda c: self.add_constant_pollution(c, count=50)
                )
            
            # Validate transformed
            is_valid, message = self.verify_code_object(code_obj)
            if not is_valid:
                self.log(f"Transformed code invalid: {message}", "ERROR")
                if backup:
                    self.log("Restoring from backup...", "INFO")
                    pyc_path.write_bytes(backup_path.read_bytes())
                return False
            
            # Write transformed
            self.write_pyc(pyc_path, magic, bit_field, timestamp, code_obj)
            
            # Report stats
            new_size = len(code_obj.co_code)
            growth = ((new_size - original_size) / original_size) * 100 if original_size > 0 else 0
            
            self.log(f"Success! Size: {original_size} → {new_size} (+{growth:.1f}%)", "SUCCESS")
            
            self.stats['files_processed'] += 1
            self.stats['files_succeeded'] += 1
            
            return True
            
        except Exception as e:
            self.log(f"Transformation failed: {e}", "ERROR")
            self.stats['files_processed'] += 1
            self.stats['files_failed'] += 1
            return False
    
    def get_statistics(self) -> Dict:
        return self.stats.copy()
class BreakerAST:
    def __init__(self, workspace_dir: str = "workspace", version: Optional[Tuple[int, int]] = None):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(exist_ok=True)
        self.breaker = MainBreaker()
        self.bytecode_transformer = BytecodeTransformer(version=version if version else PY_VERSION)
    
    def gen_break_ast(self, preset: str = "balanced", debug: bool = False) -> ast.Module:
        body = []
        
        if debug:
            print(f"[*] Using {preset.upper()} preset")
        if preset == "vodka":
            if debug:
                print("[*] PRIMARY: Generating LOAD_GLOBAL triggers (300 instances - VODKA MODE)...")
            body.extend(self.breaker.ld_glob_err(count=300))
            
            if debug:
                print("[*] PRIMARY: Generating call chains (20x - VODKA MODE)...")
            for _ in range(20):
                body.append(self.breaker.call_chain_err(length=150))
        else:
            if debug:
                print("[*] PRIMARY: Generating LOAD_GLOBAL triggers (200 instances)...")
            body.extend(self.breaker.ld_glob_err(count=200))
            
            if debug:
                print("[*] PRIMARY: Generating call chains...")
            for _ in range(8):
                body.append(self.breaker.call_chain_err(length=100))
        
        if preset == "minimal":

            if debug:
                print("[*] Minimal preset: PRIMARY attack only")
            
        elif preset == "oom":

            if debug:
                print("[*] OOM: Generating memory bombs...")
            body.extend(self.breaker.mem_bomb_oom(count=100))
            

            if debug:
                print("[*] OOM: Adding complexity boosters...")
            for _ in range(5):
                body.append(self.breaker.comp_nest(depth=5))
            
            for _ in range(5):
                body.append(self.breaker.lambda_nest(depth=12))
            
        elif preset == "vodka":

            if debug:
                print("[*] VODKA: Generating FULL control flow timeouts...")
            body.extend(self.breaker.try_nest_timeout(depth=18, count=30))
            
            if debug:
                print("[*] VODKA: Generating MASSIVE boolean chains...")
            for _ in range(8):
                body.extend(self.breaker.bool_chain_timeout(count=200))
            
            if debug:
                print("[*] VODKA: Generating DEEP comprehensions...")
            for _ in range(25):
                body.append(self.breaker.comp_nest(depth=6))
            
            if debug:
                print("[*] VODKA: Generating DEEP lambdas...")
            for _ in range(30):
                body.append(self.breaker.lambda_nest(depth=15))
            
            if debug:
                print("[*] VODKA: Generating LONG walrus chains...")
            for _ in range(15):
                body.append(self.breaker.walrus_chain(count=100))
            
            if debug:
                print("[*] VODKA: Generating match cases...")
            body.extend(self.breaker.match_cases())
            
            if debug:
                print("[*] VODKA: Generating MASSIVE memory bombs...")
            body.extend(self.breaker.mem_bomb_oom(count=150))
            
            if debug:
                print("[*] VODKA: Flooding constant pool...")
            body.extend(self.breaker.const_flood(count=500))
            
            if debug:
                print("[*] VODKA: Creating decorator chaos...")
            body.extend(self.breaker.decorator_chaos(count=20))
            
            if debug:
                print("[*] VODKA: Nesting classes deeply...")
            body.extend(self.breaker.class_nest_bomb(depth=10, count=5))
            
            if debug:
                print("[*] VODKA: Bombing async patterns...")
            body.extend(self.breaker.async_bomb(count=15))
            
            if debug:
                print("[*] VODKA: Chaos slicing...")
            body.extend(self.breaker.slice_chaos(count=50))
            
            if debug:
                print("[*] VODKA: Chaining attributes...")
            body.extend(self.breaker.attr_chain_bomb(count=30))
            
            if debug:
                print("[*] VODKA: Starring expressions...")
            body.extend(self.breaker.starred_chaos(count=40))
            
            if debug:
                print("[*] VODKA: Yielding generators...")
            body.extend(self.breaker.yield_bomb(count=20))
            
            if debug:
                print("[*] VODKA: Adding extra spice...")

            for i in range(10):
                body.append(self.breaker.call_chain_err(length=50 + i * 10))
            

            for _ in range(10):
                body.append(self.breaker.comp_nest(depth=4))
                body.append(self.breaker.lambda_nest(depth=8))
            
            if debug:
                print("[*] VODKA: Full strength achieved! 🍸💥🔥")
            
        elif preset == "maximum":

            if debug:
                print("[*] BACKUP: Generating control flow timeouts (FULL)...")
            body.extend(self.breaker.try_nest_timeout(depth=18, count=20))
            
            for _ in range(4):
                body.extend(self.breaker.bool_chain_timeout(count=150))
            
            for _ in range(15):
                body.append(self.breaker.comp_nest(depth=5))
            
            for _ in range(20):
                body.append(self.breaker.lambda_nest(depth=12))
            
            for _ in range(8):
                body.append(self.breaker.walrus_chain(count=75))
            
            body.extend(self.breaker.match_cases())
            
        else:

            if debug:
                print("[*] BACKUP: Generating control flow timeouts (MODERATE)...")
            body.extend(self.breaker.try_nest_timeout(depth=18, count=12))
            
            for _ in range(2):
                body.extend(self.breaker.bool_chain_timeout(count=100))
            
            for _ in range(10):
                body.append(self.breaker.comp_nest(depth=5))
            
            for _ in range(12):
                body.append(self.breaker.lambda_nest(depth=12))
            
            for _ in range(5):
                body.append(self.breaker.walrus_chain(count=50))
            
            body.extend(self.breaker.match_cases())
        
        module = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(module)
        
        return module
    
    def break_code(self, original_code: str, output_file: Path, preset: str = "balanced", 
                   bytecode_transform: bool = True, debug: bool = False) -> Path:
        """
        Args:
            original_code: Original Python source code
            output_file: Output .pyc file path
            preset: Attack preset to use
            bytecode_transform: Apply bytecode transformation
            debug: Enable debug output
            
        Returns:
            Path to output .pyc file
        """
        if debug:
            print(f"[DEBUG] Preset: {preset}")
            print(f"[DEBUG] Bytecode transform: {bytecode_transform}")
        

        breaking_ast = self.gen_break_ast(preset, debug)
        

        try:
            original_ast = ast.parse(original_code)
        except SyntaxError as e:
            if debug:
                print(f"[DEBUG] Syntax error: {e}")
            raise
        

        combined = ast.Module(
            body=breaking_ast.body + original_ast.body,
            type_ignores=[]
        )
        ast.fix_missing_locations(combined)
        
        if debug:
            print(f"[DEBUG] Combined AST nodes: {len(combined.body)}")
        

        code = ast.unparse(combined)
        
        if debug:
            print(f"[DEBUG] Combined source size: {len(code)} bytes")
        

        temp = self.workspace / f"t_{secrets.token_hex(4)}.py"
        
        try:
            temp.write_text(code, encoding='utf-8')
            
            if debug:
                print(f"[DEBUG] Compiling with optimize=2")
            
            py_compile.compile(
                str(temp),
                cfile=str(output_file),
                optimize=2,
                doraise=True
            )
            
            if debug:
                print(f"[DEBUG] Compiled to {output_file}")
            

            if bytecode_transform:
                intensity = "vodka" if preset == "vodka" else "high" if preset == "maximum" else "medium"
                
                if debug:
                    print(f"[DEBUG] Transforming bytecode: {intensity}")
                
                self.bytecode_transformer.transform_file(output_file, TransformIntensity.EXTREME if intensity == "vodka" else TransformIntensity.AGGRESSIVE if intensity == "high" else TransformIntensity.MODERATE,min_nops=8, max_nops=10)

                if debug:
                    print(f"[DEBUG] Bytecode transformed")
            
            return output_file
            
        finally:
            if temp.exists():
                temp.unlink()
    
    def break_file(self, input_file: Path, output_file: Path,
                   preset: str = "balanced", keep_debug: bool = False,
                   bytecode_transform: bool = True, debug: bool = False) -> Path:
        """
        Args:
            input_file: Input .py file
            output_file: Output .pyc file
            preset: Attack preset
            keep_debug: Keep debug .py file with combined source
            bytecode_transform: Apply bytecode transformation
            debug: Enable debug output
        """
        if not input_file.exists():
            raise FileNotFoundError(f"Not found: {input_file}")
        
        if debug:
            print(f"[DEBUG] Reading: {input_file}")
        
        original = input_file.read_text(encoding='utf-8')
        

        if keep_debug:

            breaking_ast = self.gen_break_ast(preset, debug=False)
            original_ast = ast.parse(original)
            combined = ast.Module(body=breaking_ast.body + original_ast.body, type_ignores=[])
            ast.fix_missing_locations(combined)
            debug_code = ast.unparse(combined)
            
            debug_file = self.workspace / f"{output_file.stem}_debug.py"
            debug_file.write_text(debug_code, encoding='utf-8')
            
            if debug:
                print(f"[DEBUG] Saved debug file: {debug_file}")
        

        return self.break_code(original, output_file, preset, bytecode_transform, debug)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='vodka',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Attack Strategy:
  PRIMARY: LOAD_GLOBAL KeyError 
  BACKUP:  Control flow timeout (3600s)
  OOM:     Out of memory during decompilation
  VODKA:   IDK

Presets:
  minimal  - PRIMARY only,
  balanced - PRIMARY + moderate backup 
  maximum  - PRIMARY + full backup 
  oom      - PRIMARY + OOM bomb
  vodka    - IDK
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input .py file')
    parser.add_argument('-o', '--output', required=True, help='Output .pyc file')
    parser.add_argument('-p', '--preset',
                       choices=['minimal', 'balanced', 'maximum', 'oom', 'vodka'],
                       default='balanced',
                       help='Attack preset (default: balanced)')
    parser.add_argument('-d', '--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--keep-debug', action='store_true',
                       help='Keep debug .py file with combined source')
    parser.add_argument('-w', '--workspace', default='workspace',
                       help='Workspace directory')
    parser.add_argument('--no-bytecode', action='store_true',
                       help='Disable bytecode transformation')
    
    args = parser.parse_args()
    
    if sys.version_info < (3, 9):
        print("[!] Requires Python 3.9+")
        sys.exit(1)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not output_path.suffix == '.pyc':
        output_path = output_path.with_suffix('.pyc')
    
    breaker = BreakerAST(workspace_dir=args.workspace)
    
    try:
        
        breaker.break_file(input_path, output_path,
                          preset=args.preset, keep_debug=args.keep_debug,
                          bytecode_transform=not args.no_bytecode, debug=args.debug)
        print("[+] generated") # left this to make users acknowledge completion.
        sys.exit(0)
        
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    __name__+='vodka'
    main()