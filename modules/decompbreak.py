import ast
import py_compile
import dis
import sys
from pathlib import Path
from typing import List, Dict
import secrets
import marshal
import types
import random


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


class BytecodeTransformer: 
    def __init__(self):
        self.py_version = sys.version_info[:2]
    
    def read_pyc(self, pyc_path: Path) -> tuple:
        with open(pyc_path, 'rb') as f:

            magic = f.read(4)
            
            if self.py_version >= (3, 7):

                flags = f.read(4)
            
            if self.py_version >= (3, 3):
                timestamp = f.read(4)
                size = f.read(4)
            else:
                timestamp = f.read(4)
                size = None
            

            code_obj = marshal.load(f)
            
            return magic, timestamp, size, code_obj
    
    def write_pyc(self, pyc_path: Path, magic: bytes, timestamp: bytes, 
                  size: bytes, code_obj: types.CodeType):
        with open(pyc_path, 'wb') as f:
            f.write(magic)
            
            if self.py_version >= (3, 7):
                f.write(b'\x00\x00\x00\x00')  # flags
            
            f.write(timestamp)
            if size:
                f.write(size)
            
            marshal.dump(code_obj, f)
    
    def _get_instruction_sizes(self, code: types.CodeType) -> Dict[int, int]:
        instructions = list(dis.get_instructions(code))
        sizes = {}
        
        for i, instr in enumerate(instructions):
            if i < len(instructions) - 1:

                size = instructions[i + 1].offset - instr.offset
            else:

                size = len(code.co_code) - instr.offset
            
            sizes[instr.offset] = size
        
        return sizes
    
    def inject_nops(self, code: types.CodeType, min_nops: int, max_nops: int) -> types.CodeType:
        original = bytearray(code.co_code)
        instructions = list(dis.get_instructions(code))
        
        if not instructions:
            return code
        

        instr_sizes = self._get_instruction_sizes(code)
        

        new_code = bytearray()
        old_to_new: Dict[int, int] = {}
        nop_op = dis.opmap['NOP']
        extended_arg_op = dis.opmap.get('EXTENDED_ARG', -1)
        
        for idx, instr in enumerate(instructions):
            old_offset = instr.offset
            new_offset = len(new_code)
            old_to_new[old_offset] = new_offset
            

            instr_size = instr_sizes[old_offset]
            new_code.extend(original[old_offset:old_offset + instr_size])
            



            if instr.opcode == extended_arg_op or idx == len(instructions) - 1:
                continue
            

            nop_count = random.randint(min_nops, max_nops)
            for _ in range(nop_count):
                new_code.extend([nop_op, 0])
        

        patch_count = 0
        jump_opcodes = set(dis.hasjrel) | set(dis.hasjabs)
        for instr in instructions:
            if instr.opcode not in jump_opcodes:
                continue
            
            new_offset = old_to_new[instr.offset]
            instr_size = instr_sizes[instr.offset]
            

            if instr.opcode in dis.hasjrel:

                if instr.opname == 'JUMP_BACKWARD':

                    old_target = (instr.offset + instr_size) - (instr.arg or 0) * 2
                else:

                    old_target = instr.offset + instr_size + (instr.arg or 0) * 2
                
                if old_target in old_to_new:
                    new_target = old_to_new[old_target]
                    

                    if instr.opname == 'JUMP_BACKWARD':

                        new_distance = (new_offset + instr_size) - new_target
                        new_arg = new_distance // 2
                    else:

                        new_distance = new_target - (new_offset + instr_size)
                        new_arg = new_distance // 2
                    

                    if new_arg < 0 or new_arg > 255:
                        new_arg = max(0, min(255, new_arg))
                    

                    new_code[new_offset + 1] = new_arg
                    patch_count += 1
                    

            elif instr.opcode in dis.hasjabs:
                old_target = (instr.arg or 0) * 2
                
                if old_target in old_to_new:
                    new_target = old_to_new[old_target]
                    new_arg = new_target // 2
                    
                    if new_arg > 255:
                        new_arg = new_arg & 0xFF
                    
                    new_code[new_offset + 1] = new_arg
                    patch_count += 1
        

        new_code_obj = code.replace(co_code=bytes(new_code))
        

        new_consts = []
        for const in new_code_obj.co_consts:
            if isinstance(const, types.CodeType):
                const = self.inject_nops(const, min_nops, max_nops)
            new_consts.append(const)
        
        new_code_obj = new_code_obj.replace(co_consts=tuple(new_consts))
        
        return new_code_obj
    
    def add_nops_between_instructions(self, source: str, min_nops: int = 1,
                                     max_nops: int = 21):
        """
        Args:
            source: Path to .pyc file to modify (will be overwritten) or code object
            min_nops: Minimum number of NOPs to inject after each instruction
            max_nops: Maximum number of NOPs to inject after each instruction
        
        Note: Large max_nops values (>10) may cause jump distance overflows
        """
        if min_nops < 0 or max_nops < min_nops:
            raise ValueError(f"Invalid NOP range: min={min_nops}, max={max_nops}")
        
        if max_nops > 15:
            print(f"⚠ Warning: max_nops={max_nops} is high and may cause issues")
        
        if not isinstance(source, types.CodeType):
            magic, bit_field, mtime_or_hash, code = self.read_pyc(source)
        else:
            code = source

        new_code = self.inject_nops(code, min_nops, max_nops)
        
        if not isinstance(source, types.CodeType):
            self.write_pyc(source, magic, bit_field, mtime_or_hash, new_code)
        else:
            return new_code
        # return types.CodeType(
        #     code_obj.co_argcount,
        #     code_obj.co_posonlyargcount,
        #     code_obj.co_kwonlyargcount,
        #     code_obj.co_nlocals,
        #     code_obj.co_stacksize,
        #     code_obj.co_flags,
        #     bytes(new_code),
        #     code_obj.co_consts,
        #     code_obj.co_names,
        #     code_obj.co_varnames,
        #     code_obj.co_filename,
        #     code_obj.co_name,
        #     code_obj.co_qualname if hasattr(code_obj, "co_qualname") else code_obj.co_name,
        #     code_obj.co_firstlineno,
        #     code_obj.co_linetable if hasattr(code_obj, "co_linetable") else code_obj.co_lnotab,
        #     code_obj.co_exceptiontable if hasattr(code_obj, "co_exceptiontable") else b"",
        #     code_obj.co_freevars,
        #     code_obj.co_cellvars
        # )
    
    def mangle_line_numbers(self, code_obj: types.CodeType) -> types.CodeType:
        if sys.version_info >= (3, 10):

            if hasattr(code_obj, 'co_linetable'):

                mangled_linetable = bytes(
                    (b + 13) % 256 for b in code_obj.co_linetable
                )
                
                code_obj = code_obj.replace(co_linetable=mangled_linetable)
        else:

            if hasattr(code_obj, 'co_lnotab'):
                mangled_lnotab = bytes(
                    (b + 13) % 256 for b in code_obj.co_lnotab
                )
                
                code_obj = code_obj.replace(co_lnotab=mangled_lnotab)
        
        return code_obj
    
    def add_unreachable_code(self, bytecode: bytes) -> bytes:

        if sys.version_info >= (3, 11):

            unreachable = b'\x72\x14'
            unreachable += b'\x64\x00' * 10
            unreachable += b'\x53\x00'
        else:

            unreachable = b'\x6e\x0a'
            unreachable += b'\x64\x00' * 5
            unreachable += b'\x53'
        

        result = bytecode[:10] + unreachable + bytecode[10:]
        return result
    
    def duplicate_constants(self, code_obj: types.CodeType) -> types.CodeType:
        if hasattr(code_obj, 'co_consts'):
            original_consts = list(code_obj.co_consts)
            

            duplicates = []
            for const in original_consts[:min(len(original_consts), 50)]:
                if isinstance(const, (int, float, str)):
                    duplicates.append(const)
            
            new_consts = tuple(original_consts + duplicates)
            code_obj = code_obj.replace(co_consts=new_consts)
        
        return code_obj
    
    def transform_bytecode(self, pyc_path: Path, intensity: str = "high"):
        """
        Args:
            pyc_path: Path to .pyc file
            intensity: "low", "medium", "high", "vodka"
        """

        magic, timestamp, size, code_obj = self.read_pyc(pyc_path)
        

        if intensity in ["medium", "high", "vodka"]:
            code_obj = self.mangle_line_numbers(code_obj)
        
        if intensity in ["high", "vodka"]:
            code_obj = self.duplicate_constants(code_obj)
        
        if intensity == "vodka":

            code_obj = self._transform_nested(code_obj)
        

        self.write_pyc(pyc_path, magic, timestamp, size, code_obj)
    
    def _transform_nested(self, code_obj: types.CodeType) -> types.CodeType:
        if hasattr(code_obj, 'co_consts'):
            new_consts = []
            for const in code_obj.co_consts:
                if isinstance(const, types.CodeType):

                    const = self.mangle_line_numbers(const)
                    const = self.duplicate_constants(const)
                    const = self._transform_nested(const)
                new_consts.append(const)
            
            code_obj = code_obj.replace(co_consts=tuple(new_consts))
        
        return code_obj


class BreakerAST:
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(exist_ok=True)
        self.breaker = MainBreaker()
        self.bytecode_transformer = BytecodeTransformer()
    
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
                
                self.bytecode_transformer.transform_bytecode(output_file, intensity)

                # self.bytecode_transformer.add_nops_between_instructions(output_file, min_nops=5, max_nops=10) # isnt working. Crashes upon run attempts.

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
        description='Dual-Attack PyLingual Breaker - LOAD_GLOBAL error + Control flow timeout + OOM + VODKA MODE 🍸',
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
    main()