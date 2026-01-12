#
import ast
import py_compile
import os
import sys
from typing import List, Any
from utils import RandomGenerator
sys.setrecursionlimit(1_000_000_000) # maximum value is 2,147,483,647??

WORKSPACE_DIR = os.path.abspath("workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

class Bomber3000:    
    def __init__(self):
        self.var_counter = 0
        self.random=RandomGenerator()
    def _fresh_var(self, prefix="_v"):
        """Generate unique variable name"""
        # hash prefix to avoid attackers to attack lolololo
        
        prefix=hash(prefix+self.random.random_id()) % self.random.random_int(1000, 2000)
        self.var_counter += 1
        return f"_{hex(prefix)}{self.var_counter}"
    
    def try_except(self, depth=18) -> List[ast.stmt]:
        depth = min(depth, 20)  # Limit depth to avoid excessive nesting
        statements = []
        
        # Start with innermost statement
        body = [ast.Assign(
            targets=[ast.Name(id=self._fresh_var(), ctx=ast.Store())],
            value=ast.Constant(value=42)
        )]
        
        # Wrap in nested try-except (stay under 20 limit)
        for i in range(depth):
            body = [ast.Try(
                body=[
                    ast.Assign(
                        targets=[ast.Name(id=self._fresh_var(f"_t{i}_"), ctx=ast.Store())],
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
                    body=[
                        ast.Assign(
                            targets=[ast.Name(id=self._fresh_var(f"_e{i}_"), ctx=ast.Store())],
                            value=ast.Constant(value=0)
                        )
                    ]
                )],
                orelse=[],
                finalbody=[]
            )]
        
        return body
    
    def bool_chain(self, count=200) -> List[ast.stmt]:
        statements = []
        
        # First, define all the variables we'll use
        var_names = []
        for i in range(count):
            var_name = self._fresh_var(f'_bool')
            var_names.append(var_name)
            statements.append(ast.Assign(
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
        
        # Chain them all with 'and'
        result = comparisons[0]
        for comp in comparisons[1:]:
            result = ast.BoolOp(
                op=ast.And(),
                values=[result, comp]
            )
        
        statements.append(ast.Assign(
            targets=[ast.Name(id='_bool_mega', ctx=ast.Store())],
            value=result
        ))
        
        return statements
    
    def comp(self, depth=5) -> ast.stmt:
        # Innermost comprehension
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
        
        # Wrap in multiple layers (stay safe)
        for level in range(depth):
            var = f'_{chr(97 + (level % 26))}{level}'
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
            targets=[ast.Name(id=self._fresh_var('_comp'), ctx=ast.Store())],
            value=comp
        )
    
    def lambda_chain(self, depth=15) -> ast.stmt:
        expr = ast.Constant(value=42)
        
        for i in range(depth):
            expr = ast.Call(
                func=ast.Lambda(
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg=f'_x{i}', annotation=None)],
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
            targets=[ast.Name(id=self._fresh_var('_lambda'), ctx=ast.Store())],
            value=expr
        )
    
    def call_chain_flat(self, length=200) -> ast.stmt:
        expr = ast.Constant(value=1)
        
        for i in range(length):
            if i % 3 == 0:
                func = 'abs'
            elif i % 3 == 1:
                func = 'int'
            else:
                func = 'float'
            
            expr = ast.Call(
                func=ast.Name(id=func, ctx=ast.Load()),
                args=[expr],
                keywords=[]
            )
        
        return ast.Assign(
            targets=[ast.Name(id=self._fresh_var('_chain'), ctx=ast.Store())],
            value=expr
        )
    
    def walrus_chain(self, count=100) -> ast.stmt:
        exprs = []
        for i in range(count):
            exprs.append(ast.NamedExpr(
                target=ast.Name(id=self._fresh_var(f'_w{i}_'), ctx=ast.Store()),
                value=ast.BinOp(
                    left=ast.Constant(value=i),
                    op=ast.Mult(),
                    right=ast.Constant(value=2)
                )
            ))
        
        # Chain with 'and'
        result = exprs[0]
        for expr in exprs[1:]:
            result = ast.BoolOp(
                op=ast.And(),
                values=[result, expr]
            )
        
        return ast.Expr(value=result)
    
    def match_state(self) -> List[ast.FunctionDef]:
        if sys.version_info < (3, 10):
            return []
        
        functions = []
        
        for idx in range(20):  # 20 match functions
            func_name = f'_match_func_{idx}'
            
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
                            keys=[
                                ast.Constant(value='a'),
                                ast.Constant(value='b')
                            ],
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
                    args=[ast.arg(arg='val', annotation=None)],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[]
                ),
                body=[match_stmt],
                decorator_list=[],
                returns=None
            )
            
            functions.append(func)
        
        return functions
    
    def keyerror_trigger(self, count=150) -> List[ast.stmt]:
        assignments = []
        
        for i in range(count):
            # Pattern 1: result = abs(constant)
            assignments.append(ast.Assign(
                targets=[ast.Name(id=self._fresh_var(f'_abs{i}_'), ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='abs', ctx=ast.Load()),
                    args=[ast.Constant(value=i * 42)],
                    keywords=[]
                )
            ))
            
            # Pattern 2: result = int(constant)
            assignments.append(ast.Assign(
                targets=[ast.Name(id=self._fresh_var(f'_int{i}_'), ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='int', ctx=ast.Load()),
                    args=[ast.Constant(value=3.14 * i)],
                    keywords=[]
                )
            ))
             
            assignments.append(ast.Assign(
                targets=[ast.Name(id=self._fresh_var(f'_float{i}_'), ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='float', ctx=ast.Load()),
                    args=[ast.Constant(value=i)],
                    keywords=[]
                )
            ))
        
        return assignments


class BreakerAST:
    def __init__(self):
        self.bomb = Bomber3000()
    
    def gen_ast(self) -> ast.Module:
        body = []
        
        for i in range(30):
            body.extend(self.bomb.try_except(depth=18))
        for i in range(5):
            body.extend(self.bomb.bool_chain(count=200))
        for i in range(20):
            body.append(self.bomb.comp(depth=5))
    
        for i in range(30):
            body.append(self.bomb.lambda_chain(depth=15))
        
        for i in range(10):
            body.append(self.bomb.call_chain_flat(length=200))

        for i in range(10):
            body.append(self.bomb.walrus_chain(count=100))

        match_funcs = self.bomb.match_state()
        if match_funcs:
            body.extend(match_funcs)
        else:
            print("skipped match statements (Python < 3.10)")

        body.extend(self.bomb.keyerror_trigger(count=150))
        
        module = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(module)
        
        return module
    
    def gimmecode(self, original_code: str, obfuscation_ast: ast.Module) -> str:
        try:
            original_ast = ast.parse(original_code)
        except SyntaxError as e:
            print(f"[!] Original code has syntax error: {e}")
            raise
        
        combined = ast.Module(
            body=obfuscation_ast.body + original_ast.body,
            type_ignores=[]
        )
        
        ast.fix_missing_locations(combined)
        
        if hasattr(ast, 'unparse'):
            return ast.unparse(combined)
        else:
            raise RuntimeError("Need Python 3.9+ for ast.unparse")
    
    def gimmefile(self, input_file: str, output_file: str):
        with open(input_file, 'r', encoding='utf-8') as f:
            original_code = f.read()
        
        ast_tree = self.gen_ast()
        estimated_instructions = 0
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Try):
                estimated_instructions += 8
            elif isinstance(node, ast.BoolOp):
                estimated_instructions += 6
            elif isinstance(node, ast.Call):
                estimated_instructions += 4
            elif isinstance(node, ast.Lambda):
                estimated_instructions += 5
            elif isinstance(node, ast.NamedExpr):
                estimated_instructions += 3
            elif isinstance(node, ast.ListComp):
                estimated_instructions += 10
            elif isinstance(node, ast.Assign):
                estimated_instructions += 2
        outout = self.gimmecode(original_code, ast_tree)

        base_name = os.path.basename(output_file).replace(".pyc", "")

        debug_file = os.path.join(WORKSPACE_DIR, f"{base_name}_q.py")

        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(outout)
        
        temp_py   = os.path.join(WORKSPACE_DIR, f"{base_name}_w.py") # also put this in workspace then clean up later
        
        try:
            with open(temp_py, 'w', encoding='utf-8') as f:
                f.write(outout)
            
            py_compile.compile(
                temp_py,
                cfile=output_file,
                optimize=2,
                doraise=True
            )
        except SyntaxError as e:
            print(f"[!] A wild syntax error has occurred. Check debug file: {debug_file} - workspace folder")
            raise
        except Exception as e:
            print(f"[!] Compilation error: {e}")
            raise
        finally:
            if os.path.exists(temp_py):
                os.remove(temp_py)
        
        # No shit i will execute

        # print(f"\n[+] Testing execution...")
        # try:
        #     import subprocess
        #     result = subprocess.run(
        #         [sys.executable, output_file],
        #         capture_output=True,
        #         timeout=30,
        #         text=True
        #     )
            
        #     if result.returncode == 0:
        #         print(f"[✓] Bytecode executes successfully!")
        #         if result.stdout:
        #             print(f"\nOutput preview:\n{result.stdout[:200]}")
        #     else:
        #         print(f"[!] Execution failed: {result.returncode}")
        #         if result.stderr:
        #             print(f"Error: {result.stderr[:500]}")
        # except subprocess.TimeoutExpired:
        #     print(f"[✓] Execution timeout (expected with {estimated_instructions:,} instructions)")
        # except Exception as e:
        #     print(f"[!] Test error: {e}")

        return output_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='vodka',
        epilog='vodka is good for your mental health',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input .py file')
    parser.add_argument('-o', '--output', required=True, help='Output .pyc file')
    
    args = parser.parse_args()
    
    if sys.version_info < (3, 9):
        print("Error: Requires Python 3.9+ for ast.unparse")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if not args.output.endswith('.pyc'):
        args.output += '.pyc'
    
    obfuscator = BreakerAST()
    
    try:
        result = obfuscator.gimmefile(args.input, args.output)
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()