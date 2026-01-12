from .type import FunctionMetadata
from .base import SecurityLevel, IntegrityError
from .binary_container import BinaryContainer
from .security_monitor import SecurityMonitor
from .memory_loader import MemoryLoader
from .hardware_binding import HardwareBinding
from .data import PythonObfuscator
from .utils import RandomGenerator
from typing import List, Optional
import hashlib
import sys
import os
import ctypes
class CompiledLibrary:
    def __init__(self, dll_bytes: bytes, func_metadata: List[FunctionMetadata],
                 security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.dll_bytes = dll_bytes
        self.func_metadata = func_metadata
        self.security_level = security_level
        
        # Create security monitor
        self.security = SecurityMonitor(enabled=(security_level.value >= SecurityLevel.STANDARD.value))
        
        # Verify integrity
        self.dll_hash = hashlib.sha256(dll_bytes).digest()
        if not self.security.verify_integrity(dll_bytes, self.dll_hash):
            raise IntegrityError("DLL integrity check failed")
        
        # Load library
        self.security.anti_tamper_check()
        self.lib = MemoryLoader.load(dll_bytes)
        
        # Setup function pointers
        self._setup_functions()
        
        # Post-load integrity check
        if not self.security.verify_integrity(dll_bytes, self.dll_hash):
            raise IntegrityError("Post-load integrity check failed")
    
    def _setup_functions(self):
        """Setup function pointers with obfuscated names"""
        self.functions = {}
        
        for meta in self.func_metadata:
            try:
                func = getattr(self.lib, meta.obfuscated_name)
                func.restype = ctypes.c_int64
                func.argtypes = [ctypes.c_int64] * meta.param_count
                
                # Store by index only (no name exposure)
                self.functions[meta.index] = func
            except AttributeError:
                pass  # Silent failure
    
    def call(self, func_index: int, *args) -> int:
        """Call function by index (minimal API surface)"""
        # Anti-tamper check before each call
        if self.security_level.value >= SecurityLevel.AGGRESSIVE.value:
            self.security.anti_tamper_check()
        
        if func_index not in self.functions:
            return 0  # Silent failure
        
        func = self.functions[func_index]
        
        try:
            result = func(*args)
            return int(result)
        except:
            return 0  # Silent failure
    
    def get_function_count(self) -> int:
        """Get number of available functions"""
        return len(self.functions)
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes (encrypted container)"""
        container = BinaryContainer()
        
        # Add DLL - compress only
        container.add_section("dll", self.dll_bytes, compress=True, encrypt=False)
        
        # Add metadata as JSON (not pickle - avoid class dependencies)
        import json
        metadata_dict = {
            'functions': [
                {
                    'original_name': m.original_name,
                    'obfuscated_name': m.obfuscated_name,
                    'param_count': m.param_count,
                    'index': m.index
                }
                for m in self.func_metadata
            ]
        }
        metadata_bytes = json.dumps(metadata_dict).encode()
        container.add_section("metadata", metadata_bytes, compress=True, encrypt=False)
        
        return container.pack()
    
    @staticmethod
    def from_bytes(data: bytes, security_level: SecurityLevel = SecurityLevel.STANDARD):
        """Deserialize from bytes"""
        container = BinaryContainer()
        sections = container.unpack(data)
        
        dll_bytes = sections["dll"]
        
        import pickle
        func_metadata = pickle.loads(sections["metadata"])
        
        return CompiledLibrary(dll_bytes, func_metadata, security_level)
    
    def to_python_code(self, module_name: str,
                         hardware_binding: bool = False):
        """Generate standalone obfuscated Python module with all phases"""
        # Encode to base64
        import base64
        container_bytes = self.to_bytes()
        encoded = base64.b64encode(container_bytes).decode('ascii')
        
        # Phase 10: Hardware binding
        hw_cpu = HardwareBinding.get_cpu_id() if hardware_binding else None
        hw_mac = HardwareBinding.get_mac_address() if hardware_binding else None
        
        # Phase 7: Generate obfuscated loader code
        py_obfuscator = PythonObfuscator(self.security_level)
        obf_loader = self._generate_obfuscated_loader(module_name, encoded, 
                                                      hw_cpu, hw_mac, py_obfuscator)
        
        return obf_loader, hw_cpu, hw_mac
        

    def to_python_module(self, output_path: str, module_name: str,
                         hardware_binding: bool = False):
        """Generate standalone obfuscated Python module with all phases"""
        obf_loader, hw_cpu, hw_mac = self.to_python_code(module_name, hardware_binding)
        with open(output_path, 'w') as f:
            f.write(obf_loader)
        
        if not sys.platform.startswith("win"):
            os.chmod(output_path, 0o755)
        
        print(f"[✓] Generated module: {output_path}")
        print(f"[*] Size: {os.path.getsize(output_path):,} bytes")
        print(f"[*] Security: {self.security_level.name}")
        if hardware_binding:
            print(f"[*] Hardware-bound: CPU={hw_cpu[:8]}..., MAC={hw_mac[:8]}...")
    
    def _generate_obfuscated_loader(self, module_name: str, encoded_data: str,
                                   hw_cpu: Optional[str], hw_mac: Optional[str],
                                   py_obf: PythonObfuscator) -> str:
        """Generate obfuscated Python loader with anti-analysis (Phases 6, 7, 10)"""
        
        # Obfuscate variable names
        var1 = f"_{RandomGenerator.random_id()}"
        var2 = f"_{RandomGenerator.random_id()}"
        var3 = f"_{RandomGenerator.random_id()}"
        var4 = f"_{RandomGenerator.random_id()}"
        
        # Anti-debug checks only for non-MINIMAL security
        anti_debug_checks = ""
        if self.security_level.value >= SecurityLevel.STANDARD.value:
            anti_debug_checks = f"""
# Anti-analysis checks (module load time only)
def _check_debug():
    if {var1}.gettrace() is not None: 
        return False
    return True

if not _check_debug(): {var1}.exit(1)
"""
        
        # Phase 10: Hardware binding checks
        hw_checks = ""
        if hw_cpu or hw_mac:
            hw_checks = f"""
# Hardware binding verification
def _verify_hw():
    import hashlib, uuid
    {'if hashlib.sha256(open("/proc/cpuinfo").read().encode()).hexdigest()[:16] != "' + hw_cpu + '": return False' if hw_cpu else ''}
    {'if hashlib.sha256(str(uuid.getnode()).encode()).hexdigest()[:16] != "' + hw_mac + '": return False' if hw_mac else ''}
    return True

if not _verify_hw(): {var1}.exit(1)
"""
        
        # Phase 7: Fake functions
        fake_functions = py_obf.add_fake_functions()
        
        # Phase 7: Namespace pollution
        namespace_pollution = py_obf.add_namespace_pollution()
        
        # Get the encryption key from the container
        key_hex = self.dll_hash.hex()
        
        template = f'''#!/usr/bin/env python3
"""Auto-generated secure module - DO NOT MODIFY"""
import sys as {var1}, ctypes as {var2}, base64 as {var3}, hashlib as {var4}
{anti_debug_checks}
{hw_checks}
# Phase 7: Namespace pollution
{namespace_pollution}

# Phase 7: Fake functions
{fake_functions}

# Embedded encrypted data
_d = """{encoded_data}"""

class _L:
    def __init__(self):
        try:
            # Decode and load
            _b = {var3}.b64decode(_d)
            _h = {var4}.sha256(_b).digest()
            
            # Import main module components
            import json, zlib, struct, os, tempfile, time, platform
            from pathlib import Path
            
            # Unpack container
            _o = 0
            _m = _b[_o:_o+4]
            if _m != b"PY2C": {var1}.exit(1)
            _o += 4
            
            _v = struct.unpack('<I', _b[_o:_o+4])[0]
            _o += 4
            
            _ml = struct.unpack('<I', _b[_o:_o+4])[0]
            _o += 4 + _ml
            
            _sc = struct.unpack('<I', _b[_o:_o+4])[0]
            _o += 4
            
            _s = {{}}
            for _ in range(_sc):
                _nl = struct.unpack('<I', _b[_o:_o+4])[0]
                _o += 4
                _n = _b[_o:_o+_nl].decode()
                _o += _nl
                
                _f = struct.unpack('<I', _b[_o:_o+4])[0]
                _o += 4
                
                _dl = struct.unpack('<I', _b[_o:_o+4])[0]
                _o += 4
                _sd = _b[_o:_o+_dl]
                _o += _dl + 32
                
                # Decrypt FIRST if needed (was encrypted last during packing)
                if _f & 0x02:
                    # Use same key derivation as packer
                    _k = bytes.fromhex("{key_hex}")
                    _k = {var4}.sha256(_k).digest()
                    _sd = bytes(_sd[i] ^ _k[i % len(_k)] for i in range(len(_sd)))
                
                # Decompress SECOND if needed (was compressed first during packing)
                if _f & 0x01:
                    _sd = zlib.decompress(_sd)
                
                _s[_n] = _sd
            
            # Load DLL
            _dll = _s["dll"]
            
            # Load metadata from JSON
            import json
            _meta_dict = json.loads(_s["metadata"].decode())
            
            # Create simple namedtuple-like objects
            class _M:
                def __init__(self, d):
                    self.original_name = d['original_name']
                    self.obfuscated_name = d['obfuscated_name']
                    self.param_count = d['param_count']
                    self.index = d['index']
            
            _meta = [_M(f) for f in _meta_dict['functions']]
            
            # Platform-specific loading
            if {var1}.platform.startswith("linux"):
                suffix = ".so"
            elif {var1}.platform == "darwin":
                suffix = ".dylib"
            else:
                suffix = ".dll"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                _path = f.name
                f.write(_dll)
            
            self._lib = {var2}.CDLL(_path)
            
            # Setup functions
            self._funcs = {{}}
            for m in _meta:
                try:
                    func = getattr(self._lib, m.obfuscated_name)
                    func.restype = {var2}.c_int64
                    func.argtypes = [{var2}.c_int64] * m.param_count
                    self._funcs[m.index] = func
                except:
                    pass
            
            # Cleanup
            try: os.unlink(_path)
            except: pass
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            {var1}.exit(1)
    
    def __call__(self, idx, *args):
        if idx not in self._funcs: return 0
        try: return int(self._funcs[idx](*args))
        except: return 0

_lib = _L()
call = lambda i, *a: _lib(i, *a)

if __name__ == "__main__":
    if len({var1}.argv) < 2:
        print("Usage: {{}} <function_index> [args...]".format({var1}.argv[0]))
        {var1}.exit(1)
    
    idx = int({var1}.argv[1])
    args = [int(x) for x in {var1}.argv[2:]] if len({var1}.argv) > 2 else []
    result = call(idx, *args)
    print(f"Result: {{result}}")
'''
        
        # Phase 7: Apply control-flow obfuscation to entire loader
        if self.security_level.value >= SecurityLevel.PARANOID.value:
            template = py_obf.add_control_flow_obfuscation(template)
        
        return template

    # NEW METHOD - Added for executable module generation
    def to_python_module_executable(self, output_path: str, module_name: str,
                                    original_code: str,
                                    hardware_binding: bool = False):
        """Generate executable module that runs like normal Python"""
        import ast
        
        obf_loader, hw_cpu, hw_mac = self.to_python_code(module_name, hardware_binding)
        tree = ast.parse(original_code)
        
        function_names = []
        executable_statements = []
        
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
            else:
                executable_statements.append(node)
        
        lines = obf_loader.split('\n')
        main_index = -1
        for i, line in enumerate(lines):
            if 'if __name__ == "__main__":' in line:
                main_index = i
                break
        
        if main_index != -1:
            lines = lines[:main_index]
        
        loader_code = '\n'.join(lines)
        
        name_to_index = {}
        for meta in self.func_metadata:
            name_to_index[meta.original_name] = meta.index
        
        wrapper_functions = []
        for func_name in function_names:
            if func_name in name_to_index:
                idx = name_to_index[func_name]
                wrapper = f"""
def {func_name}(*args):
    return call({idx}, *args)
"""
                wrapper_functions.append(wrapper)
        
        if executable_statements:
            exec_tree = ast.Module(body=executable_statements, type_ignores=[])
            exec_code = ast.unparse(exec_tree)
            indented_exec = '\n'.join('    ' + line for line in exec_code.split('\n'))
            main_block = f"""
if __name__ == "__main__":
{indented_exec}
"""
        else:
            main_block = ""
        
        full_code = f"""{loader_code}

{''.join(wrapper_functions)}
{main_block}"""
        
        with open(output_path, 'w') as f:
            f.write(full_code)
        
        if not sys.platform.startswith("win"):
            os.chmod(output_path, 0o755)
        
        print(f"[✓] Generated executable module: {output_path}")
        print(f"[*] Size: {os.path.getsize(output_path):,} bytes")
        print(f"[*] Security: {self.security_level.name}")
        if hardware_binding:
            print(f"[*] Hardware-bound: CPU={hw_cpu[:8]}..., MAC={hw_mac[:8]}...")