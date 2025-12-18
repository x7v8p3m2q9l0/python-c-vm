import sys
import os
import time
import hashlib
import ctypes
from typing import Optional
class SecurityMonitor:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._initial_hash: Optional[bytes] = None
        self.check_count = 0
        self._code_hashes = {}
        self._environment_baseline = None
        
        # Immediate environment check on initialization
        if self.enabled:
            self._check_environment()
    
    def _check_environment(self):
        """Comprehensive environment analysis on init"""
        # Multiple checks - any failure = exit
        if self.check_debugger():
            self._trigger_exit()
        
        if self._detect_vm():
            self._trigger_exit()
        
        if self._detect_analysis_tools():
            self._trigger_exit()
        
        # Store environment baseline
        self._environment_baseline = self._get_environment_fingerprint()
    
    def check_debugger(self) -> bool:
        """Multi-layer debugger detection"""
        if not self.enabled:
            return False
        
        # Layer 1: Python-level debugging
        if sys.gettrace() is not None:
            return True
        
        # Layer 2: Check for debugger modules
        debugger_modules = {'pdb', 'bdb', 'pydevd', 'debugpy', 'ipdb', 'pudb'}
        if debugger_modules & set(sys.modules.keys()):
            return True
        
        # Layer 3: Timing attack - debuggers slow execution
        try:
            start = time.perf_counter()
            x = sum(i * i for i in range(1000))
            elapsed = time.perf_counter() - start
            if elapsed > 0.01:  # 10ms threshold
                return True
        except:
            pass
        
        # Layer 4: Platform-specific checks
        if sys.platform.startswith("win"):
            if self._check_debugger_windows():
                return True
        elif sys.platform.startswith("linux"):
            if self._check_debugger_linux():
                return True
        
        return False
    
    def _check_debugger_windows(self) -> bool:
        """Windows: Multiple debugger detection methods"""
        try:
            kernel32 = ctypes.windll.kernel32
            
            # Method 1: IsDebuggerPresent
            if kernel32.IsDebuggerPresent() != 0:
                return True
            
            # Method 2: CheckRemoteDebuggerPresent
            is_debugged = ctypes.c_bool()
            kernel32.CheckRemoteDebuggerPresent(
                kernel32.GetCurrentProcess(),
                ctypes.byref(is_debugged)
            )
            if is_debugged.value:
                return True
            
            # Method 3: NtQueryInformationProcess
            try:
                ProcessDebugPort = 7
                ntdll = ctypes.WinDLL('ntdll')
                debug_port = ctypes.c_int()
                ntdll.NtQueryInformationProcess(
                    kernel32.GetCurrentProcess(),
                    ProcessDebugPort,
                    ctypes.byref(debug_port),
                    ctypes.sizeof(debug_port),
                    None
                )
                if debug_port.value != 0:
                    return True
            except:
                pass
            
        except:
            pass
        return False
    
    def _check_debugger_linux(self) -> bool:
        """Linux: Check /proc/self/status for TracerPid"""
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('TracerPid:'):
                        pid = int(line.split(':')[1].strip())
                        return pid != 0
        except:
            pass
        return False
    
    def _detect_vm(self) -> bool:
        """Detect virtual machine / sandbox environment"""
        if not self.enabled:
            return False
        
        vm_indicators = []
        
        # Check 1: Low CPU count (VMs often limited)
        try:
            import os
            if os.cpu_count() and os.cpu_count() < 2:
                vm_indicators.append('cpu')
        except:
            pass
        
        # Check 2: VM-specific strings in platform info
        try:
            import platform
            system_info = platform.platform().lower()
            vm_strings = ['vmware', 'virtualbox', 'qemu', 'xen', 'kvm', 'bochs', 'parallels']
            if any(vm in system_info for vm in vm_strings):
                vm_indicators.append('platform')
        except:
            pass
        
        # Check 3: MAC address analysis (VM vendors use specific ranges)
        try:
            import uuid
            mac = hex(uuid.getnode())[2:].upper().zfill(12)
            # Common VM MAC prefixes
            vm_macs = ['000569', '000C29', '001C14', '005056', '080027', '0A0027']
            if any(mac.startswith(prefix) for prefix in vm_macs):
                vm_indicators.append('mac')
        except:
            pass
        
        # Check 4: Sandbox environment variables
        try:
            import os
            sandbox_vars = ['SANDBOX', 'WINE', 'VM', 'VIRTUAL', 'VBOX', 'VMWARE']
            for var in os.environ:
                if any(s in var.upper() for s in sandbox_vars):
                    vm_indicators.append('env')
                    break
        except:
            pass
        
        # If 2+ indicators, likely VM/sandbox
        return len(vm_indicators) >= 2
    
    def _detect_analysis_tools(self) -> bool:
        """Detect reverse engineering / analysis tools"""
        if not self.enabled:
            return False
        
        # Check 1: Known RE tool modules
        re_modules = {
            'frida', 'r2pipe', 'pwntools', 'capstone', 'keystone',
            'unicorn', 'angr', 'pwndbg', 'gef', 'peda',
            'voltron', 'plasma', 'mcsema'
        }
        if re_modules & set(sys.modules.keys()):
            return True
        
        # Check 2: Process name analysis
        try:
            import os
            process_name = os.path.basename(sys.argv[0]).lower()
            tool_names = [
                'ida', 'ida64', 'ghidra', 'x64dbg', 'x32dbg', 'ollydbg',
                'windbg', 'gdb', 'lldb', 'radare2', 'r2', 'cutter',
                'binaryninja', 'hopper', 'immunity'
            ]
            if any(tool in process_name for tool in tool_names):
                return True
        except:
            pass
        
        # Check 3: Suspicious loaded libraries (on Linux)
        if sys.platform.startswith('linux'):
            try:
                with open('/proc/self/maps', 'r') as f:
                    maps = f.read().lower()
                    suspicious = ['frida', 'inject', 'hook', 'preload']
                    if any(sus in maps for sus in suspicious):
                        return True
            except:
                pass
        
        return False
    
    def _get_environment_fingerprint(self) -> str:
        """Create fingerprint of current environment"""
        try:
            import platform
            import os
            
            parts = [
                platform.python_version(),
                platform.platform(),
                str(os.cpu_count()),
                str(len(sys.modules))
            ]
            return '|'.join(parts)
        except:
            return "unknown"
    
    def verify_integrity(self, data: bytes, expected_hash: Optional[bytes] = None) -> bool:
        """Verify data integrity with anti-tamper"""
        if not self.enabled:
            return True
        
        current_hash = hashlib.sha256(data).digest()
        
        if expected_hash is None:
            # First check - store hash
            if self._initial_hash is None:
                self._initial_hash = current_hash
            return True
        
        # Verify against expected hash
        if current_hash != expected_hash:
            return False
        
        # Also verify against initial hash (detect runtime modification)
        if self._initial_hash and current_hash != self._initial_hash:
            return False
        
        return True
    
    def anti_tamper_check(self):
        """Comprehensive anti-tamper checks"""
        if not self.enabled:
            return
        
        self.check_count += 1
        
        # Every call: Check for debugger
        if self.check_debugger():
            self._trigger_exit()
        
        # Every 100 calls: Re-check environment
        if self.check_count % 100 == 0:
            if self._detect_vm() or self._detect_analysis_tools():
                self._trigger_exit()
            
            # Check if environment changed
            current_fp = self._get_environment_fingerprint()
            if self._environment_baseline and current_fp != self._environment_baseline:
                self._trigger_exit()
        
        # Every 50 calls: Code integrity check
        if self.check_count % 50 == 0:
            self._check_code_integrity()
    
    def _check_code_integrity(self):
        """Verify our own bytecode hasn't been modified"""
        try:
            import inspect
            
            # Get current frame's code object
            frame = inspect.currentframe()
            if frame and frame.f_code:
                code_hash = hashlib.sha256(frame.f_code.co_code).digest()
                
                # Store first hash
                if 'frame' not in self._code_hashes:
                    self._code_hashes['frame'] = code_hash
                elif code_hash != self._code_hashes['frame']:
                    # Code modified!
                    self._trigger_exit()
        except:
            pass
    
    def _trigger_exit(self):
        """Exit silently - make it look like normal operation"""
        # Strategy: Mix of silent exits and fake operation
        # This makes analysis harder as behavior is unpredictable
        
        import random
        choice = random.randint(0, 2)
        
        if choice == 0:
            # Silent immediate exit
            os._exit(0)
        elif choice == 1:
            # Delayed exit with fake work
            time.sleep(random.uniform(0.01, 0.05))
            os._exit(0)
        else:
            # Return fake data (let caller handle with 0 return)
            return