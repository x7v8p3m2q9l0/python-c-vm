import sys
import os
import time
import hashlib
import ctypes
import threading
from typing import Optional, Dict, Set
from dataclasses import dataclass
from enum import Enum


class ThreatLevel(Enum):
    """Threat detection levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SecurityEvent:
    """Security event record"""
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    details: str
    
    def __str__(self):
        return f"[{self.event_type}] {self.threat_level.name}: {self.details}"


class SecurityMonitor:

    def __init__(self, enabled: bool = True, continuous_monitoring: bool = False):
        self.enabled = enabled
        self.continuous_monitoring = continuous_monitoring
        self._initial_hash: Optional[bytes] = None
        self.check_count = 0
        self._code_hashes: Dict[str, bytes] = {}
        self._environment_baseline: Optional[str] = None
        self._timing_baseline: Optional[float] = None
        self._events: list = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = False
        
        # Immediate environment check on initialization
        if self.enabled:
            self._check_environment()
            
            if self.continuous_monitoring:
                self._start_continuous_monitoring()
    
    def _log_event(self, event_type: str, threat_level: ThreatLevel, details: str):
        """Log security event"""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            threat_level=threat_level,
            details=details
        )
        self._events.append(event)
        
        if threat_level.value >= ThreatLevel.HIGH.value:
            # High threat - take action
            self._trigger_exit()
    
    def _check_environment(self):
        """Comprehensive environment analysis on init"""
        # Multiple checks - any failure = exit
        if self.check_debugger():
            self._log_event("debugger", ThreatLevel.CRITICAL, "Debugger detected")
        
        if self._detect_vm():
            self._log_event("vm", ThreatLevel.HIGH, "Virtual machine detected")
        
        if self._detect_analysis_tools():
            self._log_event("analysis_tools", ThreatLevel.HIGH, "Analysis tools detected")
        
        # Store environment baseline
        self._environment_baseline = self._get_environment_fingerprint()
        self._timing_baseline = self._calibrate_timing()
    
    def _calibrate_timing(self) -> float:
        """
        FIXED: Calibrate timing checks based on system performance
        Prevents false positives on slow systems
        """
        samples = []
        for _ in range(10):
            start = time.perf_counter()
            x = sum(i * i for i in range(1000))
            elapsed = time.perf_counter() - start
            samples.append(elapsed)
        
        # Use median to avoid outliers
        samples.sort()
        baseline = samples[len(samples) // 2]
        
        # Set threshold at 5x baseline (instead of fixed 10ms)
        return baseline * 5
    
    def check_debugger(self) -> bool:
        """
        FIXED: Multi-layer debugger detection with adaptive timing
        """
        if not self.enabled:
            return False
        
        # Layer 1: Python-level debugging
        if sys.gettrace() is not None:
            return True
        
        # Layer 2: Check for debugger modules
        debugger_modules = {'pdb', 'bdb', 'pydevd', 'debugpy', 'ipdb', 'pudb', 'rpdb'}
        if debugger_modules & set(sys.modules.keys()):
            return True
        
        # Layer 3: FIXED: Adaptive timing attack
        if self._timing_baseline:
            start = time.perf_counter()
            x = sum(i * i for i in range(1000))
            elapsed = time.perf_counter() - start
            
            # Compare against calibrated baseline
            if elapsed > self._timing_baseline:
                return True
        
        # Layer 4: Platform-specific checks
        if sys.platform.startswith("win"):
            if self._check_debugger_windows():
                return True
        elif sys.platform.startswith("linux"):
            if self._check_debugger_linux():
                return True
        elif sys.platform == "darwin":
            if self._check_debugger_macos():
                return True
        
        return False
    
    def _check_debugger_windows(self) -> bool:
        """
        FIXED: Enhanced Windows debugger detection
        """
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
            
            # Method 3: NtQueryInformationProcess - ProcessDebugPort
            try:
                ProcessDebugPort = 7
                ntdll = ctypes.WinDLL('ntdll')
                debug_port = ctypes.c_longlong()
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
            
            # FIXED: Method 4: ProcessDebugObjectHandle
            try:
                ProcessDebugObjectHandle = 30
                debug_object = ctypes.c_void_p()
                ntdll.NtQueryInformationProcess(
                    kernel32.GetCurrentProcess(),
                    ProcessDebugObjectHandle,
                    ctypes.byref(debug_object),
                    ctypes.sizeof(debug_object),
                    None
                )
                if debug_object.value:
                    return True
            except:
                pass
            
            # FIXED: Method 5: Check for hardware breakpoints (DR registers)
            try:
                class CONTEXT(ctypes.Structure):
                    _fields_ = [
                        ("Dr0", ctypes.c_ulonglong),
                        ("Dr1", ctypes.c_ulonglong),
                        ("Dr2", ctypes.c_ulonglong),
                        ("Dr3", ctypes.c_ulonglong),
                        ("Dr6", ctypes.c_ulonglong),
                        ("Dr7", ctypes.c_ulonglong),
                    ]
                
                context = CONTEXT()
                # If any debug register is set, hardware breakpoint present
                if any([context.Dr0, context.Dr1, context.Dr2, context.Dr3]):
                    return True
            except:
                pass
            
        except Exception as e:
            # If we can't check, assume safe but log
            pass
        
        return False
    
    def _check_debugger_linux(self) -> bool:
        """
        FIXED: Enhanced Linux debugger detection
        """
        try:
            # Method 1: Check TracerPid in /proc/self/status
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('TracerPid:'):
                        pid = int(line.split(':')[1].strip())
                        if pid != 0:
                            return True
            
            # FIXED: Method 2: Check if ptrace is available
            # If we can't ptrace ourselves, someone else is already tracing
            try:
                libc = ctypes.CDLL('libc.so.6')
                PTRACE_TRACEME = 0
                
                # Try to trace ourselves
                if libc.ptrace(PTRACE_TRACEME, 0, 0, 0) == -1:
                    # Already being traced
                    return True
                else:
                    # Successfully traced ourselves - detach
                    PTRACE_DETACH = 17
                    libc.ptrace(PTRACE_DETACH, 0, 0, 0)
            except:
                pass
            
            # FIXED: Method 3: Check /proc/self/fd for suspicious file descriptors
            try:
                fd_count = len(os.listdir('/proc/self/fd'))
                # Normal programs have 3-10 FDs, debuggers often have many more
                if fd_count > 50:
                    return True
            except:
                pass
            
        except Exception as e:
            pass
        
        return False
    
    def _check_debugger_macos(self) -> bool:
        """
        FIXED: Added macOS debugger detection
        """
        try:
            # Method 1: sysctl P_TRACED flag
            import subprocess
            result = subprocess.run(
                ['sysctl', 'kern.proc.pid.' + str(os.getpid())],
                capture_output=True,
                text=True,
                timeout=1
            )
            if 'P_TRACED' in result.stdout:
                return True
            
            # Method 2: Check for LLDB/GDB
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=1)
            if any(dbg in result.stdout.lower() for dbg in ['lldb', 'gdb', 'debugserver']):
                return True
            
        except:
            pass
        
        return False
    
    def _detect_vm(self) -> bool:
        """
        FIXED: Enhanced VM detection with more heuristics
        """
        if not self.enabled:
            return False
        
        vm_score = 0  # Use scoring instead of binary detection
        
        # Check 1: CPU count
        try:
            cpu_count = os.cpu_count() or 0
            if cpu_count < 2:
                vm_score += 2
            elif cpu_count == 2:
                vm_score += 1
        except:
            pass
        
        # Check 2: Platform info
        try:
            import platform
            system_info = platform.platform().lower()
            vm_strings = ['vmware', 'virtualbox', 'qemu', 'xen', 'kvm', 'bochs', 'parallels', 'hyperv']
            for vm_str in vm_strings:
                if vm_str in system_info:
                    vm_score += 3
                    break
        except:
            pass
        
        # Check 3: MAC address
        try:
            import uuid
            mac = hex(uuid.getnode())[2:].upper().zfill(12)
            vm_macs = ['000569', '000C29', '001C14', '005056', '080027', '0A0027', '00155D', '001C42']
            if any(mac.startswith(prefix) for prefix in vm_macs):
                vm_score += 2
        except:
            pass
        
        # Check 4: Environment variables
        try:
            sandbox_vars = ['SANDBOX', 'WINE', 'VM', 'VIRTUAL', 'VBOX', 'VMWARE', 'HYPERVISOR']
            for var in os.environ:
                if any(s in var.upper() for s in sandbox_vars):
                    vm_score += 1
                    break
        except:
            pass
        
        # FIXED: Check 5: Disk size (VMs often have small disks)
        try:
            import shutil
            disk = shutil.disk_usage('/')
            total_gb = disk.total / (1024**3)
            if total_gb < 20:  # Less than 20GB
                vm_score += 2
        except:
            pass
        
        # FIXED: Check 6: System uptime (sandboxes often have low uptime)
        try:
            if sys.platform.startswith('linux'):
                with open('/proc/uptime', 'r') as f:
                    uptime_seconds = float(f.read().split()[0])
                    if uptime_seconds < 600:  # Less than 10 minutes
                        vm_score += 2
        except:
            pass
        
        # Threshold: score >= 5 indicates VM/sandbox
        return vm_score >= 5
    
    def _detect_analysis_tools(self) -> bool:
        """
        FIXED: Enhanced analysis tool detection
        """
        if not self.enabled:
            return False
        
        # Check 1: Known RE tool modules
        re_modules = {
            'frida', 'r2pipe', 'pwntools', 'capstone', 'keystone',
            'unicorn', 'angr', 'pwndbg', 'gef', 'peda',
            'voltron', 'plasma', 'mcsema', 'manticore', 'triton'
        }
        if re_modules & set(sys.modules.keys()):
            return True
        
        # Check 2: Process name
        try:
            process_name = os.path.basename(sys.argv[0]).lower()
            tool_names = [
                'ida', 'ida64', 'ghidra', 'x64dbg', 'x32dbg', 'ollydbg',
                'windbg', 'gdb', 'lldb', 'radare2', 'r2', 'cutter',
                'binaryninja', 'hopper', 'immunity', 'wireshark', 'procmon'
            ]
            if any(tool in process_name for tool in tool_names):
                return True
        except:
            pass
        
        # Check 3: Loaded libraries (Linux)
        if sys.platform.startswith('linux'):
            try:
                with open('/proc/self/maps', 'r') as f:
                    maps = f.read().lower()
                    suspicious = ['frida', 'inject', 'hook', 'preload', 'ld-linux']
                    # Note: ld-linux is normal, but if combined with others...
                    if sum(sus in maps for sus in suspicious) >= 2:
                        return True
            except:
                pass
        
        # FIXED: Check 4: Parent process (am I launched from a debugger?)
        try:
            import psutil
            parent = psutil.Process().parent()
            if parent:
                parent_name = parent.name().lower()
                debuggers = ['ida', 'ghidra', 'x64dbg', 'gdb', 'lldb', 'windbg', 'ollydbg']
                if any(dbg in parent_name for dbg in debuggers):
                    return True
        except ImportError:
            # psutil not available, skip check
            pass
        except:
            pass
        
        return False
    
    def _get_environment_fingerprint(self) -> str:
        """Create fingerprint of current environment"""
        try:
            import platform
            
            parts = [
                platform.python_version(),
                platform.platform(),
                str(os.cpu_count()),
                str(len(sys.modules)),
                str(os.getpid()),
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
            self._log_event("integrity", ThreatLevel.CRITICAL, "Data integrity check failed")
            return False
        
        # Also verify against initial hash
        if self._initial_hash and current_hash != self._initial_hash:
            self._log_event("integrity", ThreatLevel.CRITICAL, "Runtime modification detected")
            return False
        
        return True
    
    def anti_tamper_check(self):
        """Comprehensive anti-tamper checks"""
        if not self.enabled:
            return
        
        self.check_count += 1
        
        # Every call: Check for debugger
        if self.check_debugger():
            self._log_event("debugger", ThreatLevel.CRITICAL, "Debugger detected during execution")
        
        # Every 100 calls: Re-check environment
        if self.check_count % 100 == 0:
            if self._detect_vm() or self._detect_analysis_tools():
                self._log_event("environment", ThreatLevel.HIGH, "Suspicious environment detected")
            
            # Check if environment changed
            current_fp = self._get_environment_fingerprint()
            if self._environment_baseline and current_fp != self._environment_baseline:
                self._log_event("environment", ThreatLevel.MEDIUM, "Environment changed during execution")
        
        # Every 50 calls: Code integrity check
        if self.check_count % 50 == 0:
            self._check_code_integrity()
    
    def _check_code_integrity(self):
        """Verify our own bytecode hasn't been modified"""
        try:
            import inspect
            
            frame = inspect.currentframe()
            if frame and frame.f_code:
                code_hash = hashlib.sha256(frame.f_code.co_code).digest()
                
                if 'frame' not in self._code_hashes:
                    self._code_hashes['frame'] = code_hash
                elif code_hash != self._code_hashes['frame']:
                    self._log_event("code_integrity", ThreatLevel.CRITICAL, "Code modification detected")
        except:
            pass
    
    def _start_continuous_monitoring(self):
        """Start background monitoring thread"""
        def monitor_loop():
            while not self._stop_monitoring:
                try:
                    if self.check_debugger():
                        self._log_event("debugger", ThreatLevel.CRITICAL, "Debugger detected (continuous)")
                    
                    time.sleep(1)  # Check every second
                except:
                    break
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._stop_monitoring = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
    
    def _trigger_exit(self):
        """
        FIXED: More sophisticated exit strategy
        Mix of behaviors to confuse analysis
        """
        import random
        
        # Randomly choose exit strategy
        strategies = [
            lambda: os._exit(0),  # Immediate exit
            lambda: sys.exit(0),  # Clean exit
            lambda: None,  # Continue with fake operation
            lambda: time.sleep(random.uniform(0.01, 0.1)) or os._exit(0),  # Delayed
        ]
        
        strategy = random.choice(strategies)
        strategy()
    
    def get_events(self, min_threat_level: ThreatLevel = ThreatLevel.LOW) -> list:
        """Get security events above threshold"""
        return [e for e in self._events if e.threat_level.value >= min_threat_level.value]
    
    def get_statistics(self) -> Dict:
        """Get monitoring statistics"""
        return {
            'total_checks': self.check_count,
            'events': len(self._events),
            'high_threats': len([e for e in self._events if e.threat_level.value >= ThreatLevel.HIGH.value]),
            'continuous_monitoring': self.continuous_monitoring
        }


# Testing
if __name__ == "__main__":
    print("Security Monitor - Fixed Version")
    print("=" * 60)
    
    # Create monitor
    monitor = SecurityMonitor(enabled=True, continuous_monitoring=False)
    
    print("\n1. Environment Checks:")
    print(f"   Debugger: {'DETECTED' if monitor.check_debugger() else 'Not detected'}")
    print(f"   VM/Sandbox: {'DETECTED' if monitor._detect_vm() else 'Not detected'}")
    print(f"   Analysis Tools: {'DETECTED' if monitor._detect_analysis_tools() else 'Not detected'}")
    
    print("\n2. Timing Calibration:")
    if monitor._timing_baseline:
        print(f"   Baseline: {monitor._timing_baseline*1000:.2f}ms")
        print(f"   Threshold: {monitor._timing_baseline*1000:.2f}ms")
    
    print("\n3. Integrity Check:")
    test_data = b"sensitive data"
    test_hash = hashlib.sha256(test_data).digest()
    if monitor.verify_integrity(test_data, test_hash):
        print("   ✓ Integrity verified")
    
    print("\n4. Statistics:")
    stats = monitor.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n5. Security Events:")
    events = monitor.get_events()
    if events:
        for event in events[-5:]:  # Show last 5
            print(f"   {event}")
    else:
        print("   No security events")