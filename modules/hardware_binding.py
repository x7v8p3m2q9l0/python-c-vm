import hashlib
import sys
import time
import subprocess
from typing import Optional

class HardwareBinding:
    @staticmethod
    def get_cpu_id() -> str:
        """Get stable CPU identifier - IMPROVED"""
        try:
            if sys.platform == "linux":
                # Use multiple CPU features for stability
                identifiers = []
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        for line in f:
                            # Use stable features
                            if any(key in line for key in ['model name', 'cpu family', 'stepping', 'microcode']):
                                identifiers.append(line.strip())
                except:
                    pass
                
                if identifiers:
                    combined = ''.join(sorted(identifiers))
                    return hashlib.sha256(combined.encode()).hexdigest()[:16]
            
            elif sys.platform == "darwin":
                try:
                    result = subprocess.run(
                        ['sysctl', '-n', 'machdep.cpu.brand_string'],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0 and result.stdout:
                        return hashlib.sha256(result.stdout.encode()).hexdigest()[:16]
                except:
                    pass
            
            elif sys.platform.startswith("win"):
                try:
                    result = subprocess.run(
                        ['wmic', 'cpu', 'get', 'ProcessorId'],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0 and result.stdout:
                        # Skip header line
                        proc_id = result.stdout.strip().split('\n')[-1].strip()
                        if proc_id:
                            return hashlib.sha256(proc_id.encode()).hexdigest()[:16]
                except:
                    pass
        
        except Exception:
            pass
        
        return "generic_cpu"
    
    @staticmethod
    def get_mac_address() -> str:
        """Get MAC address for binding"""
        try:
            import uuid
            mac = uuid.getnode()
            return hashlib.sha256(str(mac).encode()).hexdigest()[:16]
        except:
            return "generic_mac"
    
    @staticmethod
    def generate_binding_key(bind_cpu: bool = False, bind_mac: bool = False) -> bytes:
        """Generate hardware-bound encryption key"""
        components = [str(time.time())]
        
        if bind_cpu:
            components.append(HardwareBinding.get_cpu_id())
        
        if bind_mac:
            components.append(HardwareBinding.get_mac_address())
        
        key_material = ''.join(components).encode()
        return hashlib.sha256(key_material).digest()
    
    @staticmethod
    def verify_hardware(expected_cpu: Optional[str] = None,
                       expected_mac: Optional[str] = None) -> bool:
        """Verify hardware matches expected values"""
        if expected_cpu and HardwareBinding.get_cpu_id() != expected_cpu:
            return False
        
        if expected_mac and HardwareBinding.get_mac_address() != expected_mac:
            return False
        
        return True
