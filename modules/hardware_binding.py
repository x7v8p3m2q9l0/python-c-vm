import hashlib
import sys
import time
from typing import Optional
class HardwareBinding:
    @staticmethod
    def get_cpu_id() -> str:
        """Get CPU identifier"""
        try:
            if sys.platform == "linux":
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'processor' in line:
                            return hashlib.sha256(line.encode()).hexdigest()[:16]
        except:
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

