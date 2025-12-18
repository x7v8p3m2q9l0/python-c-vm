import os, time, hashlib, secrets
from typing import Dict, Optional
class RandomGenerator:
    @staticmethod
    def seed_from_env() -> bytes:
        """Generate seed from environment"""
        env_data = f"{os.getpid()}{time.time()}{os.urandom(16)}"
        return hashlib.sha256(env_data.encode()).digest()
    
    @staticmethod
    def random_id(length: int = 8) -> str:
        """Generate random identifier"""
        return secrets.token_hex(length // 2)
    
    @staticmethod
    def random_symbol(prefix: str = "fn") -> str:
        """Generate random C symbol name"""
        return f"{prefix}_{secrets.token_hex(16)}"
    
    @staticmethod
    def random_int(min_val: int, max_val: int) -> int:
        """Generate random integer"""
        return secrets.randbelow(max_val - min_val + 1) + min_val

class SymbolTable:
    """Manages symbol name obfuscation"""
    
    def __init__(self, seed: Optional[bytes] = None):
        self.mapping: Dict[str, str] = {}
        self.reverse: Dict[str, str] = {}
        self.seed = seed or RandomGenerator.seed_from_env()
    
    def obfuscate(self, name: str) -> str:
        """Obfuscate a symbol name"""
        if name not in self.mapping:
            # Generate deterministic but random-looking name
            hash_input = f"{self.seed.hex()}{name}".encode()
            hash_val = hashlib.sha256(hash_input).hexdigest()[:16]
            obf_name = f"_{hash_val}"
            self.mapping[name] = obf_name
            self.reverse[obf_name] = name
        return self.mapping[name]
    
    def get_original(self, obfuscated: str) -> Optional[str]:
        """Get original name from obfuscated"""
        return self.reverse.get(obfuscated)