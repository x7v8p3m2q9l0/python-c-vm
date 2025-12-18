from typing import List, Dict, Optional
from .base import VERSION, MAGIC_HEADER, CONTAINER_VERSION, IntegrityError
from .type import BinarySection
from .utils import RandomGenerator
import os
import struct
import sys
import platform
import time
import zlib
import hashlib

class BinaryContainer:
    def __init__(self, key: Optional[bytes] = None):
        self.sections: List[BinarySection] = []
        self.key = key or self._derive_key()
        self.metadata = {
            'platform': sys.platform,
            'arch': platform.machine(),
            'version': VERSION,
            'timestamp': int(time.time()),
        }
    
    def _derive_key(self) -> bytes:
        """Derive encryption key from environment"""
        # Environment-based key derivation
        env_factors = [
            str(os.getpid()),
            platform.node(),
            platform.processor(),
            str(time.time()),
        ]
        key_material = ''.join(env_factors).encode()
        return hashlib.sha256(key_material).digest()
    
    def add_section(self, name: str, data: bytes, compress: bool = True, 
                   encrypt: bool = True):
        """Add section to container"""
        processed_data = data
        
        # Compress FIRST if requested (compresses better before encryption)
        if compress:
            processed_data = zlib.compress(processed_data, level=9)
        
        # Encrypt SECOND if requested (after compression)
        if encrypt:
            processed_data = self._encrypt(processed_data)
        
        section = BinarySection(name, processed_data, compress, encrypt)
        self.sections.append(section)
    
    def _encrypt(self, data: bytes) -> bytes:
        """Encrypt data using XOR with key stream (simple but effective)"""
        key_stream = hashlib.sha256(self.key).digest()
        encrypted = bytearray()
        
        for i, byte in enumerate(data):
            key_byte = key_stream[i % len(key_stream)]
            encrypted.append(byte ^ key_byte)
        
        return bytes(encrypted)
    
    def _decrypt(self, data: bytes) -> bytes:
        """Decrypt data (XOR is symmetric)"""
        return self._encrypt(data)  # XOR decryption is same as encryption
    
    def pack(self) -> bytes:
        """Pack container to bytes"""
        buffer = bytearray()
        
        # Header
        buffer.extend(MAGIC_HEADER)
        buffer.extend(struct.pack('<I', CONTAINER_VERSION))
        
        # Metadata
        metadata_bytes = str(self.metadata).encode()
        buffer.extend(struct.pack('<I', len(metadata_bytes)))
        buffer.extend(metadata_bytes)
        
        # Sections
        buffer.extend(struct.pack('<I', len(self.sections)))
        
        for section in self.sections:
            # Section header
            name_bytes = section.name.encode()
            buffer.extend(struct.pack('<I', len(name_bytes)))
            buffer.extend(name_bytes)
            
            # Flags
            flags = 0
            if section.compressed:
                flags |= 0x01
            if section.encrypted:
                flags |= 0x02
            buffer.extend(struct.pack('<I', flags))
            
            # Data
            buffer.extend(struct.pack('<I', len(section.data)))
            buffer.extend(section.data)
            
            # Section checksum
            checksum = hashlib.sha256(section.data).digest()
            buffer.extend(checksum)
        
        # Global integrity hash
        global_hash = hashlib.sha256(bytes(buffer)).digest()
        buffer.extend(global_hash)
        
        # Add random padding for anti-fingerprinting
        padding_size = RandomGenerator.random_int(64, 256)
        buffer.extend(os.urandom(padding_size))
        
        return bytes(buffer)
    
    def unpack(self, data: bytes) -> Dict[str, bytes]:
        """Unpack container from bytes"""
        offset = 0
        
        # Verify header
        magic = data[offset:offset + 4]
        offset += 4
        if magic != MAGIC_HEADER:
            raise IntegrityError("Invalid magic header")
        
        version = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        
        # Read metadata
        meta_len = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        metadata_bytes = data[offset:offset + meta_len]
        offset += meta_len
        
        # Read sections
        section_count = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        
        result = {}
        
        for _ in range(section_count):
            # Section name
            name_len = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4
            name = data[offset:offset + name_len].decode()
            offset += name_len
            
            # Flags
            flags = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4
            compressed = bool(flags & 0x01)
            encrypted = bool(flags & 0x02)
            
            # Data
            data_len = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4
            section_data = data[offset:offset + data_len]
            offset += data_len
            
            # Verify checksum
            stored_checksum = data[offset:offset + 32]
            offset += 32
            actual_checksum = hashlib.sha256(section_data).digest()
            
            if stored_checksum != actual_checksum:
                raise IntegrityError(f"Section checksum mismatch: {name}")
            
            # Decrypt if needed
            if encrypted:
                section_data = self._decrypt(section_data)
            
            # Decompress if needed
            if compressed:
                section_data = zlib.decompress(section_data)
            
            result[name] = section_data
        
        return result
