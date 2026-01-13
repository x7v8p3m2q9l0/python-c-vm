from typing import List, Dict, Optional
import os
import struct
import sys
import platform
import time
import zlib
import hashlib
import secrets

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("[WARNING] cryptography library not available. Install with: pip install cryptography")

from base import VERSION, MAGIC_HEADER, CONTAINER_VERSION, IntegrityError
from type import BinarySection
from utils import RandomGenerator


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
        
        # Ensure key is 32 bytes for AES-256
        if len(self.key) != 32:
            self.key = hashlib.sha256(self.key).digest()
    
    def _derive_key(self) -> bytes:
        """
        FIXED: More robust key derivation using PBKDF2
        """
        # Collect entropy sources
        env_factors = [
            str(os.getpid()).encode(),
            platform.node().encode(),
            platform.processor().encode(),
            str(time.time()).encode(),
            secrets.token_bytes(32),  # High-quality random bytes
        ]
        
        # Combine entropy
        key_material = b''.join(env_factors)
        
        # Use PBKDF2 for proper key derivation
        if CRYPTO_AVAILABLE:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            
            # Use a fixed salt for deterministic key generation
            # In production, you might want to store this salt
            salt = hashlib.sha256(b"pysec_container_v2").digest()
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # 256-bit key for AES-256
                salt=salt,
                iterations=100000,  # Standard recommendation
            )
            return kdf.derive(key_material)
        else:
            # Fallback to double SHA256
            return hashlib.sha256(hashlib.sha256(key_material).digest()).digest()
    
    def add_section(self, name: str, data: bytes, compress: bool = True, 
                   encrypt: bool = True):
        """Add section to container"""
        processed_data = data
        
        # Compress FIRST if requested
        if compress:
            processed_data = zlib.compress(processed_data, level=9)
        
        # Encrypt SECOND if requested
        if encrypt:
            processed_data = self._encrypt_aesgcm(processed_data)
        
        section = BinarySection(name, processed_data, compress, encrypt)
        self.sections.append(section)
    
    def _encrypt_aesgcm(self, data: bytes) -> bytes:
        """
        FIXED: Use AES-GCM authenticated encryption
        Provides both confidentiality and integrity
        """
        if not CRYPTO_AVAILABLE:
            print("[WARNING] Using fallback XOR encryption - not secure!")
            return self._encrypt_xor_fallback(data)
        
        try:
            # Generate random 96-bit nonce (12 bytes) for GCM
            nonce = secrets.token_bytes(12)
            
            # Create AESGCM cipher
            aesgcm = AESGCM(self.key)
            
            # Encrypt with authentication
            # associated_data=None means we're not using additional authenticated data
            ciphertext = aesgcm.encrypt(nonce, data, None)
            
            # Prepend nonce to ciphertext (needed for decryption)
            return nonce + ciphertext
            
        except Exception as e:
            raise IntegrityError(f"Encryption failed: {e}")
    
    def _decrypt_aesgcm(self, data: bytes) -> bytes:
        """
        FIXED: Decrypt using AES-GCM
        Automatically verifies authentication tag
        """
        if not CRYPTO_AVAILABLE:
            print("[WARNING] Using fallback XOR decryption - not secure!")
            return self._decrypt_xor_fallback(data)
        
        try:
            if len(data) < 12:
                raise IntegrityError("Invalid encrypted data - too short")
            
            # Extract nonce and ciphertext
            nonce = data[:12]
            ciphertext = data[12:]
            
            # Create AESGCM cipher
            aesgcm = AESGCM(self.key)
            
            # Decrypt and verify authentication tag
            # Will raise exception if data has been tampered with
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            
            return plaintext
            
        except Exception as e:
            raise IntegrityError(f"Decryption or authentication failed: {e}")
    
    def _encrypt_xor_fallback(self, data: bytes) -> bytes:
        """
        Fallback XOR encryption (NOT SECURE - only for when cryptography unavailable)
        Better than nothing but should not be used in production
        """
        encrypted = bytearray()
        nonce = secrets.token_bytes(16)
        encrypted.extend(nonce)
        
        for i, byte in enumerate(data):
            # Extend key by hashing
            extended_key = hashlib.sha256(
                self.key + nonce + i.to_bytes(4, 'little')
            ).digest()
            key_byte = extended_key[0]
            encrypted.append(byte ^ key_byte)
        
        return bytes(encrypted)
    
    def _decrypt_xor_fallback(self, data: bytes) -> bytes:
        """Fallback XOR decryption"""
        if len(data) < 16:
            raise IntegrityError("Invalid encrypted data")
        
        nonce = data[:16]
        ciphertext = data[16:]
        
        decrypted = bytearray()
        for i, byte in enumerate(ciphertext):
            extended_key = hashlib.sha256(
                self.key + nonce + i.to_bytes(4, 'little')
            ).digest()
            key_byte = extended_key[0]
            decrypted.append(byte ^ key_byte)
        
        return bytes(decrypted)
    
    # Backward compatibility aliases
    def _encrypt(self, data: bytes) -> bytes:
        """Legacy method - redirects to AES-GCM"""
        return self._encrypt_aesgcm(data)
    
    def _decrypt(self, data: bytes) -> bytes:
        """Legacy method - redirects to AES-GCM"""
        return self._decrypt_aesgcm(data)
    
    def pack(self) -> bytes:
        """
        FIXED: Pack container with HMAC for additional integrity verification
        """
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
            
            # Section checksum (SHA256)
            checksum = hashlib.sha256(section.data).digest()
            buffer.extend(checksum)
        
        # FIXED: Global HMAC instead of simple hash
        # This provides authentication that the container was created with the correct key
        import hmac
        global_mac = hmac.new(self.key, bytes(buffer), hashlib.sha256).digest()
        buffer.extend(global_mac)
        
        # Add random padding for anti-fingerprinting
        padding_size = RandomGenerator.random_int(64, 256)
        buffer.extend(secrets.token_bytes(padding_size))
        
        return bytes(buffer)
    
    def unpack(self, data: bytes) -> Dict[str, bytes]:
        """
        FIXED: Unpack with HMAC verification
        """
        # First, verify global HMAC (last 32 bytes before padding)
        # We need to find where the HMAC starts
        # The padding is variable length, so we work backwards
        
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
        
        sections_data = []
        
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
            
            sections_data.append((name, section_data, compressed, encrypted))
        
        # Verify global HMAC
        stored_hmac = data[offset:offset + 32]
        import hmac
        computed_hmac = hmac.new(self.key, data[:offset], hashlib.sha256).digest()
        
        if stored_hmac != computed_hmac:
            raise IntegrityError("Container HMAC verification failed - data may be tampered")
        
        # Process sections
        result = {}
        for name, section_data, compressed, encrypted in sections_data:
            # Decrypt if needed
            if encrypted:
                section_data = self._decrypt_aesgcm(section_data)
            
            # Decompress if needed
            if compressed:
                section_data = zlib.decompress(section_data)
            
            result[name] = section_data
        
        return result


# Example usage and testing
if __name__ == "__main__":
    print("Binary Container - Fixed Version")
    print("=" * 50)
    
    # Test encryption availability
    if CRYPTO_AVAILABLE:
        print("✓ AES-GCM encryption available")
    else:
        print("✗ AES-GCM not available (install: pip install cryptography)")
    
    # Create container
    container = BinaryContainer()
    
    # Add test data
    test_data = b"This is sensitive data that should be protected"
    container.add_section("test", test_data, compress=True, encrypt=True)
    
    # Pack
    packed = container.pack()
    print(f"✓ Packed container: {len(packed)} bytes")
    
    # Unpack
    container2 = BinaryContainer(key=container.key)
    unpacked = container2.unpack(packed)
    
    # Verify
    if unpacked["test"] == test_data:
        print("✓ Data integrity verified")
    else:
        print("✗ Data corruption detected!")
    
    # Test tampering detection
    print("\nTesting tampering detection...")
    try:
        tampered = bytearray(packed)
        tampered[100] ^= 0xFF  # Flip a bit
        container3 = BinaryContainer(key=container.key)
        container3.unpack(bytes(tampered))
        print("✗ Tampering NOT detected - FAIL")
    except IntegrityError as e:
        print(f"✓ Tampering detected: {e}")