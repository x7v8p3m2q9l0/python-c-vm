import hashlib
import platform
import subprocess
import uuid
import sys
from typing import Optional

class HardwareBinding:

    # ---------- Low-level identifiers ----------
    @staticmethod
    def _get_tpm_id() -> str:
        try:
            if sys.platform.startswith("win"):
                out = subprocess.check_output(
                    'wmic /namespace:\\\\root\\cimv2\\security\\microsofttpm path win32_tpm get ManufacturerId,ManufacturerVersion,SpecVersion',
                    shell=True
                ).decode()
                return out.strip().replace("\n", "").replace(" ", "")

            elif sys.platform == "linux":
                out = subprocess.check_output(
                    "cat /sys/class/tpm/tpm0/device/description",
                    shell=True
                ).decode().strip()
                return out

            elif sys.platform == "darwin":
                out = subprocess.check_output(
                    ["system_profiler", "SPiBridgeDataType"]
                ).decode()
                return out.strip()

        except:
            pass

        return "no_tpm"
    @staticmethod
    def _get_machine_id() -> str:
        try:
            if sys.platform.startswith("win"):
                out = subprocess.check_output(
                    "wmic csproduct get uuid", shell=True
                ).decode().splitlines()
                return out[1].strip()
            elif sys.platform == "darwin":
                out = subprocess.check_output(
                    ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"]
                ).decode()
                for line in out.splitlines():
                    if "IOPlatformUUID" in line:
                        return line.split('"')[-2]
            else:
                with open("/etc/machine-id") as f:
                    return f.read().strip()
        except:
            return "unknown_machine"

    @staticmethod
    def _get_disk_id() -> str:
        try:
            if sys.platform.startswith("win"):
                out = subprocess.check_output(
                    "wmic diskdrive get serialnumber", shell=True
                ).decode().splitlines()
                return out[1].strip()
            elif sys.platform == "darwin":
                out = subprocess.check_output(
                    ["system_profiler", "SPStorageDataType"]
                ).decode()
                for line in out.splitlines():
                    if "Serial Number" in line:
                        return line.split(":")[-1].strip()
        except:
            pass
        return "unknown_disk"

    @staticmethod
    def _get_cpu_signature() -> str:
        return f"{platform.processor()}|{platform.machine()}"

    @staticmethod
    def _get_mac() -> str:
        try:
            return str(uuid.getnode())
        except:
            return "unknown_mac"

    # ---------- Composite Fingerprint ----------

    @staticmethod
    def get_fingerprint() -> str:
        parts = [
            HardwareBinding._get_machine_id(),
            HardwareBinding._get_disk_id(),
            HardwareBinding._get_cpu_signature(),
            HardwareBinding._get_mac(),
            HardwareBinding._get_tpm_id(),
            platform.node(),
            platform.system()
        ]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ---------- Public API ----------

    @staticmethod
    def generate_binding_key() -> bytes:
        fingerprint = HardwareBinding.get_fingerprint()
        return hashlib.sha256(fingerprint.encode()).digest()

    @staticmethod
    def verify_hardware(expected_fingerprint: str) -> bool:
        return HardwareBinding.get_fingerprint() == expected_fingerprint
