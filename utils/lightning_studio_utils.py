"""
Lightning AI Studio utilities, it supports automatic studio creation and cleanup for the benchmarking process.
"""

import os
import platform
import subprocess
import time
from typing import Optional, Tuple
import json


def is_auto_management_supported() -> bool:
    """
    Check if automatic studio management is supported on the current platform.
    Returns:
        bool: True if auto-management is supported (macOS/Linux/WSL), False for Windows
    """
    system = platform.system().lower()
    if system == "linux":
        try:
            with open("/proc/version", "r") as f:
                version_info = f.read().lower()
                if "microsoft" in version_info or "wsl" in version_info:
                    return True  
        except FileNotFoundError:
            pass
        return True  
    return system == "darwin"  


def run_lightning_command(command: str, timeout: int = 300) -> Tuple[bool, str]:
    """
    Run a lightning CLI command and return success status and output.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        return result.returncode == 0, result.stdout + result.stderr
        
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout} seconds"
    except Exception as e:
        return False, f"Command failed: {str(e)}"




def create_studio_if_needed(
    studio_name: str,
    teamspace: str,
    gpu_type: str = "L4",
    timeout: int = 600
) -> Tuple[bool, str, bool]:
    """
    Create a Lightning AI studio.
    Args:
        studio_name: Name of the studio
        teamspace: Teamspace name
        gpu_type: GPU type
        timeout: Creation/start timeout in seconds  
    Returns:
        Tuple of (success: bool, message: str, was_created: bool)
    """
    if not is_auto_management_supported():
        return False, "Auto studio management not supported on this platform", False
    
    print(f"Creating studio '{studio_name}' with GPU type '{gpu_type}'...")
    command = f"lightning create studio {studio_name} --start {gpu_type} --teamspace {teamspace}"
    success, output = run_lightning_command(command, timeout)
    
    if success:
        return True, "Studio ready", True
    else:
        return False, f"Failed to create studio: {output}", False


def stop_studio_if_created(studio_name: str, teamspace: str, was_created: bool = False) -> Tuple[bool, str]:
    """
    Stops the studio.
    """
    if not is_auto_management_supported():
        return False, "Auto studio management not supported on this platform"
    
    print(f"Stopping studio '{studio_name}'...")
    command = f"lightning stop studio {studio_name} --teamspace {teamspace}"
    success, output = run_lightning_command(command)
    
    if success:
        return True, "Studio stopped successfully"
    else:
        # If stop failed, it might already be stopped or not exist - that's fine
        if "not found" in output.lower() or "does not exist" in output.lower():
            return True, "Studio already stopped or doesn't exist"
        return False, f"Failed to stop studio: {output}"



def convert_gpu_id_to_lightning_format(gpu_id: str, gpu_count: int = 1) -> str:
    """
    Convert GPU ID and count to Lightning AI format.
    Args:
        gpu_id: GPU identifier (e.g., "L4", "A100", "H100")
        gpu_count: Number of GPUs needed     
    Returns:
        Lightning AI GPU type string (e.g., "L4_X_2")
    """
    # Use GPU ID as-is, just handle multi-GPU naming
    gpu_base = gpu_id.strip()
    
    # Handle multi-GPU configurations
    if gpu_count == 1:
        return gpu_base
    elif gpu_count > 1:
        return f"{gpu_base}_X_{gpu_count}"
    else:
        return gpu_base


