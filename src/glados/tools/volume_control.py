"""Volume control tool implementation for GLaDOS."""

import subprocess
import re
from typing import Dict, Any, Optional
from loguru import logger

from .text_to_number import text_to_number


class VolumeControlTool:
    """Tool for controlling system volume through GLaDOS."""
    
    def __init__(self):
        self.name = "volume_control"
        self.description = "Control system volume (up/down/set)"
        
    def get_current_volume(self) -> Optional[int]:
        """Get current system volume percentage."""
        try:
            # Try PulseAudio first (most common on Linux)
            result = subprocess.run(
                ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # Extract volume percentage from pactl output
                match = re.search(r'(\d+)%', result.stdout)
                if match:
                    return int(match.group(1))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        try:
            # Fallback to ALSA (amixer)
            result = subprocess.run(
                ["amixer", "sget", "Master"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # Extract volume percentage from amixer output
                match = re.search(r'\[(\d+)%\]', result.stdout)
                if match:
                    return int(match.group(1))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        logger.error("Could not get current volume - neither PulseAudio nor ALSA available")
        return None
    
    def set_volume(self, level: int) -> Dict[str, Any]:
        """Set volume to specific percentage (0-100)."""
        if not 0 <= level <= 100:
            return {"success": False, "error": "Volume must be between 0 and 100"}
        
        try:
            # Try PulseAudio first
            result = subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Volume set to {level}% using PulseAudio")
                return {"success": True, "volume": level, "method": "pulseaudio"}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        try:
            # Fallback to ALSA
            result = subprocess.run(
                ["amixer", "sset", "Master", f"{level}%"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Volume set to {level}% using ALSA")
                return {"success": True, "volume": level, "method": "alsa"}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        error_msg = "Failed to set volume - neither PulseAudio nor ALSA available"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    def adjust_volume(self, direction: str, amount: int) -> Dict[str, Any]:
        """Adjust volume up or down by percentage."""
        current = self.get_current_volume()
        if current is None:
            return {"success": False, "error": "Could not get current volume"}
        
        if direction.lower() in ["up", "increase", "raise"]:
            new_volume = min(100, current + amount)
        elif direction.lower() in ["down", "decrease", "lower", "reduce"]:
            new_volume = max(0, current - amount)
        else:
            return {"success": False, "error": f"Invalid direction: {direction}"}
        
        return self.set_volume(new_volume)
    
    def parse_volume_command(self, command: str) -> Dict[str, Any]:
        """
        Parse volume command and execute appropriate action.
        
        Examples:
        - "volume down by 13 percent"
        - "set volume to 50"
        - "increase volume by 10"
        - "volume up 25"
        """
        command_lower = command.lower()
        
        # Extract number from command
        amount = text_to_number(command)
        if amount is None:
            return {"success": False, "error": "Could not extract number from command"}
        
        # Determine action type
        if "set" in command_lower or "to" in command_lower:
            # Set volume to specific value
            return self.set_volume(amount)
        
        elif any(word in command_lower for word in ["up", "increase", "raise", "higher"]):
            # Increase volume
            return self.adjust_volume("up", amount)
        
        elif any(word in command_lower for word in ["down", "decrease", "lower", "reduce", "quieter"]):
            # Decrease volume
            return self.adjust_volume("down", amount)
        
        else:
            return {"success": False, "error": "Could not determine volume action from command"}
    
    def execute(self, command: str) -> Dict[str, Any]:
        """Execute volume control command."""
        logger.info(f"Volume control executing command: {command}")
        result = self.parse_volume_command(command)
        
        if result.get("success"):
            if "set" in command.lower():
                action = f"set to {result.get('volume', 'unknown')}"
            else:
                direction = "up" if any(word in command.lower() for word in ["up", "increase", "raise"]) else "down"
                action = f"adjusted {direction} to {result.get('volume', 'unknown')}"
            logger.success(f"Volume {action}%")
        else:
            logger.error(f"Volume control failed: {result.get('error')}")
        
        return result


# Test the tool
if __name__ == "__main__":
    tool = VolumeControlTool()
    
    test_commands = [
        "volume down by 13 percent",
        "set volume to 50",
        "increase volume by 10",
        "volume up 25"
    ]
    
    for cmd in test_commands:
        print(f"Testing: {cmd}")
        result = tool.execute(cmd)
        print(f"Result: {result}\n")
