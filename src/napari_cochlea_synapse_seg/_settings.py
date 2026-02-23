"""
Settings persistence for napari-cochlea-synapse-seg plugin.

Provides JSON-based user-level configuration storage for all widget settings.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any


def get_settings_path() -> Path:
    """
    Get the user-level settings file path.
    
    Returns:
        Path to settings JSON in user config directory.
    """
    # Use platform-appropriate config directory
    if os.name == 'nt':  # Windows
        config_dir = Path(os.environ.get('APPDATA', Path.home()))
    elif os.name == 'posix':  # macOS/Linux
        config_dir = Path.home() / '.config'
    else:
        config_dir = Path.home()
    
    plugin_config_dir = config_dir / 'napari-cochlea-synapse-seg'
    plugin_config_dir.mkdir(parents=True, exist_ok=True)
    
    return plugin_config_dir / 'settings.json'


def load_settings(settings_path=None) -> Dict[str, Any]:
    """
    Load settings from JSON file.
    
    Returns:
        Settings dictionary with version and widget sections.
        Returns default empty structure if file doesn't exist.
    """
    if settings_path is None:
        settings_path = get_settings_path()
    
    if not settings_path.exists():
        return {
            'version': '1.0',
            'GTWidget': {},
            'PredWidget': {},
            'CropWidget': {},
            'PreProcessWidget': {}
        }
    
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        # Ensure all required sections exist
        for key in ['GTWidget', 'PredWidget', 'CropWidget', 'PreProcessWidget']:
            if key not in settings:
                settings[key] = {}
        
        return settings
    
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load settings from {settings_path}: {e}")
        return {
            'version': '1.0',
            'GTWidget': {},
            'PredWidget': {},
            'CropWidget': {},
            'PreProcessWidget': {}
        }


def save_settings(settings: Dict[str, Any], settings_path=None) -> bool:
    """
    Save settings to JSON file.
    
    Args:
        settings: Dictionary containing all widget settings.
        
    Returns:
        True if save was successful, False otherwise.
    """
    if settings_path is None:
        settings_path = get_settings_path()
    
    try:
        # Ensure version is set
        if 'version' not in settings:
            settings['version'] = '1.0'
        
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        
        return True
    
    except IOError as e:
        print(f"Error: Failed to save settings to {settings_path}: {e}")
        return False
