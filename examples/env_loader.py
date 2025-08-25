#!/usr/bin/env python3
"""
Environment Variable Loader for Parlant Examples

This module provides utilities for loading environment variables from .env files
in a consistent way across different examples.
"""

import os
from pathlib import Path
from typing import Optional


def load_env_file(env_file_name: str = ".env", search_paths: Optional[list[Path]] = None) -> bool:
    """
    Load environment variables from a .env file.
    
    Args:
        env_file_name: Name of the .env file to load
        search_paths: List of paths to search for the .env file. 
                     If None, searches current directory and project root.
    
    Returns:
        bool: True if .env file was found and loaded, False otherwise
    """
    if search_paths is None:
        # Default search paths: current directory and project root
        current_dir = Path.cwd()
        project_root = Path(__file__).parent.parent
        search_paths = [current_dir, project_root]
    
    # Try to import python-dotenv
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("ℹ️  python-dotenv not installed, using system environment variables")
        return False
    
    # Search for .env file in specified paths
    for search_path in search_paths:
        env_file = search_path / env_file_name
        if env_file.exists():
            load_dotenv(env_file)
            print(f"📄 Loaded environment variables from {env_file}")
            return True
    
    print(f"ℹ️  No {env_file_name} file found in search paths, using system environment variables")
    return False


def check_required_env_vars(required_vars: list[str]) -> bool:
    """
    Check if required environment variables are set.
    
    Args:
        required_vars: List of required environment variable names
    
    Returns:
        bool: True if all required variables are set, False otherwise
    """
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in one of these ways:")
        print("1. Create a .env file with the required variables")
        print("2. Set environment variables: export VAR_NAME=value")
        print("3. Set in code: os.environ['VAR_NAME'] = 'value'")
        return False
    
    return True


def setup_environment(env_file_name: str = ".env", required_vars: Optional[list[str]] = None) -> bool:
    """
    Complete environment setup: load .env file and check required variables.
    
    Args:
        env_file_name: Name of the .env file to load
        required_vars: List of required environment variable names to check
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    # Load .env file
    load_env_file(env_file_name)
    
    # Check required variables if specified
    if required_vars:
        return check_required_env_vars(required_vars)
    
    return True


# Auto-load environment variables when module is imported
load_env_file()
