#!/usr/bin/env python3
"""
Quick fix to see what Claude Code is actually returning in JSON format.
"""

import subprocess
import json
import sys

def debug_claude_json():
    """Debug what Claude Code returns in JSON format."""
    
    query = "What is Python?"
    cmd = ["claude", "--print", query, "--output-format", "json"]
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        print(f"Return code: {result.returncode}")
        print(f"Stderr: {result.stderr}")
        print("Raw stdout:")
        print(repr(result.stdout))
        print("-" * 50)
        print("Formatted stdout:")
        print(result.stdout)
        print("-" * 50)
        
        if result.stdout.strip():
            try:
                data = json.loads(result.stdout)
                print("Parsed JSON:")
                print(json.dumps(data, indent=2))
                print("-" * 50)
                print("Available keys:", list(data.keys()) if isinstance(data, dict) else "Not a dict")
                
                # Show all fields and their types
                if isinstance(data, dict):
                    for key, value in data.items():
                        print(f"  {key}: {type(value).__name__} = {repr(value)}")
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_claude_json()
