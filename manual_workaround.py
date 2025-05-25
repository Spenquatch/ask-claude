#!/usr/bin/env python3
"""
Manual workaround to extract content from Claude Code JSON responses.
"""

import subprocess
import json

def ask_claude_manual(query, use_json=False):
    """Manually call Claude Code and handle the response."""
    
    cmd = ["claude", "--print", query]
    if use_json:
        cmd.extend(["--output-format", "json"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"Error: Command failed with code {result.returncode}")
            print(f"Stderr: {result.stderr}")
            return None
        
        if use_json:
            try:
                data = json.loads(result.stdout)
                print("DEBUG - JSON structure:")
                print(json.dumps(data, indent=2))
                
                # Try different possible content locations
                content = None
                session_id = None
                
                if isinstance(data, dict):
                    # Common content field names
                    for field in ['content', 'response', 'text', 'message', 'output', 'result']:
                        if field in data and data[field]:
                            content = data[field]
                            break
                    
                    # Common session ID field names
                    for field in ['session_id', 'sessionId', 'conversation_id', 'id']:
                        if field in data:
                            session_id = data[field]
                            break
                    
                    # If no direct content, check for messages array
                    if not content and 'messages' in data:
                        messages = data['messages']
                        if isinstance(messages, list):
                            for msg in reversed(messages):
                                if isinstance(msg, dict) and msg.get('role') == 'assistant':
                                    content = msg.get('content', '')
                                    break
                    
                    # Last resort: if content is still empty, show all string values
                    if not content:
                        print("DEBUG - No obvious content field found. All string values:")
                        for key, value in data.items():
                            if isinstance(value, str) and len(value) > 10:
                                print(f"  {key}: {value[:100]}...")
                                if not content:  # Use the first substantial string
                                    content = value
                
                return {
                    'content': content or "No content found",
                    'session_id': session_id,
                    'raw_data': data
                }
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                return {'content': result.stdout, 'session_id': None, 'raw_data': None}
        else:
            return {'content': result.stdout.strip(), 'session_id': None, 'raw_data': None}
    
    except Exception as e:
        print(f"Error running command: {e}")
        return None

def main():
    print("=== Manual Claude Code Test ===\n")
    
    query = "What is Python?"
    
    # Test text format
    print("1. Text format:")
    result = ask_claude_manual(query, use_json=False)
    if result:
        print(f"Content: {result['content']}")
    
    print("\n" + "-"*40 + "\n")
    
    # Test JSON format
    print("2. JSON format:")
    result = ask_claude_manual(query, use_json=True)
    if result:
        print(f"Content: '{result['content']}'")
        print(f"Session ID: {result['session_id']}")
        if result['raw_data']:
            print(f"Available fields: {list(result['raw_data'].keys())}")

if __name__ == "__main__":
    main()
