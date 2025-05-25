#!/usr/bin/env python3
"""
Improved basic usage example with debugging capabilities.

This example shows how to use the Claude Code wrapper with better error handling
and debugging information to help troubleshoot issues.
"""

import logging
from claude_code_wrapper import ClaudeCodeWrapper, OutputFormat, ask_claude, ask_claude_json

# Enable debug logging to see what's happening
logging.basicConfig(level=logging.DEBUG)

def main():
    print("=== Claude Code Wrapper - Basic Usage ===\n")
    
    # 1. Simple text query
    print("1. Simple text query:")
    try:
        response = ask_claude("What is Python?")
        print(f"Response: {response.content}")
        print(f"Return code: {response.returncode}")
        if response.stderr:
            print(f"Stderr: {response.stderr}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "-"*50 + "\n")
    
    # 2. JSON query with debugging
    print("2. JSON query with debugging:")
    try:
        wrapper = ClaudeCodeWrapper()
        response = wrapper.run("Explain machine learning briefly", output_format=OutputFormat.JSON)
        
        print(f"Content: '{response.content}'")
        print(f"Content length: {len(response.content)}")
        print(f"Session ID: {response.session_id}")
        print(f"Metadata: {response.metadata}")
        print(f"Return code: {response.returncode}")
        
        # Show raw output for debugging
        print(f"\nRaw output (first 500 chars):")
        print(repr(response.raw_output[:500]))
        
        if response.stderr:
            print(f"\nStderr: {response.stderr}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-"*50 + "\n")
    
    # 3. Test different output formats
    print("3. Testing different output formats:")
    
    query = "What is 2+2?"
    
    for fmt_name, fmt in [("TEXT", OutputFormat.TEXT), ("JSON", OutputFormat.JSON)]:
        print(f"\n{fmt_name} format:")
        try:
            wrapper = ClaudeCodeWrapper()
            response = wrapper.run(query, output_format=fmt)
            print(f"  Content: '{response.content[:100]}{'...' if len(response.content) > 100 else ''}'")
            print(f"  Session ID: {response.session_id}")
            print(f"  Has metadata: {bool(response.metadata)}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "-"*50 + "\n")
    
    # 4. Test streaming (if it works)
    print("4. Testing streaming:")
    try:
        from claude_code_wrapper import ask_claude_streaming
        
        print("Streaming response:")
        event_count = 0
        for event in ask_claude_streaming("Count from 1 to 3"):
            event_count += 1
            print(f"  Event {event_count}: {event}")
            if event_count > 10:  # Safety limit
                break
                
    except Exception as e:
        print(f"Streaming error: {e}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
