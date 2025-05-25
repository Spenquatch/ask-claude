"""
Debug script to examine Claude Code JSON responses

This script helps debug what Claude Code actually returns in different formats
so we can adjust the wrapper parsing logic accordingly.
"""

import json
import subprocess
import sys
from pathlib import Path

def run_claude_debug(query: str, output_format: str = "json"):
    """Run Claude Code and show the raw response."""
    print(f"\n{'='*60}")
    print(f"DEBUGGING CLAUDE CODE RESPONSE")
    print(f"Query: {query}")
    print(f"Format: {output_format}")
    print(f"{'='*60}")
    
    # Build command
    cmd = ["claude", "--print", query]
    if output_format != "text":
        cmd.extend(["--output-format", output_format])
    
    print(f"Command: {' '.join(cmd)}")
    print(f"{'-'*60}")
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDERR: {result.stderr}")
        print(f"{'-'*60}")
        print("RAW STDOUT:")
        print(repr(result.stdout))
        print(f"{'-'*60}")
        print("FORMATTED STDOUT:")
        print(result.stdout)
        
        # Try to parse as JSON if format is json
        if output_format == "json" and result.stdout.strip():
            print(f"{'-'*60}")
            print("JSON PARSING ATTEMPT:")
            try:
                parsed = json.loads(result.stdout)
                print("✅ Valid JSON")
                print("JSON Structure:")
                print(json.dumps(parsed, indent=2))
                
                print(f"{'-'*40}")
                print("FIELD ANALYSIS:")
                if isinstance(parsed, dict):
                    print(f"Available fields: {list(parsed.keys())}")
                    for key, value in parsed.items():
                        print(f"  {key}: {type(value).__name__} = {repr(value)[:100]}...")
                else:
                    print(f"Response is a {type(parsed).__name__}: {repr(parsed)[:200]}...")
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print("This might not be valid JSON")
        
        return result
        
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with return code {e.returncode}")
        print(f"STDERR: {e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def test_wrapper_parsing():
    """Test our wrapper's parsing logic."""
    print(f"\n{'='*60}")
    print("TESTING WRAPPER PARSING")
    print(f"{'='*60}")
    
    try:
        # Import our wrapper
        sys.path.append(str(Path.cwd()))
        from claude_code_wrapper import ClaudeCodeWrapper, OutputFormat
        
        wrapper = ClaudeCodeWrapper()
        
        # Test with JSON format
        print("Testing JSON format...")
        response = wrapper.run("What is 2+2?", output_format=OutputFormat.JSON)
        
        print(f"Content: '{response.content}'")
        print(f"Session ID: {response.session_id}")
        print(f"Metadata: {response.metadata}")
        print(f"Raw output length: {len(response.raw_output)}")
        print(f"Raw output preview: {repr(response.raw_output[:200])}")
        
    except ImportError as e:
        print(f"❌ Could not import wrapper: {e}")
        print("Make sure claude_code_wrapper.py is in the current directory")
    except Exception as e:
        print(f"❌ Error testing wrapper: {e}")


def main():
    """Main debug function."""
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What is Python?"
    
    # Test different formats
    for fmt in ["text", "json"]:
        run_claude_debug(query, fmt)
    
    # Test our wrapper
    test_wrapper_parsing()
    
    print(f"\n{'='*60}")
    print("DEBUG COMPLETE")
    print("If you see issues, please share this output to help fix the wrapper!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
