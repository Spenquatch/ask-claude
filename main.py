from claude_code_wrapper import ask_claude, ask_claude_json, ask_claude_streaming

# Quick single questions
response = ask_claude("Write a hello world program in Python")
print(response.content)

# JSON response
response = ask_claude_json("Analyze this data structure")
print(response.metadata)

# Streaming response
for event in ask_claude_streaming("Generate a long explanation"):
    if event.get("type") == "message":
        print(event.get("content", ""), end="", flush=True)