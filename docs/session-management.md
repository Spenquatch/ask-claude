# Session Management Guide

The Claude Code Wrapper provides comprehensive session management capabilities essential for building autonomous development pipelines and maintaining conversation continuity.

## Table of Contents
- [Basic Session Management](#basic-session-management)
- [Session Continuation](#session-continuation)
- [Session Persistence](#session-persistence)
- [Advanced Features](#advanced-features)
- [Autonomous Pipeline Integration](#autonomous-pipeline-integration)

## Basic Session Management

### Starting a New Session

```python
from claude_code_wrapper import ClaudeCodeWrapper

wrapper = ClaudeCodeWrapper()
response = wrapper.run("Create a Python web server")
print(f"Session ID: {response.session_id}")
```

### Creating Named Sessions

```python
session = wrapper.create_session("project-backend-v1")
response = session.ask("Design a REST API for user management")
```

## Session Continuation

### Continue Most Recent Conversation (-c flag)

The wrapper supports the Claude Code CLI's `-c` flag for continuing conversations:

```python
# Method 1: Using the wrapper method
response = wrapper.continue_conversation("What about authentication?")

# Method 2: Using the convenience function
from claude_code_wrapper import continue_claude
response = continue_claude()

# Method 3: Using configuration
from claude_code_wrapper import ClaudeCodeConfig
config = ClaudeCodeConfig(continue_session=True)
wrapper = ClaudeCodeWrapper(config)
response = wrapper.run("Continue with the previous topic")
```

### Resume Specific Session (--resume flag)

Resume a specific session by ID:

```python
# Method 1: Using the wrapper method
response = wrapper.resume_specific_session("abc-123-def", "Add error handling")

# Method 2: Using the convenience function
from claude_code_wrapper import resume_claude
response = resume_claude("abc-123-def", "Add error handling")

# Method 3: Using configuration
config = ClaudeCodeConfig(session_id="abc-123-def")
wrapper = ClaudeCodeWrapper(config)
response = wrapper.run("Add error handling")
```

### Session-Aware Convenience Function

```python
from claude_code_wrapper import ask_claude_with_session

# Continue last session
response = ask_claude_with_session("Continue the implementation", continue_last=True)

# Resume specific session
response = ask_claude_with_session("Fix the bug", session_id="abc-123")
```

## Session Persistence

### Save and Load Sessions

```python
from session_enhancements import SessionManager

# Initialize session manager
session_mgr = SessionManager(".claude_sessions")

# Save current session
session_file = session_mgr.save_session(
    session,
    tags=["backend", "api", "python"],
    description="User management API design session"
)

# Load session later
loaded_session = session_mgr.load_session("session-id", wrapper)
```

### List and Filter Sessions

```python
# List all sessions
all_sessions = session_mgr.list_sessions()

# Filter by tags
api_sessions = session_mgr.list_sessions(tags=["api"])

# Filter by date
from datetime import datetime, timedelta
recent = session_mgr.list_sessions(
    date_from=datetime.now() - timedelta(days=7)
)
```

### Export Sessions

```python
# Export as Markdown (great for documentation)
markdown = session_mgr.export_session(session, format="markdown")
with open("api_design_discussion.md", "w") as f:
    f.write(markdown)

# Export as JSON (for further processing)
json_data = session_mgr.export_session(session, format="json")
```

## Advanced Features

### Session Branching

Explore alternative approaches without losing the main conversation:

```python
# Create a branch at message 5
branch = session_mgr.branch_session(main_session, 5, "microservices-approach")
branch_response = branch.ask("What if we used microservices instead?")

# Save the branch
session_mgr.save_session(branch, tags=["architecture", "alternative"])
```

### Checkpoints

Create checkpoints to mark important points in the conversation:

```python
# After initial design
checkpoint1 = session_mgr.create_checkpoint(session, "initial-design")

# After adding authentication
checkpoint2 = session_mgr.create_checkpoint(session, "with-auth")

# Restore from checkpoint if needed
restored = session_mgr.restore_checkpoint(checkpoint1)
```

### Session Templates

Use predefined templates for common tasks:

```python
from session_enhancements import SessionTemplate

# Start a code review session
review_session = SessionTemplate.create_from_template("code_review", wrapper)
response = review_session.ask("Review this function: ...")

# Available templates:
# - code_review
# - debugging
# - architecture_design
# - test_development
```

### Session Merging

Combine insights from multiple sessions:

```python
# Merge two architecture discussions
merged = session_mgr.merge_sessions(
    session1, 
    session2, 
    merge_strategy="interleave"  # or "append"
)
```

## Autonomous Pipeline Integration

### Auto-Recovery Sessions

Perfect for long-running autonomous pipelines:

```python
from session_enhancements import AutoRecoverySession

# Create auto-recovery session
auto_session = AutoRecoverySession(
    wrapper, 
    session_mgr, 
    auto_save_interval=5  # Save every 5 messages
)

# Start or resume
session = auto_session.start_or_resume("pipeline-session-1")

# Use with automatic saving
try:
    response = auto_session.ask_with_recovery("Generate unit tests")
except Exception as e:
    print("Error occurred, but session was auto-saved")
    # Can resume from last save point
```

### Pipeline Example

```python
class DevelopmentPipeline:
    def __init__(self):
        self.wrapper = ClaudeCodeWrapper()
        self.session_mgr = SessionManager(".pipeline_sessions")
        
    def run_pipeline(self, project_spec):
        # Start new or resume existing pipeline
        session = self.wrapper.create_session(f"pipeline-{project_spec['id']}")
        
        stages = [
            ("requirements", self.gather_requirements),
            ("design", self.design_architecture),
            ("implementation", self.implement_code),
            ("testing", self.write_tests),
            ("documentation", self.generate_docs)
        ]
        
        for stage_name, stage_func in stages:
            try:
                # Create checkpoint before each stage
                checkpoint = self.session_mgr.create_checkpoint(
                    session, 
                    f"before-{stage_name}"
                )
                
                # Run stage
                result = stage_func(session, project_spec)
                
                # Save progress
                self.session_mgr.save_session(
                    session,
                    tags=["pipeline", stage_name, project_spec['id']],
                    description=f"Completed {stage_name}"
                )
                
            except Exception as e:
                print(f"Error in {stage_name}: {e}")
                # Can restore from checkpoint
                session = self.session_mgr.restore_checkpoint(checkpoint)
                
    def gather_requirements(self, session, spec):
        return session.ask(f"Analyze these requirements: {spec['requirements']}")
        
    # ... other stage methods
```

### Parallel Session Management

For exploring multiple approaches simultaneously:

```python
from concurrent.futures import ThreadPoolExecutor

def explore_approach(approach_name, base_session, session_mgr):
    # Branch from base session
    branch = session_mgr.branch_session(base_session, 2, approach_name)
    
    # Explore this approach
    response = branch.ask(f"Implement using {approach_name} pattern")
    
    # Save results
    session_mgr.save_session(branch, tags=["exploration", approach_name])
    return approach_name, response

# Explore multiple approaches in parallel
approaches = ["mvc", "microservices", "serverless", "monolithic"]
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(explore_approach, approach, base_session, session_mgr)
        for approach in approaches
    ]
    
    results = [f.result() for f in futures]
```

## Best Practices

1. **Always Save Important Sessions**: Use tags and descriptions for easy retrieval
2. **Create Checkpoints**: Before major changes or experiments
3. **Use Templates**: For consistency across similar tasks
4. **Branch for Experiments**: Keep main conversation clean
5. **Auto-Recovery for Pipelines**: Prevents loss of progress
6. **Export for Documentation**: Generate docs from design sessions

## Command Line Usage

The wrapper fully supports Claude's session flags:

```bash
# Continue last conversation
python -c "from claude_code_wrapper import continue_claude; print(continue_claude().content)"

# Resume specific session
python -c "from claude_code_wrapper import resume_claude; print(resume_claude('abc-123').content)"
```

## Configuration Options

```python
config = ClaudeCodeConfig(
    # Session management
    session_id="specific-session-id",     # Resume this session
    continue_session=True,                # Continue last session
    
    # Other options
    timeout=120,
    max_retries=5,
    verbose=True
)
```

## Troubleshooting

### Session Not Found
```python
try:
    session = session_mgr.load_session("unknown-id")
except ValueError as e:
    print(f"Session not found: {e}")
    # List available sessions
    available = session_mgr.list_sessions()
```

### Corrupted Session
```python
# Use checkpoints to recover
checkpoints = Path(".claude_sessions/checkpoints").glob("session-id-*")
for checkpoint in checkpoints:
    try:
        restored = session_mgr.restore_checkpoint(checkpoint.stem)
        print(f"Restored from {checkpoint}")
        break
    except Exception:
        continue
```

## API Reference

### ClaudeCodeWrapper Session Methods
- `continue_conversation(query="")` - Continue most recent conversation
- `resume_specific_session(session_id, query="")` - Resume specific session
- `create_session(session_id=None)` - Create new session
- `get_last_session_id()` - Get ID of last used session

### SessionManager Methods
- `save_session(session, tags=None, description=None)` - Save session to disk
- `load_session(session_id, wrapper=None)` - Load session from disk
- `list_sessions(tags=None, date_from=None, date_to=None)` - List sessions
- `branch_session(session, branch_point, branch_name)` - Create branch
- `merge_sessions(session1, session2, merge_strategy="append")` - Merge sessions
- `create_checkpoint(session, checkpoint_name)` - Create checkpoint
- `restore_checkpoint(checkpoint_id, wrapper=None)` - Restore checkpoint
- `export_session(session, format="markdown", include_metadata=True)` - Export session

### Convenience Functions
- `continue_claude(**kwargs)` - Continue last conversation
- `resume_claude(session_id, query="", **kwargs)` - Resume specific session
- `ask_claude_with_session(query, session_id=None, continue_last=False, **kwargs)` - Session-aware query