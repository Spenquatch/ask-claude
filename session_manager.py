"""
Session Management for Claude Code Wrapper

Enhanced session features for autonomous development pipelines:
1. Session persistence (save/load from disk)
2. Session branching and merging
3. Session replay and modification
4. Session templates and presets
5. Automatic session recovery
"""

import json
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import hashlib
from claude_code_wrapper import ClaudeCodeSession, ClaudeCodeResponse


class SessionManager:
    """Enhanced session manager with persistence and advanced features."""
    
    def __init__(self, session_dir: str = ".claude_sessions"):
        """Initialize session manager with storage directory."""
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)
        self.active_sessions: Dict[str, ClaudeCodeSession] = {}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        
    def save_session(self, session: ClaudeCodeSession, 
                     tags: Optional[List[str]] = None,
                     description: Optional[str] = None) -> str:
        """Save session to disk with metadata."""
        session_data = {
            "session_id": session.session_id,
            "messages": session.messages,
            "history": [resp.to_dict() for resp in session.history],
            "created_at": session.created_at,
            "total_duration": session.total_duration,
            "total_retries": session.total_retries,
            "metadata": session.metadata,
            "saved_at": datetime.now().isoformat(),
            "tags": tags or [],
            "description": description or ""
        }
        
        # Save as JSON for human readability
        session_file = self.session_dir / f"{session.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
            
        # Update metadata index
        self._update_session_index(session.session_id, tags, description)
        
        return str(session_file)
    
    def load_session(self, session_id: str, wrapper=None) -> ClaudeCodeSession:
        """Load session from disk."""
        session_file = self.session_dir / f"{session_id}.json"
        
        if not session_file.exists():
            raise ValueError(f"Session {session_id} not found")
            
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # Reconstruct session
        from claude_code_wrapper import ClaudeCodeWrapper
        if wrapper is None:
            wrapper = ClaudeCodeWrapper()
            
        session = ClaudeCodeSession(wrapper, session_id=session_data["session_id"])
        session.messages = session_data["messages"]
        session.created_at = session_data["created_at"]
        session.total_duration = session_data["total_duration"]
        session.total_retries = session_data["total_retries"]
        session.metadata = session_data["metadata"]
        
        # Note: history contains response dicts, not ClaudeCodeResponse objects
        # This is a limitation but acceptable for session restore
        
        return session
    
    def list_sessions(self, tags: Optional[List[str]] = None,
                     date_from: Optional[datetime] = None,
                     date_to: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """List saved sessions with optional filtering."""
        index_file = self.session_dir / "session_index.json"
        
        if not index_file.exists():
            return []
            
        with open(index_file, 'r') as f:
            index = json.load(f)
            
        results = []
        for session_id, metadata in index.items():
            # Filter by tags
            if tags and not any(tag in metadata.get("tags", []) for tag in tags):
                continue
                
            # Filter by date
            saved_at = datetime.fromisoformat(metadata["saved_at"])
            if date_from and saved_at < date_from:
                continue
            if date_to and saved_at > date_to:
                continue
                
            results.append({
                "session_id": session_id,
                **metadata
            })
            
        return sorted(results, key=lambda x: x["saved_at"], reverse=True)
    
    def branch_session(self, session: ClaudeCodeSession, 
                      branch_point: int,
                      branch_name: str) -> ClaudeCodeSession:
        """Create a branch from a session at a specific message index."""
        # Create new session with messages up to branch point
        from claude_code_wrapper import ClaudeCodeWrapper
        branch_session = ClaudeCodeSession(
            session.wrapper,
            session_id=f"{session.session_id}-{branch_name}"
        )
        
        # Copy messages up to branch point
        branch_session.messages = session.messages[:branch_point].copy()
        branch_session.metadata = {
            **session.metadata,
            "branched_from": session.session_id,
            "branch_point": branch_point,
            "branch_name": branch_name
        }
        
        return branch_session
    
    def merge_sessions(self, session1: ClaudeCodeSession,
                      session2: ClaudeCodeSession,
                      merge_strategy: str = "append") -> ClaudeCodeSession:
        """Merge two sessions with different strategies."""
        from claude_code_wrapper import ClaudeCodeWrapper
        merged = ClaudeCodeSession(
            session1.wrapper,
            session_id=f"merged-{session1.session_id}-{session2.session_id}"
        )
        
        if merge_strategy == "append":
            # Simply append session2 after session1
            merged.messages = session1.messages + session2.messages
        elif merge_strategy == "interleave":
            # Interleave messages based on timestamp
            all_messages = []
            for msg in session1.messages + session2.messages:
                msg_copy = msg.copy()
                msg_copy["_source_session"] = (
                    session1.session_id if msg in session1.messages 
                    else session2.session_id
                )
                all_messages.append(msg_copy)
            
            # Sort by timestamp
            all_messages.sort(key=lambda x: x.get("timestamp", 0))
            merged.messages = all_messages
        
        merged.metadata = {
            "merged_from": [session1.session_id, session2.session_id],
            "merge_strategy": merge_strategy,
            "merge_time": datetime.now().isoformat()
        }
        
        return merged
    
    def create_checkpoint(self, session: ClaudeCodeSession, 
                         checkpoint_name: str) -> str:
        """Create a checkpoint of the current session state."""
        checkpoint_id = f"{session.session_id}-checkpoint-{checkpoint_name}"
        checkpoint_data = {
            "session_id": session.session_id,
            "checkpoint_name": checkpoint_name,
            "checkpoint_time": datetime.now().isoformat(),
            "message_count": len(session.messages),
            "messages": session.messages.copy(),
            "metadata": session.metadata.copy()
        }
        
        checkpoint_file = self.session_dir / f"checkpoints/{checkpoint_id}.json"
        checkpoint_file.parent.mkdir(exist_ok=True)
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str, 
                          wrapper=None) -> ClaudeCodeSession:
        """Restore session from a checkpoint."""
        checkpoint_file = self.session_dir / f"checkpoints/{checkpoint_id}.json"
        
        if not checkpoint_file.exists():
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
            
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            
        from claude_code_wrapper import ClaudeCodeWrapper
        if wrapper is None:
            wrapper = ClaudeCodeWrapper()
            
        session = ClaudeCodeSession(
            wrapper, 
            session_id=checkpoint_data["session_id"]
        )
        session.messages = checkpoint_data["messages"]
        session.metadata = checkpoint_data["metadata"]
        session.metadata["restored_from_checkpoint"] = checkpoint_id
        
        return session
    
    def export_session(self, session: ClaudeCodeSession,
                      format: str = "markdown",
                      include_metadata: bool = True) -> str:
        """Export session in various formats for documentation."""
        if format == "markdown":
            output = f"# Claude Code Session: {session.session_id}\n\n"
            
            if include_metadata:
                output += f"**Created**: {datetime.fromtimestamp(session.created_at).isoformat()}\n"
                output += f"**Duration**: {session.total_duration:.2f}s\n"
                output += f"**Messages**: {len(session.messages)}\n\n"
            
            output += "## Conversation\n\n"
            
            for msg in session.messages:
                role = msg["role"].capitalize()
                content = msg["content"]
                output += f"### {role}\n\n{content}\n\n"
                
                if include_metadata and msg.get("metadata"):
                    output += f"*Metadata: {json.dumps(msg['metadata'])}*\n\n"
                    
        elif format == "json":
            output = json.dumps({
                "session_id": session.session_id,
                "messages": session.messages,
                "metadata": session.metadata,
                "stats": {
                    "created_at": session.created_at,
                    "total_duration": session.total_duration,
                    "total_retries": session.total_retries,
                    "message_count": len(session.messages)
                }
            }, indent=2)
            
        return output
    
    def _update_session_index(self, session_id: str,
                             tags: Optional[List[str]],
                             description: Optional[str]):
        """Update the session index file."""
        index_file = self.session_dir / "session_index.json"
        
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
        else:
            index = {}
            
        index[session_id] = {
            "saved_at": datetime.now().isoformat(),
            "tags": tags or [],
            "description": description or ""
        }
        
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)


class SessionTemplate:
    """Predefined session templates for common development tasks."""
    
    TEMPLATES = {
        "code_review": {
            "system_prompt": "You are a senior software engineer conducting a thorough code review. Focus on architecture, performance, security, and maintainability.",
            "initial_messages": [
                {
                    "role": "system",
                    "content": "Code review session initialized. Please provide the code to review."
                }
            ],
            "metadata": {
                "template": "code_review",
                "purpose": "Systematic code review with focus on best practices"
            }
        },
        "debugging": {
            "system_prompt": "You are a debugging expert. Help identify and fix issues systematically. Ask clarifying questions and suggest debugging strategies.",
            "initial_messages": [
                {
                    "role": "system", 
                    "content": "Debugging session started. What issue are you experiencing?"
                }
            ],
            "metadata": {
                "template": "debugging",
                "purpose": "Interactive debugging assistance"
            }
        },
        "architecture_design": {
            "system_prompt": "You are a software architect. Help design scalable, maintainable systems. Consider trade-offs and best practices.",
            "initial_messages": [
                {
                    "role": "system",
                    "content": "Architecture design session started. What system are we designing?"
                }
            ],
            "metadata": {
                "template": "architecture_design",
                "purpose": "System architecture planning and design"
            }
        },
        "test_development": {
            "system_prompt": "You are a test automation expert. Help write comprehensive, maintainable tests with good coverage.",
            "initial_messages": [
                {
                    "role": "system",
                    "content": "Test development session started. What functionality needs testing?"
                }
            ],
            "metadata": {
                "template": "test_development",
                "purpose": "Test creation and automation"
            }
        }
    }
    
    @classmethod
    def create_from_template(cls, template_name: str,
                           wrapper=None) -> ClaudeCodeSession:
        """Create a new session from a template."""
        if template_name not in cls.TEMPLATES:
            raise ValueError(f"Template {template_name} not found")
            
        template = cls.TEMPLATES[template_name]
        
        from claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeConfig
        if wrapper is None:
            config = ClaudeCodeConfig(
                system_prompt=template.get("system_prompt")
            )
            wrapper = ClaudeCodeWrapper(config)
            
        session = ClaudeCodeSession(wrapper)
        
        # Add initial messages
        for msg in template.get("initial_messages", []):
            session.add_message(msg["role"], msg["content"])
            
        # Set metadata
        session.metadata.update(template.get("metadata", {}))
        
        return session


class AutoRecoverySession:
    """Session wrapper with automatic recovery and persistence."""
    
    def __init__(self, wrapper, session_manager: SessionManager,
                 auto_save_interval: int = 5):
        """Initialize auto-recovery session."""
        self.wrapper = wrapper
        self.session_manager = session_manager
        self.auto_save_interval = auto_save_interval
        self.message_count_at_last_save = 0
        self.session = None
        
    def start_or_resume(self, session_id: Optional[str] = None) -> ClaudeCodeSession:
        """Start new session or resume existing one."""
        if session_id:
            try:
                self.session = self.session_manager.load_session(session_id, self.wrapper)
                print(f"Resumed session {session_id} with {len(self.session.messages)} messages")
            except Exception as e:
                print(f"Could not resume session: {e}. Starting new session.")
                self.session = ClaudeCodeSession(self.wrapper)
        else:
            self.session = ClaudeCodeSession(self.wrapper)
            
        return self.session
    
    def ask_with_recovery(self, query: str, **kwargs) -> ClaudeCodeResponse:
        """Ask a question with automatic saving."""
        if not self.session:
            raise ValueError("No active session. Call start_or_resume first.")
            
        try:
            response = self.session.ask(query, **kwargs)
            
            # Auto-save if enough messages accumulated
            if (len(self.session.messages) - self.message_count_at_last_save 
                >= self.auto_save_interval):
                self.save_session()
                
            return response
            
        except Exception as e:
            # Save session state before re-raising
            self.save_session(tags=["error", "auto-saved"])
            raise
    
    def save_session(self, tags: Optional[List[str]] = None,
                    description: Optional[str] = None):
        """Save current session state."""
        if self.session:
            self.session_manager.save_session(
                self.session,
                tags=tags or ["auto-saved"],
                description=description or "Auto-saved session"
            )
            self.message_count_at_last_save = len(self.session.messages)
            print(f"Session {self.session.session_id} saved")


# Example usage functions
def example_session_workflow():
    """Example of advanced session management workflow."""
    from claude_code_wrapper import ClaudeCodeWrapper
    
    # Initialize wrapper and session manager
    wrapper = ClaudeCodeWrapper()
    session_mgr = SessionManager()
    
    # Create session from template
    session = SessionTemplate.create_from_template("code_review", wrapper)
    
    # Add some interactions
    response1 = session.ask("Please review this Python function for performance...")
    
    # Create checkpoint
    checkpoint_id = session_mgr.create_checkpoint(session, "initial-review")
    
    # Continue conversation
    response2 = session.ask("What about error handling?")
    
    # Save session
    session_file = session_mgr.save_session(
        session, 
        tags=["code-review", "python", "performance"],
        description="Performance review of data processing function"
    )
    
    # Later: Load and branch session
    loaded_session = session_mgr.load_session(session.session_id, wrapper)
    branch = session_mgr.branch_session(loaded_session, 2, "alternative-approach")
    
    # Export for documentation
    markdown = session_mgr.export_session(session, format="markdown")
    
    return session, checkpoint_id, session_file


if __name__ == "__main__":
    # Demonstrate session management capabilities
    print("Session Management Enhancements Demo")
    print("=" * 50)
    
    session, checkpoint, file_path = example_session_workflow()
    print(f"Session saved to: {file_path}")
    print(f"Checkpoint created: {checkpoint}")