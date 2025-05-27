#!/usr/bin/env python3
"""
Session Management Demo for Claude Code Wrapper

This demonstrates the enhanced session management capabilities including:
- Continuing conversations with -c flag
- Resuming specific sessions with --resume
- Session persistence and recovery
- Session branching and checkpoints
- Autonomous development pipeline integration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeConfig
from session_enhancements import SessionManager, SessionTemplate, AutoRecoverySession
import time


def demo_basic_session_continuation():
    """Demonstrate basic session continuation with -c flag."""
    print("\n=== Basic Session Continuation Demo ===")
    
    wrapper = ClaudeCodeWrapper()
    
    # Start a conversation
    print("\n1. Starting new conversation...")
    response1 = wrapper.run("Hello! Let's create a Python function to calculate fibonacci numbers.")
    print(f"Session ID: {response1.session_id}")
    print(f"Response preview: {response1.content[:100]}...")
    
    # Continue the conversation using the -c flag behavior
    print("\n2. Continuing conversation with -c flag...")
    response2 = wrapper.continue_conversation("Now let's optimize it using memoization")
    print(f"Session ID: {response2.session_id}")
    print(f"Continued: {response2.content[:100]}...")
    
    # Resume specific session
    if response1.session_id:
        print(f"\n3. Resuming specific session {response1.session_id}...")
        response3 = wrapper.resume_specific_session(
            response1.session_id,
            "Can you add type hints to the function?"
        )
        print(f"Resumed: {response3.content[:100]}...")
    
    return wrapper


def demo_session_persistence():
    """Demonstrate session persistence for development pipelines."""
    print("\n=== Session Persistence Demo ===")
    
    wrapper = ClaudeCodeWrapper()
    session_mgr = SessionManager(".claude_dev_sessions")
    
    # Create a code review session from template
    print("\n1. Creating code review session from template...")
    session = SessionTemplate.create_from_template("code_review", wrapper)
    
    # Simulate code review process
    print("\n2. Starting code review...")
    code_snippet = '''
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
'''
    
    response1 = session.ask(f"Please review this Python function:\n```python\n{code_snippet}\n```")
    print(f"Review started: {response1.content[:150]}...")
    
    # Create checkpoint after initial review
    print("\n3. Creating checkpoint...")
    checkpoint_id = session_mgr.create_checkpoint(session, "initial-review")
    print(f"Checkpoint created: {checkpoint_id}")
    
    # Continue with specific concerns
    response2 = session.ask("What about performance for large datasets?")
    print(f"Performance discussion: {response2.content[:150]}...")
    
    # Save session
    print("\n4. Saving session...")
    session_file = session_mgr.save_session(
        session,
        tags=["code-review", "python", "performance", "data-processing"],
        description="Review of data processing function with performance considerations"
    )
    print(f"Session saved to: {session_file}")
    
    # Demonstrate loading and continuing
    print("\n5. Loading saved session...")
    loaded_session = session_mgr.load_session(session.session_id, wrapper)
    print(f"Loaded session with {len(loaded_session.messages)} messages")
    
    return session_mgr, session


def demo_autonomous_pipeline():
    """Demonstrate session management in autonomous development pipeline."""
    print("\n=== Autonomous Development Pipeline Demo ===")
    
    wrapper = ClaudeCodeWrapper()
    session_mgr = SessionManager(".claude_pipeline")
    auto_session = AutoRecoverySession(wrapper, session_mgr, auto_save_interval=3)
    
    # Simulate multi-stage development pipeline
    stages = [
        ("requirements", "Generate requirements for a REST API that manages user profiles"),
        ("design", "Create the API design with endpoints and data models"),
        ("implementation", "Implement the User model with SQLAlchemy"),
        ("testing", "Write unit tests for the User model"),
        ("documentation", "Generate API documentation")
    ]
    
    print("\n1. Starting autonomous pipeline...")
    session = auto_session.start_or_resume()
    
    for stage_name, prompt in stages:
        print(f"\n2. Stage: {stage_name}")
        try:
            response = auto_session.ask_with_recovery(prompt)
            print(f"   Completed: {response.content[:100]}...")
            
            # Simulate processing time
            time.sleep(0.5)
            
            # Create checkpoint after each major stage
            if stage_name in ["design", "implementation"]:
                checkpoint = session_mgr.create_checkpoint(session, f"after-{stage_name}")
                print(f"   Checkpoint: {checkpoint}")
                
        except Exception as e:
            print(f"   Error in {stage_name}: {e}")
            print("   Session auto-saved for recovery")
            
    # Export final results
    print("\n3. Exporting pipeline results...")
    markdown_doc = session_mgr.export_session(session, format="markdown")
    
    with open("pipeline_results.md", "w") as f:
        f.write(markdown_doc)
    print("   Exported to pipeline_results.md")
    
    return auto_session


def demo_session_branching():
    """Demonstrate session branching for exploring alternatives."""
    print("\n=== Session Branching Demo ===")
    
    wrapper = ClaudeCodeWrapper()
    session_mgr = SessionManager()
    
    # Start architecture discussion
    print("\n1. Starting architecture discussion...")
    session = wrapper.create_session()
    session.ask("Design a scalable microservices architecture for an e-commerce platform")
    session.ask("Focus on the order processing service")
    
    # Save main session
    session_mgr.save_session(session, tags=["architecture", "main"])
    
    # Create branch to explore alternative approach
    print("\n2. Creating branch to explore event-driven approach...")
    event_branch = session_mgr.branch_session(session, 2, "event-driven")
    event_response = event_branch.ask("Let's redesign this with event sourcing and CQRS")
    print(f"Event-driven approach: {event_response.content[:150]}...")
    
    # Create another branch for monolithic approach
    print("\n3. Creating branch for monolithic comparison...")
    mono_branch = session_mgr.branch_session(session, 2, "monolithic")
    mono_response = mono_branch.ask("What if we used a modular monolith instead?")
    print(f"Monolithic approach: {mono_response.content[:150]}...")
    
    # Save both branches
    session_mgr.save_session(event_branch, tags=["architecture", "event-driven"])
    session_mgr.save_session(mono_branch, tags=["architecture", "monolithic"])
    
    # List all architecture sessions
    print("\n4. All architecture sessions:")
    sessions = session_mgr.list_sessions(tags=["architecture"])
    for s in sessions:
        print(f"   - {s['session_id']}: {s.get('description', 'No description')}")
    
    return session_mgr


def demo_session_recovery():
    """Demonstrate session recovery after interruption."""
    print("\n=== Session Recovery Demo ===")
    
    wrapper = ClaudeCodeWrapper()
    
    # Simulate getting last session ID (would be stored in practice)
    print("\n1. Checking for previous sessions...")
    last_session_id = wrapper.get_last_session_id()
    
    if last_session_id:
        print(f"   Found previous session: {last_session_id}")
        print("   Continuing previous conversation...")
        response = wrapper.continue_conversation("Where were we?")
    else:
        print("   No previous session found, starting new one...")
        response = wrapper.run("Let's build a task automation system")
        
    print(f"Response: {response.content[:150]}...")
    
    return wrapper


def main():
    """Run all demos."""
    print("Claude Code Wrapper - Enhanced Session Management Demo")
    print("=" * 60)
    
    # Run demos
    try:
        # Basic continuation
        wrapper = demo_basic_session_continuation()
        
        # Persistence
        session_mgr, session = demo_session_persistence()
        
        # Autonomous pipeline
        auto_session = demo_autonomous_pipeline()
        
        # Branching
        branch_mgr = demo_session_branching()
        
        # Recovery
        recovery_wrapper = demo_session_recovery()
        
        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- Session continuation with -c flag")
        print("- Session resumption with --resume")
        print("- Session persistence and loading")
        print("- Checkpoints and branching")
        print("- Autonomous pipeline integration")
        print("- Automatic recovery")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()