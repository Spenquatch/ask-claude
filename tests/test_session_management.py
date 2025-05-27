"""
Tests for enhanced session management functionality
"""

import pytest
import json
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_wrapper import (
    ClaudeCodeWrapper, ClaudeCodeConfig, ClaudeCodeResponse, ClaudeCodeSession,
    continue_claude, resume_claude, ask_claude_with_session
)
from session_enhancements import (
    SessionManager, SessionTemplate, AutoRecoverySession
)


class TestSessionContinuation:
    """Test session continuation features."""
    
    def test_continue_conversation(self):
        """Test continuing conversation with -c flag."""
        wrapper = ClaudeCodeWrapper()
        
        # Mock the run method
        with patch.object(wrapper, 'run') as mock_run:
            mock_response = ClaudeCodeResponse(
                content="Continued response",
                returncode=0,
                session_id="test-session-123"
            )
            mock_run.return_value = mock_response
            
            # Test continue_conversation
            response = wrapper.continue_conversation("Continue the discussion")
            
            # Verify continue flag was set
            assert wrapper.config.continue_session == False  # Should be restored
            assert response.content == "Continued response"
            assert wrapper._session_state["last_session_id"] == "test-session-123"
    
    def test_resume_specific_session(self):
        """Test resuming specific session."""
        wrapper = ClaudeCodeWrapper()
        
        with patch.object(wrapper, 'run') as mock_run:
            mock_response = ClaudeCodeResponse(
                content="Resumed response",
                returncode=0,
                session_id="existing-session"
            )
            mock_run.return_value = mock_response
            
            # Test resume_specific_session
            response = wrapper.resume_specific_session("existing-session", "Continue from here")
            
            # Verify session ID was set correctly
            assert response.content == "Resumed response"
            assert wrapper._session_state["last_session_id"] == "existing-session"
    
    def test_command_building_with_continue(self):
        """Test command building with continue flag."""
        config = ClaudeCodeConfig(continue_session=True)
        wrapper = ClaudeCodeWrapper(config)
        
        from claude_code_wrapper import OutputFormat
        cmd = wrapper._build_command("test", OutputFormat.TEXT, config)
        
        assert "--continue" in cmd
        assert "--print" in cmd
    
    def test_command_building_with_resume(self):
        """Test command building with resume."""
        config = ClaudeCodeConfig(session_id="abc-123")
        wrapper = ClaudeCodeWrapper(config)
        
        from claude_code_wrapper import OutputFormat
        cmd = wrapper._build_command("test", OutputFormat.TEXT, config)
        
        assert "--resume" in cmd
        assert "abc-123" in cmd


class TestSessionManager:
    """Test SessionManager functionality."""
    
    @pytest.fixture
    def temp_session_dir(self):
        """Create temporary directory for sessions."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def session_manager(self, temp_session_dir):
        """Create SessionManager with temp directory."""
        return SessionManager(temp_session_dir)
    
    def test_save_and_load_session(self, session_manager):
        """Test saving and loading sessions."""
        # Create a session
        wrapper = ClaudeCodeWrapper()
        session = ClaudeCodeSession(wrapper, session_id="test-save-load")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")
        session.update_metrics(duration=1.5, retries=0)
        
        # Save session
        session_file = session_manager.save_session(
            session,
            tags=["test", "demo"],
            description="Test session"
        )
        
        assert os.path.exists(session_file)
        
        # Load session
        loaded = session_manager.load_session("test-save-load", wrapper)
        
        assert loaded.session_id == "test-save-load"
        assert len(loaded.messages) == 2
        assert loaded.messages[0]["content"] == "Hello"
        assert loaded.messages[1]["content"] == "Hi there!"
    
    def test_list_sessions(self, session_manager):
        """Test listing sessions with filters."""
        wrapper = ClaudeCodeWrapper()
        
        # Create and save multiple sessions
        for i in range(3):
            session = ClaudeCodeSession(wrapper, session_id=f"test-{i}")
            session.add_message("user", f"Message {i}")
            
            tags = ["test"]
            if i == 1:
                tags.append("special")
                
            session_manager.save_session(session, tags=tags)
        
        # List all sessions
        all_sessions = session_manager.list_sessions()
        assert len(all_sessions) == 3
        
        # List with tag filter
        special_sessions = session_manager.list_sessions(tags=["special"])
        assert len(special_sessions) == 1
        assert special_sessions[0]["session_id"] == "test-1"
    
    def test_branch_session(self, session_manager):
        """Test session branching."""
        wrapper = ClaudeCodeWrapper()
        
        # Create original session
        original = ClaudeCodeSession(wrapper, session_id="original")
        original.add_message("user", "Message 1")
        original.add_message("assistant", "Response 1")
        original.add_message("user", "Message 2")
        
        # Create branch at message 2
        branch = session_manager.branch_session(original, 2, "alternative")
        
        assert branch.session_id == "original-alternative"
        assert len(branch.messages) == 2
        assert branch.metadata["branched_from"] == "original"
        assert branch.metadata["branch_point"] == 2
    
    def test_checkpoint_and_restore(self, session_manager):
        """Test checkpoint creation and restoration."""
        wrapper = ClaudeCodeWrapper()
        
        # Create session with some messages
        session = ClaudeCodeSession(wrapper, session_id="checkpoint-test")
        session.add_message("user", "Initial message")
        session.add_message("assistant", "Initial response")
        
        # Create checkpoint
        checkpoint_id = session_manager.create_checkpoint(session, "v1")
        
        # Add more messages
        session.add_message("user", "Additional message")
        
        # Restore checkpoint
        restored = session_manager.restore_checkpoint(checkpoint_id, wrapper)
        
        assert restored.session_id == "checkpoint-test"
        assert len(restored.messages) == 2  # Should have only original messages
        assert "restored_from_checkpoint" in restored.metadata
    
    def test_export_session(self, session_manager):
        """Test session export functionality."""
        wrapper = ClaudeCodeWrapper()
        
        session = ClaudeCodeSession(wrapper, session_id="export-test")
        session.add_message("user", "What is Python?")
        session.add_message("assistant", "Python is a programming language...")
        
        # Export as markdown
        markdown = session_manager.export_session(session, format="markdown")
        assert "# Claude Code Session: export-test" in markdown
        assert "### User" in markdown
        assert "What is Python?" in markdown
        
        # Export as JSON
        json_export = session_manager.export_session(session, format="json")
        data = json.loads(json_export)
        assert data["session_id"] == "export-test"
        assert len(data["messages"]) == 2


class TestSessionTemplates:
    """Test session template functionality."""
    
    def test_create_from_template(self):
        """Test creating session from template."""
        wrapper = ClaudeCodeWrapper()
        
        # Create code review session
        session = SessionTemplate.create_from_template("code_review", wrapper)
        
        assert session.metadata["template"] == "code_review"
        assert len(session.messages) > 0
        assert session.messages[0]["role"] == "system"
    
    def test_all_templates(self):
        """Test all available templates."""
        for template_name in SessionTemplate.TEMPLATES:
            session = SessionTemplate.create_from_template(template_name)
            assert session is not None
            assert "template" in session.metadata
            assert session.metadata["template"] == template_name


class TestAutoRecoverySession:
    """Test automatic recovery session."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_auto_save(self, temp_dir):
        """Test automatic saving."""
        wrapper = ClaudeCodeWrapper()
        session_mgr = SessionManager(temp_dir)
        auto_session = AutoRecoverySession(wrapper, session_mgr, auto_save_interval=2)
        
        # Start session
        session = auto_session.start_or_resume()
        
        # Mock ask method
        with patch.object(session, 'ask') as mock_ask:
            mock_ask.return_value = ClaudeCodeResponse(
                content="Response",
                returncode=0
            )
            
            # Add messages that should trigger auto-save
            auto_session.ask_with_recovery("Message 1")
            auto_session.ask_with_recovery("Message 2")
            
            # Should trigger auto-save after 2 messages
            saved_files = list(Path(temp_dir).glob("*.json"))
            assert len(saved_files) >= 1
    
    def test_error_recovery(self, temp_dir):
        """Test session saving on error."""
        wrapper = ClaudeCodeWrapper()
        session_mgr = SessionManager(temp_dir)
        auto_session = AutoRecoverySession(wrapper, session_mgr)
        
        session = auto_session.start_or_resume()
        
        # Mock ask to raise error
        with patch.object(session, 'ask') as mock_ask:
            mock_ask.side_effect = Exception("Test error")
            
            with pytest.raises(Exception):
                auto_session.ask_with_recovery("This will fail")
            
            # Session should be saved despite error
            saved_files = list(Path(temp_dir).glob("*.json"))
            assert len(saved_files) >= 1


class TestConvenienceFunctions:
    """Test session-aware convenience functions."""
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_continue_claude(self, mock_wrapper_class):
        """Test continue_claude function."""
        mock_wrapper = Mock()
        mock_wrapper.continue_conversation.return_value = ClaudeCodeResponse(
            content="Continued",
            returncode=0
        )
        mock_wrapper_class.return_value = mock_wrapper
        
        response = continue_claude()
        assert response.content == "Continued"
        mock_wrapper.continue_conversation.assert_called_once()
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_resume_claude(self, mock_wrapper_class):
        """Test resume_claude function."""
        mock_wrapper = Mock()
        mock_wrapper.resume_specific_session.return_value = ClaudeCodeResponse(
            content="Resumed",
            returncode=0
        )
        mock_wrapper_class.return_value = mock_wrapper
        
        response = resume_claude("session-123", "Continue from here")
        assert response.content == "Resumed"
        mock_wrapper.resume_specific_session.assert_called_with("session-123", "Continue from here")
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_ask_claude_with_session(self, mock_wrapper_class):
        """Test ask_claude_with_session function."""
        mock_wrapper = Mock()
        mock_wrapper.run.return_value = ClaudeCodeResponse(
            content="Response",
            returncode=0
        )
        mock_wrapper_class.return_value = mock_wrapper
        
        # Test with session ID
        response = ask_claude_with_session("Hello", session_id="test-123")
        assert response.content == "Response"
        
        # Verify config was set correctly
        config_call = mock_wrapper_class.call_args[0][0]
        assert config_call.session_id == "test-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])