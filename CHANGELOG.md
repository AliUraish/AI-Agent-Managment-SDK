# Changelog

## [1.1.0] - 2025-01-30

### Changed - Lighter SDK (Session Tracking Removal)

#### Removed Local Session Storage
- ❌ Removed `_active_sessions` dictionary from both trackers
- ❌ Removed local session state management
- ❌ Removed `get_active_sessions_count()` method
- ❌ Removed `is_session_active()` method

#### Updated Method Signatures
- 🔄 `end_conversation()` now requires `agent_id` parameter
- 🔄 `record_failed_session()` now requires `agent_id` parameter

#### Added Backend Session Management
- ✅ Added `get_session_info(session_id)` method
- ✅ Added `get_active_conversations(agent_id)` method
- ✅ Both sync and async versions available

#### Benefits
- 🚀 **Lighter Memory Usage**: No local session storage
- 🚀 **Stateless SDK**: Fully relies on backend for session management
- 🚀 **Better Scalability**: No memory constraints for large session volumes
- 🚀 **Simplified Logic**: All session logic handled by backend

#### Migration Guide

**Before (v1.0.0):**
```python
# Sessions were tracked locally
session_id = perf_tracker.start_conversation("agent_001")
perf_tracker.end_conversation(session_id, quality_score=5)  # agent_id was optional
```

**After (v1.1.0):**
```python
# Sessions managed entirely by backend
session_id = perf_tracker.start_conversation("agent_001")
perf_tracker.end_conversation(session_id, "agent_001", quality_score=5)  # agent_id required
```

#### New API Endpoints Expected by Backend
- `GET /conversations/{session_id}` - Get session information
- `GET /conversations/active` - Get active conversations
- `GET /conversations/active?agent_id={agent_id}` - Get active conversations for specific agent

---

## [1.0.0] - 2025-01-29

### Initial Release

#### Features
- Agent Operations Tracker (Active agents, Status, Activity)
- Agent Performance Tracker (Success rates, Response times, Quality, Failures)
- Secure API key handling
- Async/Sync support
- Comprehensive logging
- Thread-safe operations 