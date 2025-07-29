# Changelog

## [1.2.0] - 2025-01-30

### Added - Lightweight Session Tracking with TTL

#### Smart Session Management
- ✅ **Custom Session ID Format**: `{agent_id}_{timestamp}_{random}` 
- ✅ **Lightweight Session Cache**: Minimal in-memory tracking for validation
- ✅ **TTL-based Cleanup**: Automatic expiration and cleanup (default: 10 hours)
- ✅ **Backend Notification**: Notifies backend about expired sessions

#### New Features
- 🔧 `set_session_ttl(hours)` - Change TTL dynamically
- 📊 `get_session_stats()` - Session cache statistics
- 🧹 Automatic background cleanup with configurable intervals
- ⚡ Smart session validation with graceful fallback

#### Updated Method Signatures
- 🔄 `start_conversation()` - No longer accepts `session_id` parameter (auto-generated)
- 🔄 `end_conversation()` - No longer requires `agent_id` parameter (extracted from session)
- 🔄 `record_failed_session()` - No longer requires `agent_id` parameter

#### Benefits
- 🎯 **Session Validation**: Ensures conversations are properly tracked
- 🧹 **Automatic Cleanup**: Prevents memory leaks from abandoned sessions
- 📍 **Fallback Parsing**: Can extract agent_id from session_id if cache miss
- 🔄 **Backend Sync**: Notifies backend about session lifecycle events

#### New API Endpoints
- `POST /conversations/expired` - Notification about expired sessions

#### Configuration Options
```python
perf_tracker = AgentPerformanceTracker(
    base_url="https://api.example.com",
    session_ttl_hours=10.0,        # Session expiry time
    cleanup_interval_minutes=30    # Cleanup frequency
)
```

#### Usage Example
```python
# Start conversation (session_id auto-generated)
session_id = perf_tracker.start_conversation("agent_001", user_id="user_123")
# Returns: "agent_001_1674123456_a1b2c3d4"

# End conversation (agent_id auto-extracted)
perf_tracker.end_conversation(session_id, quality_score=ConversationQuality.GOOD)

# Check session cache stats
stats = perf_tracker.get_session_stats()
# Returns: {"total_cached_sessions": 5, "active_sessions": 4, "ttl_hours": 10.0}
```

---

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