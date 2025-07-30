# Changelog

## [1.2.1] - 2025-01-30

### Added - Hybrid Session Management System

#### 🔄 Revolutionary Hybrid Model
- ✅ **Dual TTL System**: 10-hour local sliding TTL + 20-hour backend persistence
- ✅ **Seamless Resumption**: Automatic session recovery across SDK crashes/restarts
- ✅ **Backend Fallback**: When local cache expires, automatically retrieve from backend
- ✅ **Crash Resilience**: Sessions survive SDK shutdowns through backend persistence
- ✅ **Single Source of Truth**: Backend maintains authoritative session state

#### 🏗️ Session Lifecycle Architecture
```
1. start_conversation() → Creates in local cache + backend
2. Active usage → Sliding TTL resets on each access (local)
3. Local TTL expires → Automatic backend retrieval if within 20h
4. Seamless resumption → Context preserved, no interruption
5. Backend TTL expires → Session permanently expired (20h)
```

#### 🔧 Enhanced SessionInfo (Lightweight Cache)
```python
@dataclass
class SessionInfo:
    agent_id: str
    start_time: datetime
    run_id: str              # 🔑 New: Unique conversation identifier
    user_id: Optional[str]
    last_access_time: datetime  # For sliding TTL
    is_ended: bool = False   # 🔑 New: Track completion status
```

#### 🛠️ New Hybrid Methods
- 🔧 `resume_conversation(session_id, agent_id, context)` - Explicit resumption with context
- 🔧 `resume_conversation_async()` - Async version of resumption
- 🔍 `_get_session_with_fallback()` - Hybrid cache/backend retrieval
- 🔍 `_retrieve_session_from_backend()` - Backend session recovery
- 🛡️ `_handle_session_error()` - Enhanced error handling with fallbacks
- ✅ `_validate_session_data()` - Backend data validation
- 🔄 `_create_fallback_session_info()` - Graceful degradation

#### 📊 New Data Structures
```python
@dataclass
class ConversationStartData:
    agent_id: str
    session_id: str
    run_id: str
    context: Optional[Dict] = None  # 🔑 For session resumption

@dataclass
class FailedSessionData:
    agent_id: str
    session_id: str 
    run_id: str
    error_message: str
    failure_time: str
    
@dataclass
class SessionRetrievalQuery:
    session_id: str
    include_context: bool = True
    include_history: bool = False
```

#### 🎯 Enhanced Session Statistics
```python
stats = perf_tracker.get_session_stats()
# Returns:
{
    'total_cached_sessions': 3,
    'active_sessions': 2,
    'ended_sessions': 1,
    'local_ttl_hours': 10.0,
    'backend_ttl_hours': 20.0,       # 🔑 New
    'hybrid_model_enabled': True,    # 🔑 New
    'backend_expired_sessions': 0,   # 🔑 New
    'avg_idle_time_hours': 2.1,
    'longest_idle_time_hours': 5.3
}
```

#### 🌐 New API Endpoints
- `GET /conversations/{session_id}` - Retrieve session from backend
- `POST /conversations/resume` - Resume conversation with context
- `POST /conversations/local-expired` - Notify about local cache expiry

#### 🔄 Updated Method Behavior
**Seamless Backend Integration**:
- `end_conversation()` - Now uses hybrid fallback for session lookup
- `record_failed_session()` - Hybrid fallback + enhanced error handling
- `is_session_active()` - Checks local cache → backend → dual TTL validation
- `get_session_ttl_remaining()` - Considers both local and backend TTL

#### 💡 Usage Examples

**Basic Hybrid Session**:
```python
# Start conversation (stored locally + backend)
session_id = perf_tracker.start_conversation("agent_001", user_id="user_123")

# SDK crashes/restarts here...

# Later: Check session (automatically retrieves from backend)
is_active = perf_tracker.is_session_active(session_id)  # ✅ True (seamlessly resumed)
```

**Explicit Resumption with Context**:
```python
# Resume with context after local cache expires
success = perf_tracker.resume_conversation(
    session_id=old_session_id,
    agent_id="agent_001", 
    user_id="user_123",
    context={"previous_topic": "billing", "step": 3},
    metadata={"resumption_reason": "cache_expiry"}
)
```

**Crash Recovery Simulation**:
```python
# Clear local cache (simulate crash)
with perf_tracker._cache_lock:
    perf_tracker._session_cache.clear()

# Session access still works (retrieved from backend)
active = perf_tracker.is_session_active(session_id)  # ✅ True
```

#### 🎯 Benefits
- 🚀 **Zero-Downtime Recovery**: Sessions survive crashes and restarts
- 💾 **Memory Efficient**: Local cache only stores lightweight session data
- 🔄 **Automatic Fallback**: No manual intervention required for session recovery
- 📊 **Rich Analytics**: Comprehensive session health monitoring
- ⚡ **Performance Optimized**: Local cache for speed, backend for persistence
- 🛡️ **Error Resilient**: Graceful degradation with multiple fallback strategies

#### 🔧 Configuration
```python
perf_tracker = AgentPerformanceTracker(
    base_url="https://api.example.com",
    session_ttl_hours=10.0,        # Local cache sliding TTL
    backend_ttl_hours=20.0,        # Backend persistence TTL
    cleanup_interval_minutes=30    # Cache cleanup frequency
)
```

---

## [1.3.0] - 2025-01-30

### Added - Sliding TTL (Smart Session Lifetime Management)

#### 🔄 Sliding TTL Implementation
- ✅ **Session Access Tracking**: `last_access_time` field in `SessionInfo`
- ✅ **Automatic TTL Reset**: Every session access resets the expiry timer
- ✅ **Smart Expiry Logic**: Sessions expire based on last access, not creation time
- ✅ **Active Session Protection**: Prevents expiry of actively used sessions

#### 🛠️ New Methods
- 🔧 `touch_session(session_id)` - Manually reset session TTL
- 🔍 `is_session_active(session_id)` - Check if session is active (with TTL reset)
- ⏱️ `get_session_ttl_remaining(session_id)` - Get remaining TTL in hours
- 📊 Enhanced `get_session_stats()` with idle time analytics

#### 📈 Enhanced Session Statistics
```python
stats = perf_tracker.get_session_stats()
# Returns:
{
    'total_cached_sessions': 5,
    'active_sessions': 4,
    'sliding_ttl_enabled': True,
    'avg_idle_time_hours': 2.5,
    'longest_idle_time_hours': 8.2,
    'shortest_idle_time_hours': 0.1,
    'ttl_hours': 10.0
}
```

#### 🎯 Automatic TTL Reset Triggers
- ✅ `end_conversation()` - TTL reset on conversation end
- ✅ `record_failed_session()` - TTL reset on failure recording
- ✅ `is_session_active()` - TTL reset on status check
- ✅ `get_session_ttl_remaining()` - TTL reset on remaining time query
- ✅ `touch_session()` - Manual TTL reset

#### 🔄 Sliding TTL Behavior
**Regular TTL**: Session expires X hours after creation
```
Session Created → [10 hours] → Expired (regardless of usage)
```

**Sliding TTL**: Session expires X hours after last access
```
Session Created → Access → [TTL Reset] → Access → [TTL Reset] → ... → [10h idle] → Expired
```

#### 💡 Usage Examples

**Manual Session Touch**:
```python
# Keep session alive manually
perf_tracker.touch_session(session_id)
```

**Check Session Status**:
```python
# Automatically resets TTL on check
is_alive = perf_tracker.is_session_active(session_id)
remaining = perf_tracker.get_session_ttl_remaining(session_id)
```

**Session Analytics**:
```python
stats = perf_tracker.get_session_stats()
print(f"Average idle time: {stats['avg_idle_time_hours']} hours")
print(f"Longest idle session: {stats['longest_idle_time_hours']} hours")
```

#### 🎯 Benefits
- 🚀 **Intelligent Cleanup**: Only expires truly inactive sessions
- 💾 **Memory Efficiency**: Active sessions don't consume unnecessary memory
- 🔄 **Automatic Management**: No manual intervention needed
- 📊 **Rich Analytics**: Detailed session activity insights
- ⚡ **Performance Optimized**: Minimal overhead for active conversations

#### 🔧 Configuration
```python
perf_tracker = AgentPerformanceTracker(
    base_url="https://api.example.com",
    session_ttl_hours=10.0,        # Sliding TTL window
    cleanup_interval_minutes=30    # Background cleanup frequency
)
```

---

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