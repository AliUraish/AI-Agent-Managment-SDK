"""OpenAI instrumentation 

Tracks (read-only):
- Model + response IDs
- Conversation: gen_ai.prompt.{i}.role/content; gen_ai.completion.{i}.role/content/finish_reason
- Tools: request tool defs; response tool calls (and child tool spans)
- Usage: gen_ai.usage.prompt_tokens, gen_ai.usage.completion_tokens, gen_ai.usage.total_tokens
- Streaming: gen_ai.streaming.time_to_first_token, gen_ai.streaming.time_to_generate, gen_ai.streaming.chunk_count

Does NOT record control parameters (temperature, top_p, max_tokens, penalties).

Usage:
    from MYSDK import instrument_openai
    instrument_openai()
"""

from __future__ import annotations

import json
import time
import inspect
from typing import Any, Dict, Iterable, Optional, Tuple, Callable, AsyncIterable

from wrapt import wrap_function_wrapper
from opentelemetry.trace import get_tracer, SpanKind, Status, StatusCode
from opentelemetry.instrumentation.utils import unwrap as otel_unwrap

from .config import maybe_redact
from .metrics import (
    record_token_usage,
    record_streaming_metrics,
    record_choice_count,
    record_embedding_vector_size,
    record_operation_exception,
    record_message_count,
    record_tool_call_count,
    record_tool_call_event,
    record_message_event,
)


TRACER = get_tracer("llmtracker.openai")


class Attr:
    # System
    SYSTEM = "gen_ai.system"
    # Request
    REQ_MODEL = "gen_ai.request.model"
    REQ_HEADERS = "gen_ai.request.headers"
    REQ_STREAMING = "gen_ai.request.streaming"
    REQ_FUNCTIONS = "gen_ai.request.tools"
    # Prompts/Completions
    PROMPT = "gen_ai.prompt"
    COMPLETION = "gen_ai.completion"
    # Response
    RESP_ID = "gen_ai.response.id"
    RESP_MODEL = "gen_ai.response.model"
    RESP_FINISH = "gen_ai.response.finish_reason"
    RESP_STOP = "gen_ai.response.stop_reason"
    RESP_FINGERPRINT = "gen_ai.response.system_fingerprint"
    # Usage
    USE_PROMPT = "gen_ai.usage.prompt_tokens"
    USE_COMP = "gen_ai.usage.completion_tokens"
    USE_TOTAL = "gen_ai.usage.total_tokens"
    USE_REASON = "gen_ai.usage.reasoning_tokens"
    USE_CACHE_CREATE = "gen_ai.usage.cache_creation_input_tokens"
    USE_CACHE_READ = "gen_ai.usage.cache_read_input_tokens"
    # Streaming
    STREAM_TTFB = "gen_ai.streaming.time_to_first_token"
    STREAM_TTG = "gen_ai.streaming.time_to_generate"
    STREAM_CHUNKS = "gen_ai.streaming.chunk_count"


def _safe_set(span, key: str, value: Any, max_len: int = 2000):
    if value is None:
        return
    try:
        if isinstance(value, str) and len(value) > max_len:
            value = value[: max_len - 3] + "..."
        span.set_attribute(key, value)
    except Exception:
        pass


def _model_as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()  # pydantic
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except Exception:
            pass
    return {}


def _extract_usage_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "__dict__"):
        try:
            return dict(usage.__dict__)
        except Exception:
            return {}
    return {}


def _set_usage(span, usage: Dict[str, Any]):
    if not usage:
        return
    # Support both Chat/Completions and Responses API schemas
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    if pt is None and "input_tokens" in usage:
        pt = usage.get("input_tokens")
    if ct is None and "output_tokens" in usage:
        ct = usage.get("output_tokens")
    if pt is not None:
        _safe_set(span, Attr.USE_PROMPT, pt)
    if ct is not None:
        _safe_set(span, Attr.USE_COMP, ct)
    if "total_tokens" in usage:
        _safe_set(span, Attr.USE_TOTAL, usage["total_tokens"])
    elif isinstance(pt, int) and isinstance(ct, int):
        _safe_set(span, Attr.USE_TOTAL, pt + ct)
    # Optional details
    out_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details")
    if isinstance(out_details, dict) and "reasoning_tokens" in out_details:
        _safe_set(span, Attr.USE_REASON, out_details["reasoning_tokens"])
    # Cache details
    if "cache_creation_input_tokens" in usage:
        _safe_set(span, Attr.USE_CACHE_CREATE, usage["cache_creation_input_tokens"])
    if "cache_read_input_tokens" in usage:
        _safe_set(span, Attr.USE_CACHE_READ, usage["cache_read_input_tokens"])
    # Metrics
    try:
        record_token_usage(pt if isinstance(pt, int) else None, ct if isinstance(ct, int) else None)
    except Exception:
        pass


def _set_prompts_from_messages(span, messages: Any):
    try:
        if isinstance(messages, list):
            try:
                _safe_set(span, "gen_ai.prompt.count", len(messages))
                record_message_count(len(messages))
            except Exception:
                pass
        for i, msg in enumerate(messages or []):
            prefix = f"{Attr.PROMPT}.{i}"
            role = None
            content = None
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
            else:
                d = _model_as_dict(msg)
                role = d.get("role")
                content = d.get("content")
            try:
                record_message_event(role=role or "user", provider="OpenAI")
            except Exception:
                pass
            if isinstance(content, list):
                content = json.dumps(content)
            content = maybe_redact(content)
            if role is not None:
                _safe_set(span, f"{prefix}.role", role)
            if content is not None:
                _safe_set(span, f"{prefix}.content", content)
    except Exception:
        pass


def _set_functions(span, tools_or_functions: Any):
    try:
        for i, item in enumerate(tools_or_functions or []):
            # OpenAI tools can be {"type": "function", "function": {...}}
            fn = item.get("function") if isinstance(item, dict) else None
            if not fn and isinstance(item, dict):
                fn = item  # legacy functions=[{name,description,parameters}]
            if not fn:
                continue
            prefix = f"{Attr.REQ_FUNCTIONS}.{i}"
            _safe_set(span, f"{prefix}.name", fn.get("name"))
            _safe_set(span, f"{prefix}.description", fn.get("description"))
            params = fn.get("parameters")
            if params is not None:
                _safe_set(span, f"{prefix}.parameters", json.dumps(params))
    except Exception:
        pass


def _set_completions_and_tools(span, response_dict: Dict[str, Any]):
    choices = response_dict.get("choices")
    if not isinstance(choices, list):
        return
    # Record choice count metric (non-stream)
    try:
        record_choice_count(len(choices))
        _safe_set(span, "gen_ai.completion.choice_count", len(choices))
    except Exception:
        pass
    total_tool_calls = 0
    for choice in choices:
        idx = choice.get("index", 0)
        prefix = f"{Attr.COMPLETION}.{idx}"
        finish = choice.get("finish_reason")
        if finish is not None:
            _safe_set(span, f"{prefix}.finish_reason", finish)
            _safe_set(span, Attr.RESP_FINISH, finish)
        # Content filter results (best-effort JSON)
        if isinstance(choice.get("content_filter_results"), (dict, list)):
            try:
                _safe_set(span, f"{prefix}.content_filter_results", json.dumps(choice.get("content_filter_results")))
            except Exception:
                pass
        msg = choice.get("message")
        if isinstance(msg, dict):
            role = msg.get("role")
            if role is not None:
                _safe_set(span, f"{prefix}.role", role)
            content = msg.get("content")
            if content is not None:
                red = maybe_redact(content)
                if red is not None:
                    _safe_set(span, f"{prefix}.content", red)
            # Refusal (if present)
            if msg.get("refusal") is not None:
                _safe_set(span, f"{prefix}.refusal", msg.get("refusal"))
            # Tool calls in message
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for j, tc in enumerate(tool_calls):
                    fun = (tc or {}).get("function", {})
                    _safe_set(span, f"{prefix}.tool_calls.{j}.id", tc.get("id"))
                    _safe_set(span, f"{prefix}.tool_calls.{j}.type", tc.get("type"))
                    _safe_set(span, f"{prefix}.tool_calls.{j}.name", fun.get("name"))
                    _safe_set(span, f"{prefix}.tool_calls.{j}.arguments", fun.get("arguments"))
                    try:
                        record_tool_call_event(tool_name=fun.get("name"), provider="OpenAI")
                    except Exception:
                        pass
                    total_tool_calls += 1
                    # Child tool span
                    with TRACER.start_as_current_span(
                        name=f"tool_call.{fun.get('name','function')}", kind=SpanKind.INTERNAL
                    ) as tool_span:
                        _safe_set(tool_span, Attr.SYSTEM, "OpenAI")
                        _safe_set(tool_span, f"{Attr.REQ_FUNCTIONS}.0.name", fun.get("name"))
                        _safe_set(tool_span, f"{Attr.REQ_FUNCTIONS}.0.arguments", fun.get("arguments"))
    try:
        if total_tool_calls > 0:
            _safe_set(span, "gen_ai.completion.tool_call_count", total_tool_calls)
            record_tool_call_count(total_tool_calls)
    except Exception:
        pass


def _extract_text_from_output_content(content: Any) -> str:
    """Aggregate text from Responses API output content items.

    Handles list formats like [{"type": "text"|"output_text", "text": "..."}, ...]
    and simple string content.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    try:
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # Common fields across SDK variants
                    if item.get("type") in ("text", "output_text") and isinstance(item.get("text"), str):
                        parts.append(item.get("text", ""))
                else:
                    d = _model_as_dict(item)
                    if d.get("type") in ("text", "output_text") and isinstance(d.get("text"), str):
                        parts.append(d.get("text", ""))
    except Exception:
        pass
    return "".join(parts)


def _set_prompts_from_responses_input(span, inputs: Any):
    try:
        for i, msg in enumerate(inputs or []):
            role = None
            content = None
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
            else:
                d = _model_as_dict(msg)
                role = d.get("role")
                content = d.get("content")
            try:
                record_message_event(role=role or "user", provider="OpenAI")
            except Exception:
                pass
            text = _extract_text_from_output_content(content)
            text = maybe_redact(text)
            if role is not None:
                _safe_set(span, f"{Attr.PROMPT}.{i}.role", role)
            if text:
                _safe_set(span, f"{Attr.PROMPT}.{i}.content", text)
    except Exception:
        pass


def _handle_responses_return(args: Optional[Tuple], kwargs: Optional[Dict], return_value: Any, span) -> None:
    """Specialized extraction for OpenAI Responses API results.

    Populates completion content and tool calls from `output` items, maps
    usage input/output tokens, captures stop/system fingerprint, and handles
    optional reasoning included in function_call arguments.
    """
    try:
        rd = _model_as_dict(return_value)
        if not isinstance(rd, dict) or not rd:
            return

        # System fingerprint (if present)
        if rd.get("system_fingerprint") is not None:
            _safe_set(span, Attr.RESP_FINGERPRINT, rd.get("system_fingerprint"))

        # Stop/finish reason
        if rd.get("stop_reason") is not None:
            _safe_set(span, Attr.RESP_STOP, rd.get("stop_reason"))
            _safe_set(span, Attr.RESP_FINISH, rd.get("stop_reason"))
        if rd.get("finish_reason") is not None:
            _safe_set(span, Attr.RESP_FINISH, rd.get("finish_reason"))

        # Usage mapping: input/output tokens
        usage = _extract_usage_dict(rd.get("usage"))
        if usage:
            # Mirror _set_usage behavior with explicit mapping here
            pt = usage.get("prompt_tokens") or usage.get("input_tokens")
            ct = usage.get("completion_tokens") or usage.get("output_tokens")
            if pt is not None:
                _safe_set(span, Attr.USE_PROMPT, pt)
            if ct is not None:
                _safe_set(span, Attr.USE_COMP, ct)
            total = usage.get("total_tokens")
            if total is None and isinstance(pt, int) and isinstance(ct, int):
                total = pt + ct
            if total is not None:
                _safe_set(span, Attr.USE_TOTAL, total)
            try:
                record_token_usage(pt if isinstance(pt, int) else None, ct if isinstance(ct, int) else None)
            except Exception:
                pass

        # Inputs as prompts (Responses uses `input`)
        if isinstance(kwargs, dict) and "input" in kwargs:
            inputs = kwargs.get("input")
            try:
                if isinstance(inputs, list):
                    _safe_set(span, "gen_ai.prompt.count", len(inputs))
                    record_message_count(len(inputs))
            except Exception:
                pass
            _set_prompts_from_responses_input(span, inputs)

        # Output items -> completions and tool_calls
        completion_idx = 0
        tool_call_count = 0
        output = rd.get("output")
        if isinstance(output, list):
            for i, item in enumerate(output):
                # Item may be dict or object; normalize
                d = item if isinstance(item, dict) else _model_as_dict(item)
                if not isinstance(d, dict):
                    continue
                item_type = d.get("type")

                if item_type == "message":
                    text = _extract_text_from_output_content(d.get("content"))
                    red = maybe_redact(text)
                    if red:
                        _safe_set(span, f"{Attr.COMPLETION}.{completion_idx}.content", red)
                    _safe_set(span, f"{Attr.COMPLETION}.{completion_idx}.role", "assistant")
                    completion_idx += 1

                elif item_type == "function_call":
                    args_str = d.get("arguments", "")
                    # Optional: some models include reasoning inside arguments JSON
                    if isinstance(args_str, str) and args_str:
                        try:
                            args_json = json.loads(args_str)
                            reasoning = args_json.get("reasoning")
                            if isinstance(reasoning, str) and reasoning:
                                red = maybe_redact(reasoning)
                                if red:
                                    _safe_set(
                                        span,
                                        f"{Attr.COMPLETION}.{completion_idx}.content",
                                        red,
                                    )
                                    _safe_set(span, f"{Attr.COMPLETION}.{completion_idx}.role", "assistant")
                                    completion_idx += 1
                        except Exception:
                            pass

                    # Record the tool call itself
                    _safe_set(span, f"{Attr.COMPLETION}.{i}.tool_calls.0.id", d.get("id"))
                    _safe_set(span, f"{Attr.COMPLETION}.{i}.tool_calls.0.type", "function")
                    _safe_set(span, f"{Attr.COMPLETION}.{i}.tool_calls.0.name", d.get("name"))
                    if args_str:
                        _safe_set(span, f"{Attr.COMPLETION}.{i}.tool_calls.0.arguments", args_str)
                    try:
                        record_tool_call_event(tool_name=d.get("name"), provider="OpenAI")
                    except Exception:
                        pass
                    tool_call_count += 1
                    # Create a child span for this tool call
                    try:
                        with TRACER.start_as_current_span(
                            name=f"tool_call.{d.get('name') or 'function'}", kind=SpanKind.INTERNAL
                        ) as tool_span:
                            _safe_set(tool_span, Attr.SYSTEM, "OpenAI")
                            _safe_set(tool_span, f"{Attr.REQ_FUNCTIONS}.0.name", d.get("name"))
                            if args_str:
                                _safe_set(tool_span, f"{Attr.REQ_FUNCTIONS}.0.arguments", args_str)
                    except Exception:
                        pass

                elif item_type == "reasoning":
                    # Some Responses provide reasoning summaries
                    summary = d.get("summary")
                    if isinstance(summary, str) and summary:
                        red = maybe_redact(summary)
                        if red:
                            _safe_set(span, f"{Attr.COMPLETION}.{completion_idx}.content", red)
                            _safe_set(span, f"{Attr.COMPLETION}.{completion_idx}.role", "assistant")
                            completion_idx += 1

        # Record tool call count
        try:
            if tool_call_count > 0:
                _safe_set(span, "gen_ai.completion.tool_call_count", tool_call_count)
                record_tool_call_count(tool_call_count)
        except Exception:
            pass

        # Fallback: output_text
        if completion_idx == 0 and isinstance(rd.get("output_text"), str):
            red = maybe_redact(rd.get("output_text"))
            if red:
                _safe_set(span, f"{Attr.COMPLETION}.0.content", red)
                _safe_set(span, f"{Attr.COMPLETION}.0.role", "assistant")
    except Exception:
        # Be tolerant — this is best-effort enrichment
        pass


def _handle_openai_return(span, return_value: Any):
    # Convert to dict when possible
    rd = _model_as_dict(return_value)
    if not rd and isinstance(return_value, (str, bytes)):
        return  # nothing to parse
    if not rd and hasattr(return_value, "__iter__") and not isinstance(return_value, (dict, list)):
        # likely a stream; handled by streaming wrapper
        return
    # IDs/models
    if "id" in rd:
        _safe_set(span, Attr.RESP_ID, rd.get("id"))
    if "model" in rd:
        _safe_set(span, Attr.RESP_MODEL, rd.get("model"))
    if "system_fingerprint" in rd and rd.get("system_fingerprint") is not None:
        _safe_set(span, Attr.RESP_FINGERPRINT, rd.get("system_fingerprint"))
    # Stop/finish reasons at top-level (Responses-like)
    if rd.get("stop_reason") is not None:
        _safe_set(span, Attr.RESP_STOP, rd.get("stop_reason"))
        _safe_set(span, Attr.RESP_FINISH, rd.get("stop_reason"))
    if rd.get("finish_reason") is not None:
        _safe_set(span, Attr.RESP_FINISH, rd.get("finish_reason"))
    # Usage
    usage = _extract_usage_dict(rd.get("usage"))
    _set_usage(span, usage)
    # Choices -> completions + tool calls
    _set_completions_and_tools(span, rd)
    # Embeddings vector size (first vector)
    try:
        data = rd.get("data")
        if isinstance(data, list) and data:
            first = data[0] if isinstance(data[0], dict) else _model_as_dict(data[0])
            if isinstance(first, dict):
                vec = first.get("embedding")
                if isinstance(vec, (list, tuple)):
                    size = len(vec)
                    _safe_set(span, "gen_ai.embeddings.vector_size", size)
                    try:
                        record_embedding_vector_size(size)
                    except Exception:
                        pass
    except Exception:
        pass


def _wrap_method(trace_name: str, handler, is_async: bool = False, stream_parser: Optional[Callable[[Any], Dict[str, Any]]] = None):
    def sync_wrapper(wrapped, instance, args, kwargs):
        attributes = {Attr.SYSTEM: "OpenAI"}
        # Request basics (no control params)
        model = kwargs.get("model") if isinstance(kwargs, dict) else None
        headers = (kwargs or {}).get("extra_headers") or (kwargs or {}).get("headers")
        stream = bool((kwargs or {}).get("stream", False))
        with TRACER.start_as_current_span(trace_name, kind=SpanKind.CLIENT, attributes=attributes) as span:
            _safe_set(span, Attr.REQ_MODEL, model)
            _safe_set(span, Attr.REQ_STREAMING, stream)
            if headers is not None:
                _safe_set(span, Attr.REQ_HEADERS, str(headers))
            # Messages & tools/functions
            if isinstance(kwargs, dict):
                if "messages" in kwargs:
                    _set_prompts_from_messages(span, kwargs.get("messages"))
                if "functions" in kwargs:
                    _set_functions(span, kwargs.get("functions"))
                if "tools" in kwargs:
                    _set_functions(span, kwargs.get("tools"))
            # Call underlying
            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                try:
                    tn = trace_name.lower()
                    if "embeddings" in tn:
                        record_operation_exception("embeddings")
                    elif "images" in tn:
                        record_operation_exception("images")
                except Exception:
                    pass
                raise

            # Streaming handling
            if stream and _is_stream_like(result):
                return _wrap_stream_result(span, result, stream_parser)

            # Non-stream: extract response attrs
            try:
                handler(args, kwargs, result, span)
                _handle_openai_return(span, result)
                span.set_status(Status(StatusCode.OK))
            finally:
                pass
            return result

    async def async_wrapper(wrapped, instance, args, kwargs):
        attributes = {Attr.SYSTEM: "OpenAI"}
        model = kwargs.get("model") if isinstance(kwargs, dict) else None
        headers = (kwargs or {}).get("extra_headers") or (kwargs or {}).get("headers")
        stream = bool((kwargs or {}).get("stream", False))
        with TRACER.start_as_current_span(trace_name, kind=SpanKind.CLIENT, attributes=attributes) as span:
            _safe_set(span, Attr.REQ_MODEL, model)
            _safe_set(span, Attr.REQ_STREAMING, stream)
            if headers is not None:
                _safe_set(span, Attr.REQ_HEADERS, str(headers))
            if isinstance(kwargs, dict):
                if "messages" in kwargs:
                    _set_prompts_from_messages(span, kwargs.get("messages"))
                if "functions" in kwargs:
                    _set_functions(span, kwargs.get("functions"))
                if "tools" in kwargs:
                    _set_functions(span, kwargs.get("tools"))
            try:
                result = await wrapped(*args, **kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                try:
                    tn = trace_name.lower()
                    if "embeddings" in tn:
                        record_operation_exception("embeddings")
                    elif "images" in tn:
                        record_operation_exception("images")
                except Exception:
                    pass
                raise

            # Streaming handling for async
            if stream and _is_async_stream_like(result):
                return _wrap_async_stream_result(span, result, stream_parser)

            try:
                handler(args, kwargs, result, span)
                _handle_openai_return(span, result)
                span.set_status(Status(StatusCode.OK))
            finally:
                pass
            return result

    return async_wrapper if is_async else sync_wrapper


def _is_stream_like(obj: Any) -> bool:
    return hasattr(obj, "__iter__") and not isinstance(obj, (dict, list, str, bytes))


def _is_async_stream_like(obj: Any) -> bool:
    # Avoid isinstance with typing.AsyncIterable; rely on protocol methods
    return inspect.isasyncgen(obj) or hasattr(obj, "__aiter__")


def _wrap_stream_result(span, stream_obj: Iterable[Any], parser: Optional[Callable[[Any], Dict[str, Any]]] = None):
    start_time = time.time()
    first_token_time: Optional[float] = None
    chunk_count = 0
    finish_reason: Optional[str] = None
    stop_reason: Optional[str] = None
    buffer_text: list[str] = []
    run_status: Optional[str] = None

    def generator():
        nonlocal first_token_time, chunk_count, finish_reason, stop_reason, buffer_text, run_status
        try:
            for chunk in stream_obj:
                chunk_count += 1
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                    _safe_set(span, Attr.STREAM_TTFB, first_token_time)
                # Parse streaming events for finish/stop/model/id/fingerprint
                if parser is not None:
                    try:
                        info = parser(chunk) or {}
                        if info.get("id") is not None:
                            _safe_set(span, Attr.RESP_ID, info.get("id"))
                        if info.get("model") is not None:
                            _safe_set(span, Attr.RESP_MODEL, info.get("model"))
                        if info.get("system_fingerprint") is not None:
                            _safe_set(span, Attr.RESP_FINGERPRINT, info.get("system_fingerprint"))
                        # Track last seen reasons
                        if info.get("finish_reason"):
                            finish_reason = info.get("finish_reason")
                        if info.get("stop_reason"):
                            stop_reason = info.get("stop_reason")
                        # Chat/Responses: accumulate assistant text
                        if isinstance(info.get("delta_text"), str) and info.get("delta_text"):
                            buffer_text.append(info.get("delta_text"))
                        # Runs: track status and ids
                        if info.get("run_id"):
                            _safe_set(span, "gen_ai.run.id", info.get("run_id"))
                        if info.get("status"):
                            run_status = info.get("status")
                            _safe_set(span, "gen_ai.run.status", run_status)
                        if info.get("thread_id"):
                            _safe_set(span, "gen_ai.thread.id", info.get("thread_id"))
                        if info.get("assistant_id"):
                            _safe_set(span, "gen_ai.assistant.id", info.get("assistant_id"))
                    except Exception:
                        pass
                yield chunk
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            # Finalize streaming metrics
            if first_token_time is not None:
                total_time = time.time() - start_time
                _safe_set(span, Attr.STREAM_TTG, max(total_time - first_token_time, 0.0))
            _safe_set(span, Attr.STREAM_CHUNKS, chunk_count)
            # Apply finish/stop reasons if available
            if finish_reason:
                _safe_set(span, Attr.RESP_FINISH, finish_reason)
                _safe_set(span, f"{Attr.COMPLETION}.0.finish_reason", finish_reason)
            if stop_reason:
                _safe_set(span, Attr.RESP_STOP, stop_reason)
            # If we accumulated text, set as assistant completion
            if buffer_text:
                try:
                    text = "".join(buffer_text)
                    red = maybe_redact(text)
                    if red:
                        _safe_set(span, f"{Attr.COMPLETION}.0.content", red)
                        _safe_set(span, f"{Attr.COMPLETION}.0.role", "assistant")
                except Exception:
                    pass
            try:
                record_streaming_metrics(ttfb=first_token_time, generate_time=(max(total_time - first_token_time, 0.0) if first_token_time is not None else None), chunks=chunk_count)
            except Exception:
                pass
            span.set_status(Status(StatusCode.OK))
            span.end()

    # Return a generator that keeps the span open until exhaustion
    return generator()


def _wrap_async_stream_result(span, async_stream_obj: AsyncIterable[Any], parser: Optional[Callable[[Any], Dict[str, Any]]] = None):
    start_time = time.time()
    first_token_time: Optional[float] = None
    chunk_count = 0
    finish_reason: Optional[str] = None
    stop_reason: Optional[str] = None
    buffer_text: list[str] = []
    run_status: Optional[str] = None

    async def agen():
        nonlocal first_token_time, chunk_count, finish_reason, stop_reason, buffer_text, run_status
        try:
            async for chunk in async_stream_obj:
                chunk_count += 1
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                    _safe_set(span, Attr.STREAM_TTFB, first_token_time)
                if parser is not None:
                    try:
                        info = parser(chunk) or {}
                        if info.get("id") is not None:
                            _safe_set(span, Attr.RESP_ID, info.get("id"))
                        if info.get("model") is not None:
                            _safe_set(span, Attr.RESP_MODEL, info.get("model"))
                        if info.get("system_fingerprint") is not None:
                            _safe_set(span, Attr.RESP_FINGERPRINT, info.get("system_fingerprint"))
                        if info.get("finish_reason"):
                            finish_reason = info.get("finish_reason")
                        if info.get("stop_reason"):
                            stop_reason = info.get("stop_reason")
                        if isinstance(info.get("delta_text"), str) and info.get("delta_text"):
                            buffer_text.append(info.get("delta_text"))
                        if info.get("run_id"):
                            _safe_set(span, "gen_ai.run.id", info.get("run_id"))
                        if info.get("status"):
                            run_status = info.get("status")
                            _safe_set(span, "gen_ai.run.status", run_status)
                        if info.get("thread_id"):
                            _safe_set(span, "gen_ai.thread.id", info.get("thread_id"))
                        if info.get("assistant_id"):
                            _safe_set(span, "gen_ai.assistant.id", info.get("assistant_id"))
                    except Exception:
                        pass
                yield chunk
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            if first_token_time is not None:
                total_time = time.time() - start_time
                _safe_set(span, Attr.STREAM_TTG, max(total_time - first_token_time, 0.0))
            _safe_set(span, Attr.STREAM_CHUNKS, chunk_count)
            if finish_reason:
                _safe_set(span, Attr.RESP_FINISH, finish_reason)
                _safe_set(span, f"{Attr.COMPLETION}.0.finish_reason", finish_reason)
            if stop_reason:
                _safe_set(span, Attr.RESP_STOP, stop_reason)
            if buffer_text:
                try:
                    text = "".join(buffer_text)
                    red = maybe_redact(text)
                    if red:
                        _safe_set(span, f"{Attr.COMPLETION}.0.content", red)
                        _safe_set(span, f"{Attr.COMPLETION}.0.role", "assistant")
                except Exception:
                    pass
            try:
                record_streaming_metrics(ttfb=first_token_time, generate_time=(max(total_time - first_token_time, 0.0) if first_token_time is not None else None), chunks=chunk_count)
            except Exception:
                pass
            span.set_status(Status(StatusCode.OK))
            span.end()

    return agen()


# Attribute handlers (minimal, since we already set most in wrapper)
def _handler_noop(args: Optional[Tuple], kwargs: Optional[Dict], return_value: Any, span) -> None:
    # Extract response basics and usage
    _handle_openai_return(span, return_value)


def _handle_assistant_create(args: Optional[Tuple], kwargs: Optional[Dict], return_value: Any, span) -> None:
    try:
        rd = _model_as_dict(return_value)
        if not isinstance(rd, dict):
            return
        if rd.get("id") is not None:
            _safe_set(span, "gen_ai.assistant.id", rd.get("id"))
        if rd.get("model") is not None:
            _safe_set(span, Attr.RESP_MODEL, rd.get("model"))
        # Include request-provided non-sensitive metadata
        if isinstance(kwargs, dict):
            if kwargs.get("name") is not None:
                _safe_set(span, "gen_ai.assistant.name", kwargs.get("name"))
            if kwargs.get("description") is not None:
                _safe_set(span, "gen_ai.assistant.description", kwargs.get("description"))
    except Exception:
        pass


def _handle_run_create(args: Optional[Tuple], kwargs: Optional[Dict], return_value: Any, span) -> None:
    try:
        rd = _model_as_dict(return_value)
        if not isinstance(rd, dict):
            return
        for key_src, key_dst in (
            ("id", "gen_ai.run.id"),
            ("status", "gen_ai.run.status"),
            ("thread_id", "gen_ai.thread.id"),
            ("assistant_id", "gen_ai.assistant.id"),
        ):
            if rd.get(key_src) is not None:
                _safe_set(span, key_dst, rd.get(key_src))
        if rd.get("model") is not None:
            _safe_set(span, Attr.RESP_MODEL, rd.get("model"))
        # Usage (if present on run objects)
        usage = _extract_usage_dict(rd.get("usage"))
        _set_usage(span, usage)
    except Exception:
        pass


def _handle_run_retrieve(args: Optional[Tuple], kwargs: Optional[Dict], return_value: Any, span) -> None:
    # Same mapping as create
    _handle_run_create(args, kwargs, return_value, span)


def _handle_messages_list(args: Optional[Tuple], kwargs: Optional[Dict], return_value: Any, span) -> None:
    try:
        # Thread id from args/kwargs
        thread_id = None
        if args and len(args) > 0:
            thread_id = args[0]
        elif isinstance(kwargs, dict):
            thread_id = kwargs.get("thread_id")
        if thread_id is not None:
            _safe_set(span, "gen_ai.thread.id", thread_id)

        rd = _model_as_dict(return_value)
        if not isinstance(rd, dict):
            return
        data = rd.get("data")
        if isinstance(data, list):
            _safe_set(span, "gen_ai.messages.count", len(data))
            # Enrich first few messages with lightweight fields
            try:
                limit = 5
                for i, msg in enumerate(data[:limit]):
                    d = msg if isinstance(msg, dict) else _model_as_dict(msg)
                    prefix = f"gen_ai.messages.{i}"
                    if not isinstance(d, dict):
                        continue
                    if d.get("id") is not None:
                        _safe_set(span, f"{prefix}.id", d.get("id"))
                    if d.get("role") is not None:
                        _safe_set(span, f"{prefix}.role", d.get("role"))
                    if d.get("created_at") is not None:
                        _safe_set(span, f"{prefix}.created_at", d.get("created_at"))
                    # Content can be a list of blocks; include short text values
                    content = d.get("content")
                    if isinstance(content, list):
                        idx = 0
                        for c in content:
                            cd = c if isinstance(c, dict) else _model_as_dict(c)
                            if isinstance(cd, dict) and cd.get("type") == "text":
                                text_obj = cd.get("text")
                                if isinstance(text_obj, dict) and isinstance(text_obj.get("value"), str):
                                    _safe_set(span, f"{prefix}.content.{idx}", text_obj.get("value"))
                                    idx += 1
            except Exception:
                pass
    except Exception:
        pass


def _parse_chat_stream_chunk(chunk: Any) -> Dict[str, Any]:
    """Best-effort parsing of ChatCompletion streaming chunks.

    Returns a dict with optional keys: id, model, system_fingerprint, finish_reason.
    """
    out: Dict[str, Any] = {}
    try:
        d = _model_as_dict(chunk)
        if not isinstance(d, dict):
            return out
        # top-level identifiers
        for k in ("id", "model", "system_fingerprint"):
            if d.get(k) is not None:
                out[k] = d.get(k)
        # choices[].finish_reason and delta content
        choices = d.get("choices")
        if isinstance(choices, list):
            for ch in choices:
                if isinstance(ch, dict) and ch.get("finish_reason"):
                    out["finish_reason"] = ch.get("finish_reason")
                # delta can be dict with content or text
                if isinstance(ch, dict) and isinstance(ch.get("delta"), dict):
                    delta = ch.get("delta")
                    if isinstance(delta.get("content"), str):
                        out["delta_text"] = (out.get("delta_text", "") + delta.get("content")) if out.get("delta_text") else delta.get("content")
                    elif isinstance(delta.get("content"), list):
                        # concatenate any text entries in list
                        try:
                            parts = []
                            for item in delta.get("content"):
                                idict = item if isinstance(item, dict) else _model_as_dict(item)
                                if isinstance(idict.get("text"), str):
                                    parts.append(idict.get("text"))
                            if parts:
                                text = "".join(parts)
                                out["delta_text"] = (out.get("delta_text", "") + text) if out.get("delta_text") else text
                        except Exception:
                            pass
    except Exception:
        pass
    return out


def _parse_responses_stream_event(event: Any) -> Dict[str, Any]:
    """Best-effort parsing of Responses API stream events.

    Handles diverse shapes by searching top-level and common nested keys.
    """
    out: Dict[str, Any] = {}
    try:
        d = _model_as_dict(event)
        if not isinstance(d, dict):
            return out
        # direct fields
        for k in ("id", "model", "system_fingerprint", "stop_reason", "finish_reason"):
            if d.get(k) is not None:
                out[k] = d.get(k)
        # textual deltas (common shapes): 'delta', 'text', or nested under 'item'/'output'
        try:
            if isinstance(d.get("delta"), str):
                out["delta_text"] = d.get("delta")
            elif isinstance(d.get("text"), str) and isinstance(d.get("type"), str) and d.get("type").endswith(".delta"):
                out["delta_text"] = d.get("text")
            # Some SDKs wrap into data or response structures
            for key in ("data", "response", "message"):
                sub = d.get(key)
                sub = sub if isinstance(sub, dict) else _model_as_dict(sub)
                if isinstance(sub, dict):
                    if isinstance(sub.get("delta"), str):
                        out["delta_text"] = sub.get("delta")
                    if isinstance(sub.get("text"), str) and isinstance(sub.get("type"), str) and sub.get("type").endswith(".delta"):
                        out["delta_text"] = sub.get("text")
        except Exception:
            pass
        # nested under 'response' or 'message'
        for key in ("response", "message"):
            sub = d.get(key)
            if isinstance(sub, dict):
                for k in ("id", "model", "system_fingerprint", "stop_reason", "finish_reason"):
                    if sub.get(k) is not None and k not in out:
                        out[k] = sub.get(k)
    except Exception:
        pass
    return out


def _parse_run_stream_event(event: Any) -> Dict[str, Any]:
    """Best-effort parsing of Assistants Run create_and_stream events.

    Returns keys: run_id, status, thread_id, assistant_id when discoverable.
    """
    out: Dict[str, Any] = {}
    try:
        d = _model_as_dict(event)
        if not isinstance(d, dict):
            return out
        # Direct fields
        if d.get("id") and isinstance(d.get("id"), str):
            out["run_id"] = d.get("id")
        for k in ("status", "thread_id", "assistant_id"):
            if d.get(k) is not None:
                out[k] = d.get(k)
        # Nested under 'run' or 'data'
        for key in ("run", "data"):
            sub = d.get(key)
            sub = sub if isinstance(sub, dict) else _model_as_dict(sub)
            if isinstance(sub, dict):
                if sub.get("id") and "run_id" not in out:
                    out["run_id"] = sub.get("id")
                for k in ("status", "thread_id", "assistant_id"):
                    if sub.get(k) is not None and k not in out:
                        out[k] = sub.get(k)
    except Exception:
        pass
    return out


def instrument_openai() -> None:
    """Instrument OpenAI SDK for passive LLM tracking (v1 preferred, v0 fallback)."""
    # Try v1 surfaces first
    try:
        # Chat Completions (sync/async)
        wrap_function_wrapper(
            "openai.resources.chat.completions",
            "Completions.create",
            _wrap_method(
                "openai.chat.completions",
                _handler_noop,
                stream_parser=_parse_chat_stream_chunk,
            ),
        )
        wrap_function_wrapper(
            "openai.resources.chat.completions",
            "AsyncCompletions.create",
            _wrap_method(
                "openai.chat.completions",
                _handler_noop,
                is_async=True,
                stream_parser=_parse_chat_stream_chunk,
            ),
        )
        # Beta chat parse (some SDKs)
        try:
            wrap_function_wrapper(
                "openai.resources.beta.chat.completions",
                "Completions.parse",
                _wrap_method(
                    "openai.beta.chat.parse",
                    _handler_noop,
                    stream_parser=_parse_chat_stream_chunk,
                ),
            )
            wrap_function_wrapper(
                "openai.resources.beta.chat.completions",
                "AsyncCompletions.parse",
            _wrap_method(
                "openai.beta.chat.parse",
                _handler_noop,
                is_async=True,
                stream_parser=_parse_chat_stream_chunk,
            ),
        )
        except Exception:
            pass

        # Responses API (sync/async)
        wrap_function_wrapper(
            "openai.resources.responses",
            "Responses.create",
            _wrap_method(
                "openai.responses",
                _handle_responses_return,
                stream_parser=_parse_responses_stream_event,
            ),
        )
        wrap_function_wrapper(
            "openai.resources.responses",
            "AsyncResponses.create",
            _wrap_method(
                "openai.responses",
                _handle_responses_return,
                is_async=True,
                stream_parser=_parse_responses_stream_event,
            ),
        )

        # Legacy Completions (sync/async)
        wrap_function_wrapper(
            "openai.resources.completions", "Completions.create", _wrap_method("openai.completions", _handler_noop)
        )
        wrap_function_wrapper(
            "openai.resources.completions",
            "AsyncCompletions.create",
            _wrap_method("openai.completions", _handler_noop, is_async=True),
        )

        # Embeddings (sync/async)
        wrap_function_wrapper(
            "openai.resources.embeddings", "Embeddings.create", _wrap_method("openai.embeddings", _handler_noop)
        )
        wrap_function_wrapper(
            "openai.resources.embeddings",
            "AsyncEmbeddings.create",
            _wrap_method("openai.embeddings", _handler_noop, is_async=True),
        )

        # Images (generate)
        wrap_function_wrapper(
            "openai.resources.images", "Images.generate", _wrap_method("openai.images.generate", _handler_noop)
        )

        # Assistants + Runs + Messages (beta)
        wrap_function_wrapper(
            "openai.resources.beta.assistants",
            "Assistants.create",
            _wrap_method("openai.assistants.create", _handle_assistant_create),
        )
        wrap_function_wrapper(
            "openai.resources.beta.threads.runs",
            "Runs.create",
            _wrap_method("openai.runs.create", _handle_run_create),
        )
        wrap_function_wrapper(
            "openai.resources.beta.threads.runs",
            "Runs.retrieve",
            _wrap_method("openai.runs.retrieve", _handle_run_retrieve),
        )
        wrap_function_wrapper(
            "openai.resources.beta.threads.runs",
            "Runs.create_and_stream",
            _wrap_method(
                "openai.runs.create_and_stream",
                _handle_run_create,  # initial return (often a stream handle)
                stream_parser=_parse_run_stream_event,
            ),
        )
        wrap_function_wrapper(
            "openai.resources.beta.threads.messages",
            "Messages.list",
            _wrap_method("openai.messages.list", _handle_messages_list),
        )
        return
    except Exception:
        # Fallback to v0 instrumentor surface
        pass

    # v0 legacy client
    try:
        wrap_function_wrapper("openai", "ChatCompletion.create", _wrap_method("openai.chat.completions", _handler_noop))
        wrap_function_wrapper("openai", "Completion.create", _wrap_method("openai.completions", _handler_noop))
        wrap_function_wrapper("openai", "Embedding.create", _wrap_method("openai.embeddings", _handler_noop))
        wrap_function_wrapper("openai", "Image.create", _wrap_method("openai.images.generate", _handler_noop))
    except Exception:
        # As a last resort, do nothing
        return


def uninstrument_openai() -> None:
    """Cleanly unwrap OpenAI methods (v1/v0)."""
    targets = [
        ("openai.resources.chat.completions", "Completions.create"),
        ("openai.resources.chat.completions", "AsyncCompletions.create"),
        ("openai.resources.beta.chat.completions", "Completions.parse"),
        ("openai.resources.beta.chat.completions", "AsyncCompletions.parse"),
        ("openai.resources.responses", "Responses.create"),
        ("openai.resources.responses", "AsyncResponses.create"),
        ("openai.resources.completions", "Completions.create"),
        ("openai.resources.completions", "AsyncCompletions.create"),
        ("openai.resources.embeddings", "Embeddings.create"),
        ("openai.resources.embeddings", "AsyncEmbeddings.create"),
        ("openai.resources.images", "Images.generate"),
        ("openai.resources.beta.assistants", "Assistants.create"),
        ("openai.resources.beta.threads.runs", "Runs.create"),
        ("openai.resources.beta.threads.runs", "Runs.retrieve"),
        ("openai.resources.beta.threads.runs", "Runs.create_and_stream"),
        ("openai.resources.beta.threads.messages", "Messages.list"),
        # v0 fallbacks
        ("openai", "ChatCompletion.create"),
        ("openai", "Completion.create"),
        ("openai", "Embedding.create"),
        ("openai", "Image.create"),
    ]
    for mod, meth in targets:
        try:
            otel_unwrap(mod, meth)
        except Exception:
            pass
