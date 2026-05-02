#!/usr/bin/env python3
"""Thin client for an OpenAI-compatible Vision-Language Model server.

Designed for vLLM (`python -m vllm.entrypoints.openai.api_server`)
serving a multimodal model (Qwen2-VL, InternVL, Llama-3.2-Vision,
etc.). Falls back gracefully when the server returns text without
the expected JSON wrapping.

This module is shared by phantom_recognizer_node and
instruction_planner_node; it deliberately has no ROS imports so it
can be unit-tested standalone.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "us3d_vlm requires 'requests' (pip install requests)") from exc


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def encode_image_jpeg_b64(image_bgr: np.ndarray,
                          jpeg_quality: int = 80,
                          max_side_px: int = 768) -> str:
    """Compress a BGR image into a base64 JPEG data URL string.

    Returns just the base64 payload (no `data:` prefix). Callers
    typically wrap it as `data:image/jpeg;base64,<payload>`.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required to encode images")

    img = image_bgr
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side > max_side_px:
        scale = max_side_px / float(long_side)
        new_size = (int(round(w * scale)), int(round(h * scale)))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(
        '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("cv2.imencode JPEG failed")
    return base64.b64encode(buf.tobytes()).decode('ascii')


def _scan_json_candidates(text: str):
    """Yield every well-balanced JSON-object substring of `text`.

    Robust to thinking-model output where the actual answer is
    typically the LAST top-level {...} after a wall of reasoning.
    String literals are tracked so brace counts inside quotes
    don't break matching.
    """
    n = len(text)
    i = 0
    while i < n:
        if text[i] != '{':
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        closed = False
        for j in range(i, n):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == '\\':
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    yield text[i:j + 1]
                    i = j + 1
                    closed = True
                    break
        if not closed:
            return


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from a free-form LLM response.

    Tries, in order:
      1. Plain `json.loads(text)`.
      2. Fenced code block ```json ... ```.
      3. Every balanced `{...}` substring; among those that parse,
         pick the LAST one whose keys overlap our schema (e.g.
         "phantom_type", "scan_axis", "bbox", …). Thinking models
         habitually drop the answer at the very end of the
         reasoning stream.

    Returns None if nothing parses.
    """
    if not text:
        return None
    text = text.strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    fence_matches = re.findall(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    for cand in reversed(fence_matches):
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue

    schema_keys = {
        'phantom_type', 'confidence', 'description', 'scan_axis_hint',
        'scan_length_hint_mm', 'bbox', 'bbox_norm', 'bbox_image_xyxy',
        'scan_axis', 'scan_length_mm', 'scan_speed_mms',
        'use_marker_pair', 'reverse_direction', 'notes',
    }
    parsed_objs = []
    for cand in _scan_json_candidates(text):
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            parsed_objs.append(obj)

    if not parsed_objs:
        return None

    # Prefer objects whose keys overlap the schema, taking the
    # LAST such object (thinking models put the final answer last).
    schema_objs = [o for o in parsed_objs
                   if set(o.keys()) & schema_keys]
    if schema_objs:
        return schema_objs[-1]
    return parsed_objs[-1]


# ─────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────

@dataclass
class VLMConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen2-VL-7B-Instruct"
    text_model: str = ""
    timeout_s: float = 180.0
    max_retries: int = 0
    temperature: float = 0.2
    top_p: float = 0.9
    # 0 (or any non-positive value) = do NOT send the max_tokens
    # field, letting the server use the model's own ceiling. This
    # is the safest default for thinking models because reasoning
    # length is hard to predict.
    max_tokens: int = 0
    # Set true ONLY when the backend (e.g. local vLLM, OpenAI) is
    # known to honor `response_format=json_object`. Chinese cloud
    # gateways (volcengine ark, some Moonshot routes) often fail
    # silently with this flag — leave it off and rely on prompt
    # engineering + extract_json_block().
    use_response_format: bool = False
    # Use Server-Sent-Event streaming. Strongly recommended for
    # cloud "thinking" models (kimi-k2, deepseek-r1, qwen-thinking,
    # etc.): the server sends chunks of reasoning_content while it
    # is still thinking, which keeps the TCP connection alive even
    # when the underlying HTTP read timeout is much shorter than
    # the total inference time.
    stream: bool = True


@dataclass
class VLMResult:
    ok: bool
    content: str = ""             # raw model text
    parsed: Optional[Dict[str, Any]] = None
    error: str = ""
    elapsed_s: float = 0.0
    usage: Dict[str, Any] = field(default_factory=dict)
    # Diagnostic fields. Filled even on success so callers can log
    # them when the *parsed* result is empty / unexpected.
    http_status: int = 0
    raw_response: str = ""        # truncated raw HTTP body
    request_url: str = ""
    request_model: str = ""


class VLMClient:
    """OpenAI Chat Completions client (HTTP, no SDK dependency)."""

    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg
        self._session = requests.Session()
        self._progress_cb = None
        self._progress_interval_s = 5.0

    def set_progress_callback(self, cb, interval_s: float = 5.0) -> None:
        """Register a heartbeat callback for streaming requests.

        `cb(elapsed_s, n_content_chunks, n_reasoning_chunks)` is
        invoked roughly every `interval_s` seconds while a stream
        is being consumed. Use it to surface 'still alive' progress
        messages to the operator (e.g. via rospy.loginfo).
        """
        self._progress_cb = cb
        self._progress_interval_s = max(0.5, float(interval_s))

    # ---- Public API ------------------------------------------------

    def chat(self,
             messages: List[Dict[str, Any]],
             *,
             use_text_model: bool = False,
             want_json: bool = False) -> VLMResult:
        """Send a chat-completions request.

        `messages` follows OpenAI format. Each message's `content`
        can be a string or a list of content parts (for VLM
        multimodal input).
        """
        model = (self.cfg.text_model
                 if use_text_model and self.cfg.text_model
                 else self.cfg.model)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
        }
        # max_tokens <= 0 (or None) means "do not cap output" — omit
        # the field so the server falls back to its per-model max.
        # Useful for thinking models where you cannot easily predict
        # how many reasoning tokens are needed before the answer.
        if self.cfg.max_tokens and int(self.cfg.max_tokens) > 0:
            payload["max_tokens"] = int(self.cfg.max_tokens)
        # `response_format` is supported by vLLM and OpenAI but
        # rejected (or silently ignored with empty body) by several
        # Chinese cloud OpenAI-compatible gateways (volcengine ark,
        # some Moonshot / Qwen routes, baichuan, etc.). Leave it off
        # by default; turn it on with cfg.use_response_format=True
        # only when you know the backend supports it.
        if want_json and getattr(self.cfg, 'use_response_format', False):
            payload["response_format"] = {"type": "json_object"}

        use_stream = bool(getattr(self.cfg, 'stream', False))
        if use_stream:
            payload["stream"] = True

        url = self.cfg.base_url.rstrip('/') + '/chat/completions'
        headers = {
            "Authorization": "Bearer %s" % (self.cfg.api_key or "EMPTY"),
            "Content-Type": "application/json",
        }
        if use_stream:
            headers["Accept"] = "text/event-stream"

        last_err = ""
        last_status = 0
        last_body = ""
        t0 = time.time()
        for attempt in range(self.cfg.max_retries + 1):
            try:
                if use_stream:
                    res = self._do_stream(
                        url, payload, headers, model, t0, want_json)
                else:
                    res = self._do_unary(
                        url, payload, headers, model, t0, want_json)

                # Retry only on explicit transient failures.
                if (not res.ok) and res.http_status >= 500 \
                        and attempt < self.cfg.max_retries:
                    last_err = res.error
                    last_status = res.http_status
                    last_body = res.raw_response
                    time.sleep(min(1.0 * (attempt + 1), 3.0))
                    continue
                return res
            except requests.RequestException as e:
                last_err = "%s: %s" % (type(e).__name__, e)
                if attempt >= self.cfg.max_retries:
                    break
                time.sleep(min(1.0 * (attempt + 1), 3.0))

        return VLMResult(
            ok=False,
            error=last_err or "unknown error",
            elapsed_s=time.time() - t0,
            http_status=last_status,
            raw_response=last_body,
            request_url=url,
            request_model=model,
        )

    # ---- Internal request helpers ----------------------------------

    def _do_unary(self,
                  url: str,
                  payload: Dict[str, Any],
                  headers: Dict[str, str],
                  model: str,
                  t0: float,
                  want_json: bool) -> VLMResult:
        r = self._session.post(
            url, json=payload, headers=headers,
            timeout=self.cfg.timeout_s,
        )
        last_body = r.text[:1024] if r.text else ""

        if r.status_code != 200:
            return VLMResult(
                ok=False,
                error="HTTP %d: %s" % (r.status_code, last_body),
                elapsed_s=time.time() - t0,
                http_status=r.status_code,
                raw_response=last_body,
                request_url=url, request_model=model,
            )

        try:
            data = r.json()
        except ValueError:
            return VLMResult(
                ok=False,
                error="200 OK but body is not JSON",
                elapsed_s=time.time() - t0,
                http_status=r.status_code,
                raw_response=last_body,
                request_url=url, request_model=model,
            )

        if (isinstance(data, dict) and 'error' in data
                and 'choices' not in data):
            return VLMResult(
                ok=False,
                error="API error envelope: %s"
                      % json.dumps(data.get('error'))[:300],
                elapsed_s=time.time() - t0,
                http_status=r.status_code,
                raw_response=last_body,
                request_url=url, request_model=model,
            )

        choices = (data.get("choices") or [{}])
        msg = (choices[0] or {}).get("message", {}) or {}
        content = msg.get("content", "") or ""
        if not content and isinstance(msg.get("content"), list):
            parts = []
            for part in msg["content"]:
                if isinstance(part, dict) and part.get("text"):
                    parts.append(part["text"])
            content = "".join(parts)
        if not content and isinstance(
                msg.get("reasoning_content"), str):
            content = msg["reasoning_content"]

        parsed = extract_json_block(content) if want_json else None
        return VLMResult(
            ok=True,
            content=content or "",
            parsed=parsed,
            elapsed_s=time.time() - t0,
            usage=data.get("usage", {}) or {},
            http_status=r.status_code,
            raw_response=last_body,
            request_url=url, request_model=model,
        )

    def _do_stream(self,
                   url: str,
                   payload: Dict[str, Any],
                   headers: Dict[str, str],
                   model: str,
                   t0: float,
                   want_json: bool) -> VLMResult:
        """SSE streaming: aggregate `delta.content` (and as a
        fallback `delta.reasoning_content`) from each chunk.
        timeout_s is treated as the *idle* read timeout (max gap
        between two chunks), not the total request budget.
        """
        r = self._session.post(
            url, json=payload, headers=headers,
            timeout=self.cfg.timeout_s, stream=True,
        )
        last_status = r.status_code

        if r.status_code != 200:
            body = ""
            try:
                body = r.text[:1024]
            except Exception:
                body = ""
            return VLMResult(
                ok=False,
                error="HTTP %d (stream): %s" % (r.status_code, body),
                elapsed_s=time.time() - t0,
                http_status=last_status,
                raw_response=body,
                request_url=url, request_model=model,
            )

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        usage: Dict[str, Any] = {}
        finish_reason: Optional[str] = None
        api_error: Optional[Dict[str, Any]] = None
        last_event = ""
        # Optional progress callback: invoked roughly every
        # `progress_interval_s` seconds with (elapsed_s,
        # n_content_chunks, n_reasoning_chunks). Useful so the
        # caller can log a heartbeat while a thinking model is
        # streaming reasoning tokens for a long time.
        progress_cb = getattr(self, '_progress_cb', None)
        progress_interval_s = getattr(self, '_progress_interval_s', 5.0)
        last_progress_t = t0
        n_content_chunks = 0
        n_reasoning_chunks = 0

        try:
            for raw_line in r.iter_lines(decode_unicode=True):
                # Heartbeat (even for empty keepalive lines).
                if progress_cb is not None:
                    now_t = time.time()
                    if now_t - last_progress_t >= progress_interval_s:
                        try:
                            progress_cb(now_t - t0, n_content_chunks,
                                        n_reasoning_chunks)
                        except Exception:
                            pass
                        last_progress_t = now_t
                if raw_line is None:
                    continue
                if not raw_line:
                    continue  # SSE keep-alive / event boundary
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                last_event = data_str[:512]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except (json.JSONDecodeError, ValueError):
                    continue

                if isinstance(chunk, dict) and chunk.get("error") \
                        and not chunk.get("choices"):
                    api_error = chunk["error"]
                    break

                choices = chunk.get("choices") or []
                if not choices:
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    continue

                ch0 = choices[0] or {}
                delta = ch0.get("delta") or {}
                # OpenAI-style delta.content
                c = delta.get("content")
                if isinstance(c, str) and c:
                    content_parts.append(c)
                    n_content_chunks += 1
                elif isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and part.get("text"):
                            content_parts.append(part["text"])
                            n_content_chunks += 1
                # Thinking-model delta.reasoning_content
                rc = delta.get("reasoning_content")
                if isinstance(rc, str) and rc:
                    reasoning_parts.append(rc)
                    n_reasoning_chunks += 1

                if ch0.get("finish_reason"):
                    finish_reason = ch0["finish_reason"]
                if chunk.get("usage"):
                    usage = chunk["usage"]
        finally:
            try:
                r.close()
            except Exception:
                pass

        if api_error is not None:
            return VLMResult(
                ok=False,
                error="API error envelope: %s"
                      % json.dumps(api_error)[:300],
                elapsed_s=time.time() - t0,
                http_status=last_status,
                raw_response=last_event,
                request_url=url, request_model=model,
            )

        content = "".join(content_parts)
        if not content and reasoning_parts:
            # Thinking model gave us reasoning but no final content
            # (often `finish_reason=length`). Surface reasoning so
            # the operator can see what happened.
            content = "".join(reasoning_parts)

        parsed = extract_json_block(content) if want_json else None
        return VLMResult(
            ok=True,
            content=content,
            parsed=parsed,
            elapsed_s=time.time() - t0,
            usage=usage or {},
            http_status=last_status,
            raw_response=("finish=%s last_event=%s"
                          % (finish_reason, last_event))[:1024],
            request_url=url, request_model=model,
        )

    # ---- Convenience wrappers --------------------------------------

    def chat_with_image(self,
                        system_prompt: str,
                        user_prompt: str,
                        image_bgr: np.ndarray,
                        *,
                        jpeg_quality: int = 80,
                        max_side_px: int = 768,
                        want_json: bool = True) -> VLMResult:
        b64 = encode_image_jpeg_b64(
            image_bgr, jpeg_quality=jpeg_quality, max_side_px=max_side_px)
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + b64,
                    },
                },
            ],
        })
        return self.chat(messages, use_text_model=False, want_json=want_json)

    def chat_text(self,
                  system_prompt: str,
                  user_prompt: str,
                  *,
                  want_json: bool = True) -> VLMResult:
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.chat(messages, use_text_model=True, want_json=want_json)

    # ---- Static factory from rospy params --------------------------

    @staticmethod
    def from_param_dict(p: Dict[str, Any]) -> "VLMClient":
        cfg = VLMConfig(
            base_url=str(p.get('base_url', VLMConfig.base_url)),
            api_key=str(p.get('api_key', VLMConfig.api_key)),
            model=str(p.get('model', VLMConfig.model)),
            text_model=str(p.get('text_model', '')),
            timeout_s=float(p.get('timeout_s', VLMConfig.timeout_s)),
            max_retries=int(p.get('max_retries', VLMConfig.max_retries)),
            temperature=float(p.get('temperature', VLMConfig.temperature)),
            top_p=float(p.get('top_p', VLMConfig.top_p)),
            max_tokens=int(p.get('max_tokens', VLMConfig.max_tokens)),
            use_response_format=bool(
                p.get('use_response_format', False)),
            stream=bool(p.get('stream', VLMConfig.stream)),
        )
        return VLMClient(cfg)
