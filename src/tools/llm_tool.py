"""LLM API call wrapper. Supports DeepSeek and Zhipu via OpenAI-compatible SDK."""

import base64
import time
from pathlib import Path
from openai import OpenAI

from src.config import get_llm_config

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        cfg = get_llm_config()
        _client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    return _client


def chat(system_prompt: str, user_message: str, temperature: float = 0.3) -> str:
    """Send a chat completion request and return the response text.
    Auto-retries once on transient failures (timeout, rate limit, server error)."""
    cfg = get_llm_config()
    client = _get_client()
    last_error = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                timeout=90,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            last_error = e
            if attempt == 0:
                time.sleep(2)
    return "服务繁忙，请稍后再试。"


def chat_stream(system_prompt: str, user_message: str, temperature: float = 0.3):
    """Stream chat completion response. Falls back to non-streaming on failure."""
    cfg = get_llm_config()
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            timeout=90,
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception:
        # Fall back to non-streaming
        try:
            result = chat(system_prompt, user_message, temperature)
            if result:
                yield "\n\n" + result
        except Exception:
            yield "\n\n*抱歉，生成回复时遇到问题，请稍后重试。*"


def ocr_image(image_path: str) -> str:
    """Extract text from an image using Zhipu GLM-4V vision model.
    Returns extracted text on success. Raises ValueError if not configured or on failure."""
    from src.config import ZHIPU_API_KEY, ZHIPU_BASE_URL

    if not ZHIPU_API_KEY:
        raise ValueError("Cloud OCR requires Zhipu API key configured in .env")

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = path.suffix.lower()
    mime_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    mime_type = mime_types.get(suffix, "image/jpeg")

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    client = OpenAI(api_key=ZHIPU_API_KEY, base_url=ZHIPU_BASE_URL)
    response = client.chat.completions.create(
        model="glm-4v-flash",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "请仔细识别并输出这张图片中的所有文字内容，保持原有格式和排版。"
                        "对于药品说明书、体检报告等文档，请特别仔细识别数字和单位，确保数值准确。"
                        "只输出提取的文字，不要添加任何解释或补充说明。"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                },
            ],
        }],
        temperature=0.0,
        max_tokens=4096,
        timeout=60,
    )
    text = response.choices[0].message.content or ""
    if not text.strip():
        raise ValueError("Cloud OCR returned empty result")
    return text.strip()
