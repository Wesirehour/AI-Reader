from typing import Optional

from fastapi import HTTPException
from openai import OpenAI

from app.core.config import settings


def _normalize_base_url(url):
    base = (url or "").strip().rstrip("/")
    # Accept various user inputs and canonicalize to .../v1
    # e.g.
    # - https://api.scnet.cn/api/llm
    # - https://api.scnet.cn/api/llm/v1
    # - https://api.scnet.cn/api/llm/v1/chat/completions
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base


class ModelConfigCenter:
    def __init__(self):
        models = settings.llm_model_list or []
        if settings.llm_default_model not in models:
            models = [settings.llm_default_model] + [m for m in models if m != settings.llm_default_model]
        self.models = models
        self.default_model = settings.llm_default_model
        self.provider = settings.llm_provider
        self.base_url = _normalize_base_url(settings.llm_base_url)

    def list_models(self):
        return self.models

    def resolve_model(self, requested: Optional[str]):
        model = (requested or self.default_model or "").strip()
        if not model:
            raise HTTPException(status_code=500, detail="No model configured")
        if model not in self.models:
            raise HTTPException(status_code=400, detail="Unsupported model: %s" % model)
        return model


class OpenAIModelRouter:
    def __init__(self, config_center: ModelConfigCenter):
        self.config_center = config_center

    def _client(self):
        if not settings.llm_api_key:
            raise HTTPException(status_code=500, detail="LLM_API_KEY is not configured")
        return OpenAI(
            api_key=settings.llm_api_key,
            base_url=self.config_center.base_url,
            timeout=settings.llm_timeout_sec,
        )

    def chat(self, messages: list, model: Optional[str] = None, provider: Optional[str] = None, temperature: float = 0.2):
        _ = provider  # provider kept for compatibility in API payload
        used_model = self.config_center.resolve_model(model)
        try:
            client = self._client()
            resp = client.chat.completions.create(
                model=used_model,
                messages=messages,
                temperature=temperature,
            )
            if not resp.choices or not resp.choices[0].message:
                raise HTTPException(status_code=502, detail="LLM provider returned empty response")
            content = resp.choices[0].message.content or ""
            return content, self.config_center.provider, used_model
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail="LLM provider request failed: %s" % str(exc)) from exc


model_config_center = ModelConfigCenter()
model_router = OpenAIModelRouter(model_config_center)


def stream_chat(messages, model=None, provider=None, temperature=0.2):
    """Yield (delta_text, provider, model) tuples. Used by the SSE endpoint."""
    _ = provider
    used_model = model_config_center.resolve_model(model)
    if not settings.llm_api_key:
        raise HTTPException(status_code=500, detail="LLM_API_KEY is not configured")
    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=model_config_center.base_url,
        timeout=settings.llm_timeout_sec,
    )
    try:
        stream = client.chat.completions.create(
            model=used_model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                delta = ""
            if delta:
                yield delta, model_config_center.provider, used_model
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="LLM provider stream failed: %s" % str(exc)) from exc
