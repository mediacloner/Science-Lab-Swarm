"""TabbyAPI HTTP client for model management and inference."""

import time
import logging
import requests

logger = logging.getLogger(__name__)


class TabbyClient:
    """Client for TabbyAPI model loading, unloading, and chat completion."""

    def __init__(self, base_url: str = "http://localhost:5000", api_key: str = "", timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self._current_model = None

    @property
    def current_model(self) -> str | None:
        return self._current_model

    def health_check(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", headers=self.headers, timeout=10)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def load_model(self, name: str, path: str, max_seq_len: int = 8192) -> float:
        """Load model into VRAM. Returns load time in seconds."""
        if self._current_model == name:
            logger.info(f"Model {name} already loaded, skipping")
            return 0.0

        t0 = time.time()
        self.unload_model()

        payload = {
            "model_name": name,
            "max_seq_len": max_seq_len,
        }
        r = requests.post(
            f"{self.base_url}/v1/model/load",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
            stream=True,
        )
        r.raise_for_status()

        # Stream until model is ready
        for line in r.iter_lines():
            if line:
                logger.debug(f"Load progress: {line.decode()}")

        self._current_model = name
        elapsed = time.time() - t0
        logger.info(f"Loaded {name} in {elapsed:.1f}s")
        return elapsed

    def unload_model(self):
        """Evict current model from VRAM."""
        if self._current_model is None:
            return
        try:
            r = requests.post(
                f"{self.base_url}/v1/model/unload",
                headers=self.headers,
                timeout=30,
            )
            r.raise_for_status()
            logger.info(f"Unloaded {self._current_model}")
        except requests.RequestException as e:
            logger.warning(f"Unload failed: {e}")
        self._current_model = None

    def swap_model(self, name: str, path: str, max_seq_len: int = 8192) -> float:
        """Unload current model and load a new one. Returns swap time."""
        return self.load_model(name, path, max_seq_len)

    def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> str:
        """Send chat completion request, return assistant message content."""
        payload = {
            "model": self._current_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        r = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
