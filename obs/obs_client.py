from typing import Optional


class OBSClient:
    """OBS websocket client wrapper (default port: 4455)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4455, password: str = ""):
        self.host = host
        self.port = port
        self.password = password
        self._client = None

    def connect(self) -> None:
        try:
            import obsws_python as obs
        except Exception as exc:
            raise RuntimeError("obsws-python is not installed. Install with: pip install obsws-python") from exc

        self._client = obs.ReqClient(host=self.host, port=self.port, password=self.password)

    def disconnect(self) -> None:
        self._client = None

    def set_text_source(self, source_name: str, text: str) -> None:
        if not self._client:
            return
        self._client.set_input_settings(source_name, {"text": text}, overlay=True)

    def switch_scene(self, scene_name: str) -> None:
        if not self._client:
            return
        self._client.set_current_program_scene(scene_name)
