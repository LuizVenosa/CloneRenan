import asyncio
import json
import uuid
from typing import Optional

import websockets


class VTSClient:
    """Minimal VTube Studio websocket client (default port: 8001)."""

    def __init__(self, plugin_name: str = "RenanOrchestrator", plugin_developer: str = "Codex"):
        self.plugin_name = plugin_name
        self.plugin_developer = plugin_developer
        self._ws = None
        self._auth_token: Optional[str] = None

    async def connect(self, host: str = "127.0.0.1", port: int = 8001) -> None:
        self._ws = await websockets.connect(f"ws://{host}:{port}")

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def authenticate(self, auth_token: Optional[str] = None) -> Optional[str]:
        self._auth_token = auth_token
        if self._auth_token:
            await self._request(
                "AuthenticationRequest",
                {
                    "pluginName": self.plugin_name,
                    "pluginDeveloper": self.plugin_developer,
                    "authenticationToken": self._auth_token,
                },
            )
            return self._auth_token

        response = await self._request(
            "AuthenticationTokenRequest",
            {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_developer,
                "pluginIcon": "",
            },
        )
        token = response.get("data", {}).get("authenticationToken")
        self._auth_token = token
        return token

    async def trigger_hotkey(self, hotkey_name: str, item_instance_id: Optional[str] = None) -> None:
        payload = {"hotkeyID": hotkey_name}
        if item_instance_id:
            payload["itemInstanceID"] = item_instance_id
        await self._request("HotkeyTriggerRequest", payload)

    async def set_state(self, state: str) -> None:
        mapping = {
            "thinking": "Thinking",
            "speaking": "Speaking",
            "idle": "Idle",
        }
        hotkey = mapping.get(state.lower())
        if not hotkey:
            return
        await self.trigger_hotkey(hotkey)

    async def _request(self, message_type: str, data: dict) -> dict:
        if not self._ws:
            raise RuntimeError("VTS websocket is not connected")

        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": str(uuid.uuid4()),
            "messageType": message_type,
            "data": data,
        }
        await self._ws.send(json.dumps(payload))
        raw = await self._ws.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        return json.loads(raw)
