"""Policy client adapter used by deployment entrypoints.

Customize this file when replacing the default OpenPI-compatible websocket
backend with another model server or local inference runtime.
"""

import numpy as np
from PIL import Image


class PolicyClient:
    """Thin adapter around the policy backend used by RTC Anything."""

    def __init__(
        self,
        host="localhost",
        port=8000,
        image_resize_mode="pad",
        image_size=(224, 224),
    ):
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy

        if image_resize_mode not in {"pad", "stretch"}:
            raise ValueError(
                "image_resize_mode must be 'pad' or 'stretch', "
                f"got {image_resize_mode!r}."
            )
        if len(image_size) != 2 or any(int(size) <= 0 for size in image_size):
            raise ValueError(f"image_size must contain two positive integers, got {image_size!r}.")

        self.backend = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.image_tools = image_tools
        self.image_resize_mode = image_resize_mode
        self.image_size = tuple(int(size) for size in image_size)
        self.observation = None

    def process_image(self, image):
        """Resize and convert an RGB image to the policy input layout."""
        target_height, target_width = self.image_size
        if self.image_resize_mode == "stretch":
            # Match the M2W training loader exactly: PIL resize to a square,
            # without preserving the source aspect ratio or adding padding.
            image = self.image_tools.convert_to_uint8(np.asarray(image))
            resized = np.asarray(
                Image.fromarray(image).resize((target_width, target_height))
            )
        else:
            resized = self.image_tools.resize_with_pad(
                image, target_height, target_width
            )
        return np.ascontiguousarray(
            self.image_tools.convert_to_uint8(resized.transpose(2, 0, 1))
        )

    def process_image_tree(self, value):
        """Recursively preprocess images in lists/dicts such as wrist history clips."""
        if value is None:
            return None
        if isinstance(value, dict):
            return {
                key: self.process_image_tree(item)
                for key, item in value.items()
                if item is not None
            }
        if isinstance(value, (list, tuple)):
            return [
                self.process_image_tree(item)
                for item in value
                if item is not None
            ]
        return self.process_image(value)

    def update_observation(self, obs):
        """Preprocess raw images and store the latest observation."""
        processed_obs = dict(obs)
        processed_obs["images"] = {
            name: self.process_image(image)
            for name, image in obs.get("images", {}).items()
            if image is not None
        }
        if "wrist_views" in obs:
            processed_obs["wrist_views"] = self.process_image_tree(obs["wrist_views"])
        self.observation = processed_obs

    def get_action(self):
        """Return an action chunk for the latest observation."""
        if self.observation is None:
            raise RuntimeError("Policy observation is empty. Call update_observation(obs) before get_action().")
        response = self.backend.infer(self.observation)
        if not response.get("ok", True):
            error = response.get("error", {})
            message = error.get("message", str(error))
            raise RuntimeError(f"Policy server inference failed: {message}")
        if "actions" in response:
            return self._normalize_action_chunk(response["actions"])
        data = response.get("data", {})
        if isinstance(data, dict) and "actions" in data:
            return self._normalize_action_chunk(data["actions"])
        if isinstance(data, dict) and "normalized_actions" in data:
            raise RuntimeError(
                "Policy server returned normalized_actions but no unnormalized actions. "
                "Use the M2W place-bag deploy.py adapter or add server-side unnormalization."
            )
        raise KeyError(f"Key 'actions' not found in policy response: {response.keys()}")

    @staticmethod
    def _normalize_action_chunk(actions):
        """Convert server action output to [chunk, action_dim]."""
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(
                "Policy actions must have shape [chunk, action_dim] or [1, chunk, action_dim], "
                f"got {actions.shape}."
            )
        return actions

    def get_server_metadata(self):
        """Return metadata sent by the websocket policy server, if available."""
        if hasattr(self.backend, "get_server_metadata"):
            return self.backend.get_server_metadata()
        return {}

    def reset(self):
        """Reset backend episode state if the policy requires it.

        OpenPI pi0 does not require per-episode reset, so the default adapter is
        intentionally a no-op. Override this method for stateful policy backends.
        """
        if hasattr(self.backend, "reset"):
            return self.backend.reset()
        return None
