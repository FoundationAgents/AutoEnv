from typing import Any, Dict, List, Optional, Tuple

from base.agent.base_solver import _extract_agent_observation
from base.env.base_env import SkinEnv
from base.env.base_observation import ObservationPolicy
from base.env.skin_output import SkinRenderOutput


class DummyObservationPolicy(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int):
        return {"value": env_state["value"], "t": t}


class LegacyRenderEnv(SkinEnv):
    def __init__(self):
        super().__init__(env_id=1, obs_policy=DummyObservationPolicy())

    def _dsl_config(self):
        self.configs = {"termination": {"max_steps": 5}}

    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        self._state = {"value": 0}
        self._history = []
        self._t = 0
        return self._state

    def _load_world(self, world_id: str) -> Dict[str, Any]:
        return {"value": 0}

    def _generate_world(self, seed: Optional[int] = None) -> str:
        return "dummy"

    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._state = {"value": self._state["value"] + 1}
        return self._state

    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        return 0.0, [], {}

    def observe_semantic(self):
        return self.obs_policy(self._state, self._t)

    def render_skin(self, omega) -> Any:
        return f"value={omega['value']}"


class StructuredRenderEnv(LegacyRenderEnv):
    def render_skin(self, omega) -> Any:
        return {
            "agent_view": {"compact": omega["value"]},
            "human_view": f"Value is {omega['value']}",
            "modality": "3d",
            "artifacts": {"entrypoint": "game/index.html"},
            "metadata": {"camera": "orbit"},
        }


class DoneNoArgEnv(LegacyRenderEnv):
    # Validate compatibility path where done() takes no state argument.
    def done(self) -> bool:
        return self._t >= 1


def test_legacy_render_skin_is_normalized():
    env = LegacyRenderEnv()
    env.reset()
    _, _, _, info = env.step({"action": "noop", "params": {}})

    assert info["skinned"] == "value=1"
    assert info["agent_obs"] == "value=1"
    assert info["human_view"] == "value=1"
    assert info["render_output"]["modality"] == "text"


def test_structured_render_skin_contract_is_preserved():
    env = StructuredRenderEnv()
    env.reset()
    _, _, _, info = env.step({"action": "noop", "params": {}})

    assert info["agent_obs"] == {"compact": 1}
    assert info["skinned"] == {"compact": 1}
    assert info["human_view"] == "Value is 1"
    assert info["render_output"]["modality"] == "3d"
    assert info["render_output"]["artifacts"]["entrypoint"] == "game/index.html"


def test_step_done_signature_compatibility():
    env = DoneNoArgEnv()
    env.reset()
    _, _, done, _ = env.step({"action": "noop", "params": {}})
    assert done is True


def test_solver_extracts_agent_view_from_contract_object():
    rendered = SkinRenderOutput(agent_view={"v": 1}, modality="text")
    assert _extract_agent_observation(rendered) == {"v": 1}
