"""Compatibility shim that lets us use the public PyPI versions of
robosuite / robomimic while preserving the small set of extensions that the
``s-tian/robosuite`` and ``s-tian/robomimic`` (``depth_images`` branch) forks
used to provide.

Features re-implemented here:

* ``create_env_for_data_processing`` accepting the additional kwargs used by
  vp2 (``camera_depths``, ``camera_normals``, ``camera_segmentations``,
  ``randomize_lighting``, ``randomize_color``, ``randomize_freq``,
  ``renderer``).
* ``get_object_positions`` — used by ``RobosuiteEnv.compute_score`` to score
  rollouts.  The fork shipped this as a method on the custom env class; here we
  derive object positions from the underlying ``robosuite`` model.
* Optional MuJoCo lighting / colour randomisation via robosuite's built-in
  ``LightingModder`` / ``TextureModder``.

The legacy ``renderer="igibson"`` path is not available without the iGibson
patch to robomimic; we transparently fall back to robosuite's MuJoCo renderer
and emit a warning so the user knows.
"""

from __future__ import annotations

import copy
import warnings
from typing import Iterable, Sequence

import numpy as np

import robomimic.utils.env_utils as EnvUtils


def _coerce_per_camera(value, n_cameras: int):
    """Accept either a scalar or a per-camera iterable and return a list."""
    if value is None:
        return [None] * n_cameras
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and n_cameras > 1:
            return list(value) * n_cameras
        return list(value)
    return [value] * n_cameras


def _install_randomizers(
    env,
    randomize_lighting: bool,
    randomize_color: bool,
    randomize_freq: int,
):
    """Monkey-patch a robomimic ``EnvRobosuite`` instance so it re-applies
    lighting / colour randomisation on every ``reset``/``reset_to`` and every
    ``randomize_freq`` steps.

    We patch the instance in place rather than wrapping it so that callers
    that reach into ``env.env`` (the underlying robosuite env) continue to
    work unchanged.
    """
    if not (randomize_lighting or randomize_color):
        return env

    try:
        from robosuite.utils.mjmod import LightingModder, TextureModder
    except Exception as exc:  # pragma: no cover - depends on robosuite ver
        warnings.warn(
            "robosuite LightingModder / TextureModder are unavailable "
            f"({exc}); domain randomisation will be skipped."
        )
        return env

    state = {"step": 0, "lighting": None, "texture": None}

    def _ensure_modders():
        sim = getattr(getattr(env, "env", env), "sim", None)
        if sim is None:
            return
        if randomize_lighting and state["lighting"] is None:
            state["lighting"] = LightingModder(sim=sim)
        if randomize_color and state["texture"] is None:
            state["texture"] = TextureModder(sim=sim)

    def _apply():
        if state["lighting"] is not None:
            state["lighting"].randomize()
        if state["texture"] is not None:
            state["texture"].randomize()

    orig_reset = env.reset
    orig_reset_to = env.reset_to
    orig_step = env.step

    def reset(*a, **kw):
        out = orig_reset(*a, **kw)
        state["step"] = 0
        _ensure_modders()
        _apply()
        return out

    def reset_to(*a, **kw):
        out = orig_reset_to(*a, **kw)
        state["step"] = 0
        _ensure_modders()
        _apply()
        return out

    freq = max(int(randomize_freq), 0)

    def step(action):
        out = orig_step(action)
        state["step"] += 1
        if freq > 0 and state["step"] % freq == 0:
            _apply()
        return out

    env.reset = reset
    env.reset_to = reset_to
    env.step = step
    return env


def _patch_env_kwargs(
    env_meta: dict,
    camera_names: Sequence[str],
    camera_depths: Iterable[bool] | None,
    camera_normals: Iterable[bool] | None,
    camera_segmentations,
    camera_height: int,
    camera_width: int,
    reward_shaping: bool,
) -> dict:
    """Bake the camera config into ``env_meta`` so ``robosuite.make`` sees it."""
    env_meta = copy.deepcopy(env_meta)
    env_kwargs = env_meta.setdefault("env_kwargs", {})

    env_kwargs["camera_names"] = list(camera_names)
    env_kwargs["camera_heights"] = camera_height
    env_kwargs["camera_widths"] = camera_width
    env_kwargs["reward_shaping"] = reward_shaping
    env_kwargs["use_camera_obs"] = True

    n = len(camera_names)
    depths = _coerce_per_camera(camera_depths, n)
    if any(bool(d) for d in depths):
        env_kwargs["camera_depths"] = [bool(d) for d in depths]

    segs = _coerce_per_camera(camera_segmentations, n)
    if any(s is not None and s != 0 for s in segs):
        env_kwargs["camera_segmentations"] = segs

    # ``camera_normals`` is fork-only.  We accept the kwarg here for
    # compatibility, but consumers should compute normals themselves from depth
    # if they need them.  We attach the request to the meta for later use.
    normals = _coerce_per_camera(camera_normals, n)
    if any(bool(x) for x in normals):
        env_meta["_vp2_camera_normals"] = [bool(x) for x in normals]

    return env_meta


def create_env_for_data_processing(
    env_meta,
    camera_names,
    camera_height,
    camera_width,
    reward_shaping,
    camera_depths=None,
    camera_normals=None,
    camera_segmentations=None,
    randomize_lighting=False,
    randomize_color=False,
    randomize_freq=0,
    renderer="mujoco",
):
    """Drop-in replacement for the s-tian fork's
    ``EnvUtils.create_env_for_data_processing`` that talks to the public
    robomimic API only."""
    if renderer not in (None, "mujoco"):
        warnings.warn(
            f"renderer={renderer!r} is only supported via the s-tian/robomimic "
            "depth_images fork; falling back to robosuite's MuJoCo renderer."
        )

    env_meta = _patch_env_kwargs(
        env_meta=env_meta,
        camera_names=camera_names,
        camera_depths=camera_depths,
        camera_normals=camera_normals,
        camera_segmentations=camera_segmentations,
        camera_height=camera_height,
        camera_width=camera_width,
        reward_shaping=reward_shaping,
    )

    use_depth_obs = bool(env_meta["env_kwargs"].get("camera_depths"))

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        use_depth_obs=use_depth_obs,
    )

    return _install_randomizers(
        env,
        randomize_lighting=randomize_lighting,
        randomize_color=randomize_color,
        randomize_freq=randomize_freq,
    )


# ---------------------------------------------------------------------------
# get_object_positions: replicates the helper that the fork exposed on its
# custom robosuite env class.  It looks for movable objects defined on the env
# and returns the world-frame positions of each.
# ---------------------------------------------------------------------------
def _objects_iter(robosuite_env):
    """Yield ``MujocoObject`` instances declared on a robosuite env."""
    for attr in ("objects", "movable_objects", "cubes"):
        objs = getattr(robosuite_env, attr, None)
        if objs:
            yield from objs
            return
    # Fallback: nothing structured we can find.
    return


def get_object_positions(env) -> list:
    """Return a list of (x, y, z) numpy arrays for every movable object.

    Works on a vanilla robosuite ``Environment``.  ``env`` may be either the
    robomimic ``EnvBase`` wrapper or the underlying robosuite env.
    """
    inner = getattr(env, "env", env)
    sim = getattr(inner, "sim", None)
    if sim is None:
        raise RuntimeError(
            "Underlying robosuite env exposes no `sim` - cannot read object "
            "positions."
        )

    positions = []
    for obj in _objects_iter(inner):
        body_name = None
        for cand in (
            getattr(obj, "root_body", None),
            getattr(obj, "name", None),
            getattr(obj, "naming_prefix", None),
        ):
            if cand and cand in sim.model.body_names:
                body_name = cand
                break
        if body_name is None:
            joint_names = getattr(obj, "joints", None) or []
            for j in joint_names:
                if j in sim.model.joint_names:
                    qpos_addr = sim.model.get_joint_qpos_addr(j)
                    if isinstance(qpos_addr, tuple):
                        positions.append(np.array(sim.data.qpos[qpos_addr[0] : qpos_addr[0] + 3]))
                    else:
                        positions.append(np.array(sim.data.qpos[qpos_addr : qpos_addr + 3]))
                    break
            continue
        positions.append(np.array(sim.data.get_body_xpos(body_name)))
    return positions
