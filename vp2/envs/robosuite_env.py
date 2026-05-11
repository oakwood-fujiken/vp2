import json
import os

import numpy as np
import gymnasium.spaces as spaces

from hydra.utils import to_absolute_path

import robomimic.utils.file_utils as FileUtils

from vp2.envs.base import BaseEnv
from vp2.envs._robosuite_compat import create_env_for_data_processing
from vp2.mpc.utils import resize_np_image_aa


class RobosuiteEnv(BaseEnv):
    def __init__(self, **kwargs):
        self.env_hparams = kwargs
        # get the file path of this file
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        env_meta = FileUtils.get_env_metadata_from_dataset(
            os.path.join(this_file_path, "robosuite_env_meta.hdf5")
        )

        self.keys_to_take = dict(
            rgb=f'{kwargs["camera_names"][0]}_image',
        )

        if "depth" in kwargs["planning_modalities"]:
            camera_depths = [True]
            self.keys_to_take["depth"] = f'{kwargs["camera_names"][0]}_depth'
        else:
            camera_depths = [False]

        if "normal" in kwargs["planning_modalities"]:
            camera_normals = [True]
            self.keys_to_take["normal"] = f'{kwargs["camera_names"][0]}_normal'
        else:
            camera_normals = [False]

        assert (
            len(kwargs["camera_names"]) == 1
        ), "Currently only one camera is supported!"

        self.env = create_env_for_data_processing(
            env_meta=env_meta,
            camera_names=list(kwargs["camera_names"]),
            camera_depths=camera_depths,
            camera_normals=camera_normals,
            camera_segmentations=[None],
            camera_height=kwargs["renderer_camera_height"],
            camera_width=kwargs["renderer_camera_width"],
            reward_shaping=kwargs["shaped"],
            randomize_lighting=kwargs["randomize_lighting"],
            randomize_color=kwargs["randomize_color"],
            randomize_freq=0,
            renderer=kwargs.get("renderer", "mujoco"),
        )
        print("==== Using environment with the following metadata ====")
        print(json.dumps(self.env.serialize(), indent=4))
        print("")

        self._disable_visualizations()

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=self.observation_shape
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dimension,))

    def _disable_visualizations(self):
        ### Prevent visualization from showing up in images
        vis_settings = dict()
        for vis in self.env.env._visualizations:
            vis_settings[vis] = False
        self.env.env.visualize(vis_settings)

    def reset(self):
        return self.env.reset()

    def reset_to(self, state):
        # robomimic's public `EnvBase.reset_to` resets from XML iff the state
        # dict contains a "model" key.  The s-tian fork instead exposed an
        # explicit `reset_from_xml` flag; we emulate that here.
        if not self.env_hparams.get("reset_xml", True):
            state = {k: v for k, v in state.items() if k != "model"}
        return self.env.reset_to(state)

    def reset_state(self, state):
        state = {k: v for k, v in state.items() if k != "model"}
        return self.env.reset_to(state)

    def get_image_obs(self, obs):
        return {k: obs[v] for k, v in self.keys_to_take.items()}

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = self.get_image_obs(obs)
        for key, value in obs.items():
            if len(value.shape) >= 3:  # resize observation images
                obs[key] = resize_np_image_aa(value.copy(), self.observation_shape)
        return obs, rew, done, info

    def get_state(self):
        return self.env.get_state()["states"]

    def compute_score(self, state, goal_state):
        from vp2.envs._robosuite_compat import get_object_positions

        self.env.reset_to({"states": state})
        obj_positions = get_object_positions(self.env.env)
        self.env.reset_to({"states": goal_state})
        goal_positions = get_object_positions(self.env.env)
        differences = list()
        for c, g in zip(obj_positions, goal_positions):
            differences.append(np.linalg.norm(c - g))
        return np.max(differences)

    @property
    def action_dimension(self):
        return self.env.action_dimension

    @property
    def observation_shape(self):
        return self.env_hparams["camera_height"], self.env_hparams["camera_width"]

    def goal_generator(self):
        goals_dataset_path = to_absolute_path(self.env_hparams["goals_dataset"])
        print(f"==== Loading goals from {goals_dataset_path} ====")
        yield from self.goal_generator_from_robosuite_hdf5(
            goals_dataset_path, self.env_hparams["camera_names"][0]
        )
