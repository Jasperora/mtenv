# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Skill Planar Walker Domain."""

from __future__ import absolute_import, division, print_function

import collections

from dm_control import mujoco
# from dm_control.rl import control
from . import walker_env as control
from dm_control.suite.utils import randomizers
from dm_control.utils import containers, rewards

from . import base, common

import numpy as np

from local_dm_control_suite.walker_skill_reward import Skill

_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = 0.025

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.2

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 8


SUITE = containers.TaggedTasks()


def get_model_and_assets(xml_file_id):
    """Returns a tuple containing the model XML string and a dict of assets."""
    if xml_file_id is not None:
        filename = f"walker_{xml_file_id}.xml"
        print(filename)
    else:
        filename = f"walker.xml"
    return common.read_model(filename), common.ASSETS



@SUITE.add("benchmarking")
def walk(
    time_limit=_DEFAULT_TIME_LIMIT,
    xml_file_id=None,
    random=None,
    environment_kwargs=None,
):
    """Returns the Walk task."""
    rank = int(xml_file_id.split('_')[-1])
    physics = Physics.from_xml_string(*get_model_and_assets(xml_file_id))
    task = SkillPlanarWalker(random=random, rank=rank)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )



class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Walker domain."""

    def torso_upright(self):
        """Returns projection from z-axes of torso to the z-axes of world."""
        return self.named.data.xmat["torso", "zz"]

    def torso_height(self):
        """Returns the height of the torso."""
        return self.named.data.xpos["torso", "z"]

    def horizontal_velocity(self):
        """Returns the horizontal velocity of the center-of-mass."""
        return self.named.data.sensordata["torso_subtreelinvel"][0]

    def orientations(self):
        """Returns planar orientations of all bodies."""
        return self.named.data.xmat[1:, ["xx", "xz"]].ravel()       


class SkillPlanarWalker(base.Task):
    """A planar walker task."""

    def __init__(self, random=None, rank=0):
        """Initializes an instance of `SkillPlanarWalker`.

        Args:
          move_speed: A float. If this value is zero, reward is given simply for
            standing up. Otherwise this specifies a target horizontal velocity for
            the walking task.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        print("random: ", random)
        self.seed(random)
        super(SkillPlanarWalker, self).__init__(random=random)

         # total 34 skills
        skill_list = [
            "forward_walker/vel_1",
            "forward_walker/vel_2",
            "forward_walker/vel_3",
            "forward_walker/vel_4",
            "forward_walker/vel_5",
            "forward_walker/vel_6",
            "forward_walker/vel_7",
            "forward_walker/vel_8",
            "backward_walker/vel_1",
            "backward_walker/vel_2",
            "backward_walker/vel_3",
            "backward_walker/vel_4",
            "backward_walker/vel_5",
            "backward_walker/vel_6",
            "backward_walker/vel_7",
            "backward_walker/vel_8",
            "high_knee_run_walker/slow",
            "high_knee_run_walker/vel_1",
            "crawl_walker/vel_1",
            "crawl_walker/vel_2",
            "crawl_walker/vel_3",
            "crawl_walker/vel_4",
            "crawl_walker/vel_5",
            "crawl_walker/vel_6",
            "jump_walker/vel_3",
            "jump_walker/vel_4",
            "jump_walker/vel_5",
            "jump_walker/vel_6",
            "jump_walker2/vel_1",
            "jump_walker2/vel_2",
            "jump_walker2/vel_3",
            "jump_walker2/vel_4",
            "jump_walker2/vel_5",
            "jump_walker2/vel_6",
        ]

        rank = rank - 1
        self.rank = rank
        skill_name = skill_list[self.rank]

        [self.domain_name, self.task_name] = skill_name.split("/")
        self._min_height = 0.8

        self.init_qpos = None
        self.init_named_geom_xpos = None
        self.qpos_before = None
        self.qpos_after = None
        self.named_geom_xpos_before = None
        self.named_geom_xpos_after = None
        self.action = None
        
    def seed(self, seed):
        np.random.seed(seed)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.
        In 'standing' mode, use initial orientation and small velocities.
        In 'random' mode, randomize joint angles and let fall to the floor.
        Args:
          physics: An instance of `Physics`.
        """
        # randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        if self.init_qpos is None:
            self.init_qpos = physics.data.qpos.copy()
            self.init_qvel = physics.data.qvel.copy()
        if self.init_named_geom_xpos is None:
            self.init_named_geom_xpos = physics.named.data.geom_xpos
        physics.data.qpos = self.init_qpos + (np.random.random(self.init_qpos.shape)-.5) * 1e-2
        physics.data.qvel = self.init_qvel + (np.random.random(self.init_qvel.shape)-.5) * 1e-2
        # randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of body orientations, height and velocites."""
        obs = collections.OrderedDict()
        obs["orientations"] = physics.orientations()
        obs["height"] = physics.torso_height()
        obs["velocity"] = physics.velocity()
        return obs

    def before_step(self, action, physics):
        """Sets the control signal for the actuators to values in `action`."""
        # Support legacy internal code.
        action = getattr(action, "continuous_actions", action)
        physics.set_control(action)
        self.action = action.copy()
        self.qpos_before = physics.data.qpos.copy()
        self.named_geom_xpos_before = physics.named.data.geom_xpos

    def after_step(self, physics):
        """Modifies colors according to the reward."""
        if self._visualize_reward:
            # reward = np.clip(self.get_reward(physics), 0.0, 1.0)
            reward = self.get_reward(physics)
            # _set_reward_colors(physics, reward)
        self.qpos_after = physics.data.qpos.copy()
        self.named_geom_xpos_after = physics.named.data.geom_xpos
        
    @property
    def is_healthy(self):
        z, angle = self.state_after[0], self.state_after[2]

        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle

        return is_healthy

    def get_termination(self, physics):
        # height = physics.named.data.geom_xpos["torso","z"]
        # terminated = height < self._min_height

        # not terminate until fixed timesteps
        terminated = False
        if terminated:
            return 0.

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        if self.qpos_before is None:
            qpos_before = self.init_qpos
        else:
            qpos_before = self.qpos_before
        if self.named_geom_xpos_before is None:
            named_geom_xpos_before = self.init_named_geom_xpos
        else:
            named_geom_xpos_before = self.named_geom_xpos_before

        if self.qpos_after is None:
            qpos_after = qpos_before
        else:
            qpos_after = self.qpos_after
        if self.named_geom_xpos_after is None:
            named_geom_xpos_after = named_geom_xpos_before
        else:
            named_geom_xpos_after = self.named_geom_xpos_after

        skill = Skill(self.domain_name, self.task_name)

        return skill.get_reward(qpos_before, named_geom_xpos_before, qpos_after, named_geom_xpos_after, physics, self.action)
                
    
