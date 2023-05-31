import numpy as np

_CONTROL_TIMESTEP = 0.025

class Skill():
    def __init__(self, domain_name, task_name):
        self.domain_name = domain_name
        self.task_name = task_name

        # for high knee run
        self.leg_touch_gnd = None # the leg touch gnd (left, right, both)
        self.leg_should_lift = "none" # the leg should lift

    def get_reward(self, qpos_before, named_geom_xpos_before, qpos_after, named_geom_xpos_after, physics, action):
        if self.task_name == "slow":
            self._x_vel_limit = 0.5
        else:
            self._x_vel_limit = int(self.task_name.split('_')[-1]) # get vel

        if self.domain_name == "forward_walker":          
            self._x_vel_reward = 1
            self._angle_reward = 0.1
            self._ctrl_penalty = 1e-3
            self._foot_penalty = 0.01
            self._height_penalty = 1
            self._fall_penalty = 5
            self._min_height = 0.8
            
            right_foot_before = named_geom_xpos_before["right_foot","x"]
            left_foot_before = named_geom_xpos_before["left_foot","x"]

            right_foot_after = named_geom_xpos_after["right_foot","x"]
            left_foot_after = named_geom_xpos_after["left_foot","x"]

            angle = physics.data.qpos[2]
            delta_h = physics.named.data.geom_xpos["torso","z"] - max(physics.named.data.geom_xpos["right_foot","z"], physics.named.data.geom_xpos["left_foot","z"])
            nz = np.cos(angle)
            x_vel = physics.horizontal_velocity()
            x_vel = self._x_vel_limit - abs(x_vel - self._x_vel_limit)
            right_foot_vel = abs(right_foot_after - right_foot_before) / _CONTROL_TIMESTEP
            left_foot_vel = abs(left_foot_after - left_foot_before) / _CONTROL_TIMESTEP

            # reward
            x_vel_reward = self._x_vel_reward * x_vel
            angle_reward = self._angle_reward * nz
            height_penalty = -self._height_penalty * abs(1.1 - delta_h)
            if action is None:
                ctrl_penalty = 0
            else:
                ctrl_penalty = -self._ctrl_penalty * np.sum(np.square(action))
            foot_penalty = -self._foot_penalty * (right_foot_vel + left_foot_vel)
            if physics.named.data.geom_xpos["torso","z"] < self._min_height:
                fall_penalty = -self._fall_penalty
            else:
                fall_penalty = 0


            reward = x_vel_reward + angle_reward + height_penalty + \
                    ctrl_penalty + foot_penalty + fall_penalty


            return reward

        elif self.domain_name == "backward_walker":
            self._x_vel_reward = 1
            self._angle_reward = 0.1
            self._ctrl_penalty = 1e-3
            self._foot_penalty = 0.01
            self._height_penalty = 1
            self._fall_penalty = 5
            self._min_height = 0.8
            
            right_foot_before = named_geom_xpos_before["right_foot","x"]
            left_foot_before = named_geom_xpos_before["left_foot","x"]

            right_foot_after = named_geom_xpos_after["right_foot","x"]
            left_foot_after = named_geom_xpos_after["left_foot","x"]

            height = named_geom_xpos_after["torso","z"]
            angle = physics.data.qpos[2]
            delta_h = physics.named.data.geom_xpos["torso","z"] - max(physics.named.data.geom_xpos["right_foot","z"], physics.named.data.geom_xpos["left_foot","z"])
            nz = np.cos(angle)
            x_vel = -physics.horizontal_velocity()
            x_vel = self._x_vel_limit - abs(x_vel - self._x_vel_limit)
            right_foot_vel = abs(right_foot_after - right_foot_before) / _CONTROL_TIMESTEP
            left_foot_vel = abs(left_foot_after - left_foot_before) / _CONTROL_TIMESTEP

            # reward
            x_vel_reward = self._x_vel_reward * x_vel
            angle_reward = self._angle_reward * nz
            height_penalty = -self._height_penalty * abs(1.1 - delta_h)
            if action is None:
                ctrl_penalty = 0
            else:
                ctrl_penalty = -self._ctrl_penalty * np.sum(np.square(action))
            foot_penalty = -self._foot_penalty * (right_foot_vel + left_foot_vel)
            if physics.named.data.geom_xpos["torso","z"] < self._min_height:
                fall_penalty = -self._fall_penalty
            else:
                fall_penalty = 0

            reward = x_vel_reward + angle_reward + height_penalty + \
                    ctrl_penalty + foot_penalty + fall_penalty

            return reward
        
        elif self.domain_name == "high_knee_run_walker":
            self._x_vel_reward = 1
            self._angle_reward = 0.1
            self._ctrl_penalty = 1e-3
            self._foot_penalty = 0.01
            self._delta_h_penalty = 0.5 # 5 fails
            self._walk_reward = 5
            self._fall_penalty = 5
            self._min_height = 0.8

            right_foot_before = named_geom_xpos_before["right_foot","x"]
            left_foot_before = named_geom_xpos_before["left_foot","x"]

            right_foot_after = named_geom_xpos_after["right_foot","x"]
            left_foot_after = named_geom_xpos_after["left_foot","x"]

            angle = physics.data.qpos[2]
            delta_h = physics.named.data.geom_xpos["torso","z"] - max(physics.named.data.geom_xpos["right_thigh","z"], physics.named.data.geom_xpos["left_thigh","z"])
            nz = np.cos(angle)
            x_vel = physics.horizontal_velocity()
            x_vel = self._x_vel_limit - abs(x_vel - self._x_vel_limit)
            right_foot_vel = abs(right_foot_after - right_foot_before) / _CONTROL_TIMESTEP
            left_foot_vel = abs(left_foot_after - left_foot_before) / _CONTROL_TIMESTEP
            
            if physics.named.data.geom_xpos["right_foot","z"] < 0.1 and \
                physics.named.data.geom_xpos["left_foot","z"] < 0.1:
                self.leg_touch_gnd = "both"
            elif physics.named.data.geom_xpos["right_foot","z"] < 0.1:
                self.leg_touch_gnd = "right"
            elif physics.named.data.geom_xpos["left_foot","z"] < 0.1:
                self.leg_touch_gnd = "left"
            else: # agent jump
                self.leg_touch_gnd = "none"

            # to determine which leg should be lifted initially
            if self.leg_should_lift=="none":
                if self.leg_touch_gnd=="left":
                    self.leg_should_lift = "right"
                elif self.leg_touch_gnd=="right":
                    self.leg_should_lift = "left"

            # reward
            x_vel_reward = self._x_vel_reward * x_vel
            angle_reward = self._angle_reward * nz
            delta_h_penalty = -self._delta_h_penalty * abs(delta_h)
            if action is None:
                ctrl_penalty = 0
            else:
                ctrl_penalty = -self._ctrl_penalty * np.sum(np.square(action))
            foot_penalty = -self._foot_penalty * (right_foot_vel + left_foot_vel)

            if physics.named.data.geom_xpos["left_thigh","z"] > 1 and physics.named.data.geom_xpos["right_thigh","z"] < 0.75 and \
                self.leg_should_lift=="left": # finish lift left leg
                walk_reward = self._walk_reward
                self.leg_should_lift = "right" # want it to lift right leg next
            elif physics.named.data.geom_xpos["right_thigh","z"] > 1 and physics.named.data.geom_xpos["left_thigh","z"] < 0.75 and \
                self.leg_should_lift=="right": # finish lift right leg
                walk_reward = self._walk_reward
                self.leg_should_lift = "left" # want it to lift left leg next
            else: walk_reward = 0

            if physics.named.data.geom_xpos["torso","z"] < self._min_height:
                fall_penalty = -self._fall_penalty
            else:
                fall_penalty = 0

            reward = x_vel_reward + angle_reward + delta_h_penalty + \
                    ctrl_penalty + foot_penalty + walk_reward + fall_penalty

            return reward

        elif self.domain_name == "crawl_walker":
            self._x_vel_reward = 1
            self._angle_reward = 0.1
            self._ctrl_penalty = 1e-3
            self._foot_penalty = 0.01
            self._height_penalty = 1
            self._crawl_reward = 3
            self._fall_penalty = 5
            self._min_height = 0.3

            right_foot_before = named_geom_xpos_before["right_foot","x"]
            left_foot_before = named_geom_xpos_before["left_foot","x"]

            right_foot_after = named_geom_xpos_after["right_foot","x"]
            left_foot_after = named_geom_xpos_after["left_foot","x"]

            angle = physics.data.qpos[2]
            delta_h = physics.named.data.geom_xpos["torso","z"] - max(physics.named.data.geom_xpos["right_foot","z"], physics.named.data.geom_xpos["left_foot","z"])
            nz = np.cos(angle)
            x_vel = physics.horizontal_velocity()
            x_vel = self._x_vel_limit - abs(x_vel - self._x_vel_limit)
            right_foot_vel = abs(right_foot_after - right_foot_before) / _CONTROL_TIMESTEP
            left_foot_vel = abs(left_foot_after - left_foot_before) / _CONTROL_TIMESTEP

            # reward
            x_vel_reward = self._x_vel_reward * x_vel
            angle_reward = self._angle_reward * nz
            height_penalty = -self._height_penalty * abs(1.1 - delta_h)
            if action is None:
                ctrl_penalty = 0
            else:
                ctrl_penalty = -self._ctrl_penalty * np.sum(np.square(action))
            foot_penalty = -self._foot_penalty * (right_foot_vel + left_foot_vel)
            crawl_reward = -self._crawl_reward * abs(physics.named.data.geom_xpos["torso","z"] - 0.5)

            if physics.named.data.geom_xpos["torso","z"] < self._min_height:
                fall_penalty = -self._fall_penalty
            else:
                fall_penalty = 0

            reward = x_vel_reward + angle_reward + height_penalty + \
                    ctrl_penalty + foot_penalty + crawl_reward + fall_penalty
            
            return reward

        elif self.domain_name == "jump_walker":
            self._x_vel_reward = 1 # change this from 2 to 1
            self._angle_reward = 0.1
            self._ctrl_penalty = 1e-3
            self._foot_penalty = 0.01
            self._delta_h_penalty = 1
            self._foot_diff_penalty = 1 # change this from 7 to 5
            self._thigh_diff_penalty = 1
            self._leg_diff_penalty = 1
            # self._jump_reward = 5
            self._not_jump_penalty = 3
            self._fall_penalty = 5
            self._min_height = 0.8
            
            right_foot_before = named_geom_xpos_before["right_foot","x"]
            left_foot_before = named_geom_xpos_before["left_foot","x"]

            right_foot_after = named_geom_xpos_after["right_foot","x"]
            left_foot_after = named_geom_xpos_after["left_foot","x"]

            angle = physics.data.qpos[2]
            delta_h = physics.named.data.geom_xpos["torso","z"] - max(physics.named.data.geom_xpos["right_foot","z"], physics.named.data.geom_xpos["left_foot","z"])
            nz = np.cos(angle)
            x_vel = physics.horizontal_velocity()
            x_vel = self._x_vel_limit - abs(x_vel - self._x_vel_limit)
            right_foot_vel = abs(right_foot_after - right_foot_before) / _CONTROL_TIMESTEP
            left_foot_vel = abs(left_foot_after - left_foot_before) / _CONTROL_TIMESTEP
            foot_diff = abs(physics.named.data.geom_xpos["right_foot","x"]-physics.named.data.geom_xpos["left_foot","x"]) + \
                abs(physics.named.data.geom_xpos["right_foot","z"]-physics.named.data.geom_xpos["left_foot","z"])
            thigh_diff = abs(physics.named.data.geom_xpos["right_thigh","x"]-physics.named.data.geom_xpos["left_thigh","x"]) + \
                abs(physics.named.data.geom_xpos["right_thigh","z"]-physics.named.data.geom_xpos["left_thigh","z"])
            leg_diff = abs(physics.named.data.geom_xpos["right_leg","x"]-physics.named.data.geom_xpos["left_leg","x"]) + \
                abs(physics.named.data.geom_xpos["right_leg","z"]-physics.named.data.geom_xpos["left_leg","z"])
            leg_to_gnd = min(physics.named.data.geom_xpos["right_foot","z"],physics.named.data.geom_xpos["left_foot","z"])

            # reward
            x_vel_reward = self._x_vel_reward * x_vel
            angle_reward = self._angle_reward * nz
            delta_h_penalty = -self._delta_h_penalty * abs(1.1 - delta_h)
            if action is None:
                ctrl_penalty = 0
            else:
                ctrl_penalty = -self._ctrl_penalty * np.sum(np.square(action))
            foot_penalty = -self._foot_penalty * (right_foot_vel + left_foot_vel)
            foot_diff_penalty = -self._foot_diff_penalty * foot_diff
            thigh_diff_penalty = -self._thigh_diff_penalty * thigh_diff
            leg_diff_penalty = -self._leg_diff_penalty * leg_diff
            if physics.named.data.geom_xpos["torso","z"] < self._min_height:
                fall_penalty = -self._fall_penalty
            else:
                fall_penalty = 0
            
            reward = x_vel_reward + angle_reward + delta_h_penalty + ctrl_penalty + foot_penalty + \
                foot_diff_penalty + thigh_diff_penalty + leg_diff_penalty + fall_penalty

            return reward

        elif self.domain_name == "jump_walker2":
            self._x_vel_reward = 1 # change this from 0.5 to 2
            self._angle_reward = 0.1
            self._ctrl_penalty = 1e-3
            self._foot_penalty = 0.01
            self._delta_h_penalty = 1
            # foot_diff_penalty, thigh_diff_penalty, leg_diff_penalty=1 are not suitable for vel_4-6
            self._foot_diff_penalty = 2
            self._thigh_diff_penalty = 2
            self._leg_diff_penalty = 2
            # self._jump_reward = 5
            self._not_jump_penalty = 3
            self._torso_height_reward = 1
            self._torso_z_speed_reward = 3
            self._right_position_penalty = 3
            self._fall_penalty = 5
            self._min_height = 0.8

            right_foot_before = named_geom_xpos_before["right_foot","x"]
            left_foot_before = named_geom_xpos_before["left_foot","x"]

            right_foot_after = named_geom_xpos_after["right_foot","x"]
            left_foot_after = named_geom_xpos_after["left_foot","x"]

            angle = physics.data.qpos[2]
            delta_h = physics.named.data.geom_xpos["torso","z"] - max(physics.named.data.geom_xpos["right_foot","z"], physics.named.data.geom_xpos["left_foot","z"])
            nz = np.cos(angle)
            x_vel = physics.horizontal_velocity()
            x_vel = self._x_vel_limit - abs(x_vel - self._x_vel_limit)
            right_foot_vel = abs(right_foot_after - right_foot_before) / _CONTROL_TIMESTEP
            left_foot_vel = abs(left_foot_after - left_foot_before) / _CONTROL_TIMESTEP
            foot_diff = abs(physics.named.data.geom_xpos["right_foot","x"]-physics.named.data.geom_xpos["left_foot","x"]) + \
                abs(physics.named.data.geom_xpos["right_foot","z"]-physics.named.data.geom_xpos["left_foot","z"])
            thigh_diff = abs(physics.named.data.geom_xpos["right_thigh","x"]-physics.named.data.geom_xpos["left_thigh","x"]) + \
                abs(physics.named.data.geom_xpos["right_thigh","z"]-physics.named.data.geom_xpos["left_thigh","z"])
            leg_diff = abs(physics.named.data.geom_xpos["right_leg","x"]-physics.named.data.geom_xpos["left_leg","x"]) + \
                abs(physics.named.data.geom_xpos["right_leg","z"]-physics.named.data.geom_xpos["left_leg","z"])
            leg_to_gnd = min(physics.named.data.geom_xpos["right_foot","z"],physics.named.data.geom_xpos["left_foot","z"])

            # reward
            x_vel_reward = self._x_vel_reward * x_vel
            angle_reward = self._angle_reward * nz
            delta_h_penalty = -self._delta_h_penalty * abs(1.1 - delta_h)
            if action is None:
                ctrl_penalty = 0
            else:
                ctrl_penalty = -self._ctrl_penalty * np.sum(np.square(action))
            foot_penalty = -self._foot_penalty * (right_foot_vel + left_foot_vel)
            foot_diff_penalty = -self._foot_diff_penalty * foot_diff
            thigh_diff_penalty = -self._thigh_diff_penalty * thigh_diff
            leg_diff_penalty = -self._leg_diff_penalty * leg_diff

            not_jump_penalty = -self._not_jump_penalty * abs(1 - leg_to_gnd)
            torso_height_reward = self._torso_height_reward * physics.named.data.geom_xpos["torso","z"]

            torso_z_speed_reward = self._torso_z_speed_reward * abs(physics.named.data.sensordata["torso_subtreelinvel"][2])

            if physics.named.data.geom_xpos["left_leg","z"] < physics.named.data.geom_xpos["left_foot","z"] or \
                physics.named.data.geom_xpos["right_leg","z"] < physics.named.data.geom_xpos["right_foot","z"] or \
                physics.named.data.geom_xpos["left_thigh","z"] < physics.named.data.geom_xpos["left_leg","z"] or \
                physics.named.data.geom_xpos["right_thigh","z"] < physics.named.data.geom_xpos["right_leg","z"] or \
                physics.named.data.geom_xpos["torso", "z"] < physics.named.data.geom_xpos["right_thigh","z"] or \
                physics.named.data.geom_xpos["torso", "z"] < physics.named.data.geom_xpos["left_thigh","z"]:
                right_position_penalty = -self._right_position_penalty
            else:
                right_position_penalty = 0

            if physics.named.data.geom_xpos["torso","z"] < self._min_height:
                fall_penalty = -self._fall_penalty
            else:
                fall_penalty = 0

            reward = x_vel_reward + angle_reward + delta_h_penalty + ctrl_penalty + foot_penalty + foot_diff_penalty + \
                    thigh_diff_penalty + leg_diff_penalty + not_jump_penalty + torso_height_reward + torso_z_speed_reward + right_position_penalty + fall_penalty

            return reward
