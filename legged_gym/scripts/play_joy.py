# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import isaacgym
from isaacgym import gymapi, gymtorch
from torchvision.utils import save_image
import numpy as np
import torch

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from legged_gym.envs import *
from legged_gym.utils import Logger, export_policy_as_jit, get_args, task_registry

def manual_save_im(env, path):
    env.gym.start_access_image_tensors(env.sim)
    im = env.gym.get_camera_image_gpu_tensor(
        env.sim,
        env.envs[0],
        # env.camera_handles2[1],
        # env.follow_cam,
        env.floating_cam,
        gymapi.IMAGE_COLOR,
    )
    im = gymtorch.wrap_tensor(im)

    trans_im = im.detach().clone()
    trans_im = (trans_im[..., :3]).float() / 255
    save_image(
        trans_im.view((1080, 1920, 3)).permute(2, 0, 1).float(),
        path,
    )

    env.gym.end_access_image_tensors(env.sim)

class ViNLController:
    def __init__(self, args):
        self.args = args

        self.lin_speed = [0,0,0]
        self.ang_speed = 0
        self.estop = False

        self._teleop_cmd_sub = rospy.Subscriber("/cmd_vel", Twist, self.TeleopCmdCallback)
        self._teleop_stop_sub = rospy.Subscriber("/cmd_stop", Bool, self.TeleopStopCallback)
        rospy.init_node('policy_controller', anonymous=True)

    def TeleopCmdCallback(self, msg):
        self.lin_speed = [msg.linear.x, msg.linear.y, msg.linear.z]
        self.ang_speed = msg.angular.z
        print(self.lin_speed, self.ang_speed)

    def TeleopStopCallback(self, msg):
        self.lin_speed = [0,0,0]
        self.ang_speed = 0
        self.estop = True

    def play(self):
        # EXTREEEEME H-H-H-H-H-HACK!
        if self.args.seed is None:
            self.args.seed = 1
        os.environ["ISAAC_SEED"] = str(self.args.seed)
        os.environ["ISAAC_EPISODE_ID"] = str(self.args.episode_id)
        os.environ["ISAAC_NUM_COMPLETED_EPS"] = "0"
        scene = os.path.basename(self.args.map).split(".")[0]
        os.environ["ISAAC_MAP_NAME"] = scene
        os.environ["ISAAC_EVAL_DIR"] = self.args.eval_dir
        (
            os.environ["ISAAC_BLOCK_MIN_HEIGHT"],
            os.environ["ISAAC_BLOCK_MAX_HEIGHT"],
        ) = self.args.block.split("_")
        os.environ["ISAAC_WRITE"] = "True" if not self.args.no_write else "False"
        os.environ["ISAAC_WALL_SCALE"] = str(self.args.wall_scale)
        os.environ["ISAAC_HOR_SCALE"] = str(self.args.hor_scale)

        env_cfg, train_cfg = task_registry.get_cfgs(name=self.args.task)
        env_cfg.env.follow_cam = True
        env_cfg.env.float_cam = True
        if isinstance(env_cfg, AliengoNavCfg):
            env_cfg.terrain.map_path = self.args.map
        env_cfg.terrain.no_blocks = self.args.no_blocks
        if self.args.alt_ckpt != "":
            loaded_dict = torch.load(self.args.alt_ckpt, map_location="cpu")
            if "model_state_dict" not in loaded_dict:
                train_cfg.runner.resume_path = self.args.alt_ckpt
            else:
                env_cfg.env.use_dm = True
                train_cfg.runner.alt_ckpt = self.args.alt_ckpt
            del loaded_dict

        # override some parameters for testing
        env_cfg.env.num_envs = 1 # min(train_cfg.runner.num_test_envs, 50)
        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.curriculum = False
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False

        # prepare environment
        env, _ = task_registry.make_env(
            name=self.args.task, args=self.args, env_cfg=env_cfg, record=True
        )
        env.reset()
        obs = env.get_observations()
        # print("obs shape: ", obs.shape)
        # load policy
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=self.args.task, args=self.args, train_cfg=train_cfg
        )
        policy = ppo_runner.get_inference_policy(device=env.device)

        # export policy as a jit module (used to run it from C++)
        if EXPORT_POLICY:
            path = os.path.join(
                LEGGED_GYM_ROOT_DIR,
                "logs",
                train_cfg.runner.experiment_name,
                "exported",
                "policies",
            )
            export_policy_as_jit(ppo_runner.alg.actor_critic, path)
            print("Exported policy as jit script to: ", path)

        logger = Logger(env.dt)
        robot_index = 0  # which robot is used for logging
        joint_index = 1  # which joint is used for logging
        stop_state_log = 100  # number of steps before plotting states
        stop_rew_log = (
            env.max_episode_length + 1
        )  # number of steps before print average episode rewards
        camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
        camera_vel = np.array([1.0, 1.0, 0.0])
        camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
        img_idx = 0

        # obs_actions_motor = []
        # actions_record = np.zeros((1000, 12))
        # torque_record = np.zeros((1000, 12))
        # obs_record = np.zeros((1000, 48))

        for i in range(10 * int(env.max_episode_length)):
            robot_pos = env.root_states[0][:3]
            camera_pos = [robot_pos[0], robot_pos[1] + 3, robot_pos[2] + 1]
            env.set_camera(camera_pos, robot_pos)

            if train_cfg.runner.eval_baseline:
                actions = train_cfg.runner.baseline_policy(obs)
            else:
                env.commands[:,0] = self.lin_speed[0]
                env.commands[:,1] = self.lin_speed[1]
                env.commands[:,2] = self.ang_speed
                actions = policy(obs)

            # actions_record[i] = actions.detach().cpu()[0]
            # obs_record[i] = obs.detach().cpu()[0]
            # torque_record[i] = env._compute_torques(actions).detach().cpu()[0]

            # # print(obs.shape)
            # if i == 999:
            #     pickle.dump(actions_record, open("actions.p", "wb"))
            #     pickle.dump(obs_record, open("obs.p", "wb"))
            #     pickle.dump(torque_record, open("torque.p", "wb"))
            #     break

            obs, _, rews, dones, infos = env.step(actions.detach())
            if RECORD_FRAMES:
                filename = os.path.join(
                    # LEGGED_GYM_ROOT_DIR,
                    # "/home/simar/Projects/isaacVL/localDev/legged_gym",
                    # "/home/naoki/gt/vl/legged_gym",
                    "/home/dklee98/git/ai707_ws/ViNL",
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx:05}.png",
                )
                # print("saving at fp: ", filename)
                manual_save_im(env, filename)
                img_idx += 1
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

            if i < stop_state_log:
                logger.log_states(
                    {
                        "dof_pos_target": actions[robot_index, joint_index].item()
                        * env.cfg.control.action_scale,
                        "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                        "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                        "dof_torque": env.torques[robot_index, joint_index].item(),
                        "command_x": env.commands[robot_index, 0].item(),
                        "command_y": env.commands[robot_index, 1].item(),
                        "command_yaw": env.commands[robot_index, 2].item(),
                        "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                        "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                        "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                        "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                        "contact_forces_z": env.contact_forces[
                            robot_index, env.feet_indices, 2
                        ]
                        .cpu()
                        .numpy(),
                    }
                )
            elif i == stop_state_log:
                logger.plot_states()
            if 0 < i < stop_rew_log:
                if infos["episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes > 0:
                        logger.log_rewards(infos["episode"], num_episodes)
            elif i == stop_rew_log:
                logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    agent = ViNLController(args)
    agent.play()
