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

import pandas as pd
from datetime import datetime

from legged_gym.envs import *
from legged_gym.utils import Logger, export_policy_as_jit, get_args, task_registry

# STRAIGHT = 0
# RIGHT_TURN = 1
# LEFT_TURN = 2
# WAVE = 3
# LEFT_RIGHT = 4
# RIGHT_LEFT = 5

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

        self.lin_speed = []
        self.ang_speed = []
        self.lin_x = 0.5
        self.ang_z = 0.3

        # self.episode_type = WAVE

        """ 2023-10-30: Seunghyun modify """
        self.collision_count = 0
        self.traveled_distance = 0

        # hwlee: Velocity tracking performance
        self.df_vel          = pd.DataFrame(columns=['cmd_vel_x', 'base_vel_x', 'err_vel_x', 'cmd_vel_yaw', 'base_vel_yaw', 'err_vel_yaw', 'traveled_distance', 'collision_count', 'average_collision_count'])
        self.df_date         = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.df_folder       = f"{self.df_date}_vel"
        self.df_root_dir     = '/home/dklee98/git/ai707_ws/ViNL/logs/'
        
        ##### TODO: Modify directory ####
        if not os.path.exists(self.df_root_dir + self.df_folder):
            os.makedirs(self.df_root_dir + self.df_folder)
        self.df_file = 'df_vel'

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
        env_cfg.env.num_envs = 100 # min(train_cfg.runner.num_test_envs, 50)
        env_cfg.terrain.terrain_length = 10.0 # default 8, segfaults at 10
        env_cfg.terrain.terrain_width = 10.0 # default 8, segfaults at 10
        env_cfg.terrain.num_rows = 10
        env_cfg.terrain.num_cols = 10
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

        # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
        # camera_vel = np.array([1.0, 1.0, 0.0])
        # camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
        img_idx = 0

        # obs_actions_motor = []
        # actions_record = np.zeros((1000, 12))
        # torque_record = np.zeros((1000, 12))
        # obs_record = np.zeros((1000, 48))

        ###################################
        # cmd vel 생성하는 부분
        # 총 개수는 10 * episode_length
        # self.episode_size = int(env.max_episode_length)
        # self.generate_cmd_vel()
        ###################################

        for i in range(1 * int(env.max_episode_length + 2)):
            # robot_pos = env.root_states[0][:3]
            # camera_pos = [robot_pos[0], robot_pos[1] + 3, robot_pos[2] + 1]
            # env.set_camera(camera_pos, robot_pos)

            if train_cfg.runner.eval_baseline:
                actions = train_cfg.runner.baseline_policy(obs)
            else:
                # env.commands[:,0] = self.lin_speed[i % self.episode_size]
                # env.commands[:,1] = 0.0 # self.lin_speed[1]
                # env.commands[:,2] = self.ang_speed[i % self.episode_size]
                actions = policy(obs)

            """ 2023-10-30: Seunghyun modify """
            collision = torch.any(
                torch.norm(env.contact_forces[:, env.feet_indices, :2], dim=2)
                > 5 * torch.abs(env.contact_forces[:, env.feet_indices, 2]),
                dim=1,
            )
            
            self.traveled_distance += torch.sum(torch.norm(env.base_lin_vel[:, :2], dim=1)) * env.dt
            # self.traveled_distance += torch.sum(torch.square(env.base_lin_vel[:, :2]), dim=1) * env.dt
            # print("traveled_distance: ", self.traveled_distance)
            self.collision_count += torch.sum(collision)

            cmd_vel_x = env.commands[robot_index, 0].item()
            cmd_vel_yaw = env.commands[robot_index, 2].item()
            base_vel_x = env.base_lin_vel[robot_index, 0].item()
            base_vel_yaw = env.base_ang_vel[robot_index, 2].item()
            err_vel_x = np.linalg.norm((cmd_vel_x - base_vel_x))
            err_vel_yaw = np.linalg.norm((cmd_vel_yaw - base_vel_yaw))
            
            
            if self.traveled_distance.item() != 0:
                avg = self.collision_count.item() / self.traveled_distance.item()
            else:
                avg = 0
            self.df_vel.loc[i] = [cmd_vel_x, base_vel_x, err_vel_x, cmd_vel_yaw, base_vel_yaw, err_vel_yaw, self.traveled_distance.item(), self.collision_count.item(), avg]

            
            obs, _, rews, dones, infos = env.step(actions.detach())
            if RECORD_FRAMES:
                filename = os.path.join(
                    # LEGGED_GYM_ROOT_DIR,
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
                print("collision count: ", self.collision_count)
                print("traveled distance: ", self.traveled_distance)
                print("average collision count: ", avg)
                self.df_vel.to_csv(self.df_root_dir + self.df_folder + '/' + self.df_file + '.csv')
                print(f"velocity tracking performace is recorded")
                self.traveled_distance = 0
                self.collision_count = 0

    # def generate_cmd_vel(self):
    #     # print("episode size: ", self.episode_size)
    #     if self.episode_type == STRAIGHT:
    #         for i in range(self.episode_size):
    #             self.lin_speed.append(self.lin_x)
    #             self.ang_speed.append(0.0)
    #     elif self.episode_type == RIGHT_TURN:
    #         for i in range(self.episode_size):
    #             self.lin_speed.append(self.lin_x)
    #             self.ang_speed.append(-self.ang_z)
    #     elif self.episode_type == LEFT_TURN:
    #         for i in range(self.episode_size):
    #             self.lin_speed.append(self.lin_x)
    #             self.ang_speed.append(self.ang_z)
    #     elif self.episode_type == WAVE:
    #         for i in range(self.episode_size):
    #             if i < self.episode_size // 2:
    #                 self.lin_speed.append(self.lin_x)
    #                 self.ang_speed.append(self.ang_z)
    #             else:
    #                 self.lin_speed.append(self.lin_x)
    #                 self.ang_speed.append(-self.ang_z)
    #     elif self.episode_type == LEFT_RIGHT:
    #         for i in range(self.episode_size):
    #             if i < self.episode_size / 2:
    #                 self.lin_speed.append(self.lin_x)
    #                 self.ang_speed.append(-self.ang_z)
    #             else:
    #                 self.lin_speed.append(self.lin_x)
    #                 self.ang_speed.append(self.ang_z)       
    #     elif self.episode_type == RIGHT_LEFT:
    #         for i in range(self.episode_size):
    #             if i < self.episode_size / 2:
    #                 self.lin_speed.append(self.lin_x)
    #                 self.ang_speed.append(self.ang_z)
    #             else:
    #                 self.lin_speed.append(self.lin_x)
    #                 self.ang_speed.append(-self.ang_z)


if __name__ == "__main__":
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    args = get_args()
    agent = ViNLController(args)
    agent.play()
