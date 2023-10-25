# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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

import torch

from isaacgym import gymtorch

from env.tasks.humanoid_amp_getup import HumanoidAMPGetup
from isaacgym.torch_utils import *


class HumanoidRenderAnySKill(HumanoidAMPGetup):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        
        self._recovery_episode_prob = cfg["env"]["recoveryEpisodeProb"]
        self._recovery_steps = cfg["env"]["recoverySteps"]
        self._fall_init_prob = cfg["env"]["fallInitProb"]
        self._tar_speed_min = cfg["env"]["tarSpeedMin"]
        self._tar_speed_max = cfg["env"]["tarSpeedMax"]
        self._heading_change_steps_min = cfg["env"]["headingChangeStepsMin"]
        self._heading_change_steps_max = cfg["env"]["headingChangeStepsMax"]
        self._enable_rand_heading = cfg["env"]["enableRandHeading"]

        self._reset_fall_env_ids = []

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._recovery_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        self._generate_fall_states()
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._tar_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._tar_dir[..., 0] = 1.0
        self._tar_facing_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._tar_facing_dir[..., 0] = 1.0

        self._tar_speed = torch.ones([self.num_envs], device=self.device, dtype=torch.float)
        self._heading_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
        self._similarity = torch.zeros([self.num_envs], device=self.device, dtype=torch.float32)
        self._punish_count = torch.zeros([self.num_envs], device=self.device, dtype=torch.int)
        # self.torch_rgba_tensor = torch.zeros([self.num_envs, 224, 224, 3], device=self.device, dtype=torch.float32)

        return

    def render(self, sync_frame_time=False):
        super(HumanoidRenderAnySKill, self).render()
        self._frame += 1
        if self._frame > 150 and self._frame%30 == 1:
            self.gym.refresh_actor_root_state_tensor(self.sim)
            char_root_pos = self._humanoid_root_states[:, 0:3].cpu().numpy()
            char_root_rot = self._humanoid_root_states[:, 3:7].cpu().numpy()
            self._cam_prev_char_pos[:] = char_root_pos

            for env_id in range(self.num_envs):
                cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
                cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
                cam_delta = cam_pos - self._cam_prev_char_pos[env_id]

                target = gymapi.Vec3(char_root_pos[env_id, 0], char_root_pos[env_id, 1], 1.0)
                pos = gymapi.Vec3(char_root_pos[env_id, 0] + cam_delta[0],
                                  char_root_pos[env_id, 1] + cam_delta[1],
                                  cam_pos[2])

                self.gym.viewer_camera_look_at(self.viewer, None, pos, target)
                pos_nearer = gymapi.Vec3(pos.x + 1.2, pos.y + 1.2, pos.z)
                self.gym.set_camera_location(self.camera_handles[env_id], self.envs[env_id], pos_nearer, target)

            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            for env_id in range(self.num_envs):
                camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_id], self.camera_handles[env_id],
                                                                          gymapi.IMAGE_COLOR)
                self.torch_rgba_tensor[env_id] = gymtorch.wrap_tensor(camera_rgba_tensor)[:, :, :3].float()  # [224,224,3] -> IM -> [env,224,224,3]

            print("time of render {} frames' image: {}".format(env_id, (time.time() - start)))

            self.gym.end_access_image_tensors(self.sim)

        return self.torch_rgba_tensor.permute(0, 3, 1, 2)


    def get_task_obs_size(self):
        obs_size = 512  # dim for text_latents
        return obs_size

    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)

        anyskill_obs = torch.zeros((humanoid_obs.shape[0], 512), device=self.device)
        obs = torch.cat([humanoid_obs, anyskill_obs], dim=-1)
        # obs = humanoid_obs
        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return

    def get_obs_size(self):
        obs_size = super().get_obs_size()
        return obs_size + 512

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        self._update_recovery_count()
        return

    def post_physics_step(self):
        super().post_physics_step()
        self.extras["state_embeds"] = self._rigid_state_tensor[:, :, :]

    def _generate_fall_states(self):
        max_steps = 150
        
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        root_states = self._initial_humanoid_root_states[env_ids].clone()
        root_states[..., 3:7] = torch.randn_like(root_states[..., 3:7])
        root_states[..., 3:7] = torch.nn.functional.normalize(root_states[..., 3:7], dim=-1)
        self._humanoid_root_states[env_ids] = root_states
        
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        rand_actions = np.random.uniform(-0.5, 0.5, size=[self.num_envs, self.get_action_size()])
        rand_actions = to_torch(rand_actions, device=self.device)
        self.pre_physics_step(rand_actions)

        # step physics and render each frame
        for i in range(max_steps):
            self.render()
            self.gym.simulate(self.sim)
            
        self._refresh_sim_tensors()
        
        self._fall_root_states = self._humanoid_root_states.clone()
        self._fall_root_states[:, 7:13] = 0
        self._fall_dof_pos = self._dof_pos.clone()
        self._fall_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        return

    def _reset_actors(self, env_ids):
        num_envs = env_ids.shape[0]
        recovery_probs = to_torch(np.array([self._recovery_episode_prob] * num_envs), device=self.device)
        recovery_mask = torch.bernoulli(recovery_probs) == 1.0
        terminated_mask = (self._terminate_buf[env_ids] == 1)
        recovery_mask = torch.logical_and(recovery_mask, terminated_mask)

        recovery_ids = env_ids[recovery_mask]
        if len(recovery_ids) > 0:
            self._reset_recovery_episode(recovery_ids)
            

        nonrecovery_ids = env_ids[torch.logical_not(recovery_mask)]
        fall_probs = to_torch(np.array([self._fall_init_prob] * nonrecovery_ids.shape[0]), device=self.device)
        fall_mask = torch.bernoulli(fall_probs) == 1.0
        fall_ids = nonrecovery_ids[fall_mask]
        if len(fall_ids) > 0:
            self._reset_fall_episode(fall_ids)
            

        nonfall_ids = nonrecovery_ids[torch.logical_not(fall_mask)]
        if len(nonfall_ids) > 0:
            super()._reset_actors(nonfall_ids)
            self._recovery_counter[nonfall_ids] = 0

        return

    def _reset_recovery_episode(self, env_ids):
        self._recovery_counter[env_ids] = self._recovery_steps
        return
    
    def _reset_fall_episode(self, env_ids):
        fall_state_ids = torch.randint_like(env_ids, low=0, high=self._fall_root_states.shape[0])
        self._humanoid_root_states[env_ids] = self._fall_root_states[fall_state_ids]
        self._dof_pos[env_ids] = self._fall_dof_pos[fall_state_ids]
        self._dof_vel[env_ids] = self._fall_dof_vel[fall_state_ids]
        self._recovery_counter[env_ids] = self._recovery_steps
        self._reset_fall_env_ids = env_ids
        return
    
    def _reset_envs(self, env_ids):
        self._reset_fall_env_ids = []
        super()._reset_envs(env_ids)
        return

    def _init_amp_obs(self, env_ids):
        super()._init_amp_obs(env_ids)

        if len(self._reset_fall_env_ids) > 0:
            self._init_amp_obs_default(self._reset_fall_env_ids)

        return

    def _update_recovery_count(self):
        self._recovery_counter -= 1
        self._recovery_counter = torch.clamp_min(self._recovery_counter, 0)
        return

    def _compute_reset(self):
        super()._compute_reset()

        is_recovery = self._recovery_counter > 0
        self.reset_buf[is_recovery] = 0
        self._terminate_buf[is_recovery] = 0
        return

    def compute_anyskill_reward(self, img_features_norm, text_features_norm, corresponding_id):
        similarity = 100 * torch.einsum('ij,ij->i', img_features_norm, text_features_norm[corresponding_id])

        clip_err_scale = 0.15
        clip_reward_w = 0.98
        sim_mask = similarity <= 22
        similarity[sim_mask] = 0
        # similarity_bar = torch.mean(similarity)
        clip_reward = torch.exp(clip_err_scale * similarity)
        return clip_reward_w * clip_reward

    def _update_task(self):
        reset_task_mask = self.progress_buf >= self._heading_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self._reset_task(rest_env_ids)
        return

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        self.rew_buf[:] = compute_aux_reward(root_pos, self._prev_root_pos, self._tar_dir,
                                             self._tar_speed, self.dt)
        return

    def _reset_task(self, env_ids):
        n = len(env_ids)
        if self._enable_rand_heading:
            rand_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            rand_face_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
        else:
            rand_theta = torch.zeros(n, device=self.device)
            rand_face_theta = torch.zeros(n, device=self.device)

        tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)
        tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n,
                                                                             device=self.device) + self._tar_speed_min
        change_steps = torch.randint(low=self._heading_change_steps_min, high=self._heading_change_steps_max,
                                     size=(n,), device=self.device, dtype=torch.int64)

        face_tar_dir = torch.stack([torch.cos(rand_face_theta), torch.sin(rand_face_theta)], dim=-1)

        self._tar_speed[env_ids] = tar_speed
        self._tar_dir[env_ids] = tar_dir
        self._tar_facing_dir[env_ids] = face_tar_dir
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        return


def compute_aux_reward(root_pos, prev_root_pos, tar_dir, tar_speed, dt):
    vel_err_scale = 0.25
    dir_err_scale = 2.0

    vel_reward_w = 0.01
    face_reward_w = 0.01

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)

    tar_vel_err = tar_speed - tar_dir_speed
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    movement_dir = torch.nn.functional.normalize(root_vel[..., :2], dim=-1)
    move_dir_err = tar_dir - movement_dir
    move_dir_reward = torch.exp(-dir_err_scale * torch.norm(move_dir_err, dim=-1))

    reward = vel_reward_w * vel_reward + face_reward_w * move_dir_reward

    return reward

