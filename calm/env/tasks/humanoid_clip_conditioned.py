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
import logging
# import env.tasks.humanoid as humanoid
# import env.tasks.humanoid_amp as humanoid_amp
# import env.tasks.humanoid_amp_task as humanoid_amp_task
# from utils import torch_utils

from isaacgym.torch_utils import *
from isaacgym import gymapi, gymutil, gymtorch
import time
from PIL import Image as Im
from matplotlib import pyplot as plt

from env.tasks.humanoid_heading import HumanoidHeading
from utils import torch_utils
# from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples

# from isaacgym.torch_utils import *
# import clip
from transformers import CLIPProcessor, CLIPModel

TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2
RANGE = 8

class HumanoidHeadingConditioned(HumanoidHeading):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._tar_locomotion_index = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)
        self._corresponding_speeds = torch.tensor([6, 3.5, 3.5], device=self.device, dtype=torch.float32)

        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 256
        self.camera_props.height = 256
        self.camera_props.enable_tensors = True

        self.clip_model = CLIPModel.from_pretrained("./clip")
        self.clip_processor = CLIPProcessor.from_pretrained("./clip")

        self.frame = 0
        self.similarity = 0.0
        self.memory = {}

        return


    def _adjust_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0],
                                  char_root_pos[1] + cam_delta[1],
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return new_cam_pos, new_cam_target

    def render(self, sync_frame_time=False):
        super(HumanoidHeadingConditioned, self).render()
        cam_pos, cam_target = self._adjust_camera()
        self.render_img(cam_pos, cam_target)

    def render_img(self, pos, target):
        if self.viewer:
            if self.enable_viewer_sync:
                self.frame += 1
                # if self.frame > 150 and self.frame%300 < RANGE:
                if self.frame > 150 and self.frame%300 == 1:
                    # set image render system
                    # print(self.frame)
                    self.gym.render_all_camera_sensors(self.sim)
                    self.gym.start_access_image_tensors(self.sim)

                    camera_handle = self.camera_handles[0]

                    pos_nearer = gymapi.Vec3(pos.x + 1.2, pos.y + 1.2, pos.z)
                    # 16 envs its possible for parallel working -> but its hard to merge the reward
                    self.gym.set_camera_location(camera_handle, self.envs[0], pos_nearer, target)

                    # start = time.time()
                    camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], camera_handle,
                                                                              gymapi.IMAGE_COLOR)
                    torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
                    camera_image = torch_rgba_tensor.cpu().numpy()
                    # end1 = time.time()
                    camera_image = Im.fromarray(camera_image)
                    self._similarity(camera_image)


                    # print("Success acquire the image of frame ", self.frame)

                    # # =================== debug the camera output ===================
                    # camera_image.save("./images/{}.png".format(self.frame))
                    # end2 = time.time()

                    # print("time of render is ", (end1 - start) * 1000)
                    # print("time of save is ", (end2 - start) * 1000)
                    # plt.imshow(camera_image)
                    # plt.pause(1e-9)
                    self.gym.end_access_image_tensors(self.sim)

    def _similarity(self, image):
        # print("==================== compute the similarity of CLIP ====================")
        # image = self.render(sync_frame_time=False)
        inputs = self.clip_processor(text=["waves like bird"], images=image,
        # inputs = self.clip_processor(text=["eating apple with hand"], images=image,
                           return_tensors="pt", padding=True)
        # inputs = self.clip_processor(text=["eating an apple"], images=image,
        #                              return_tensors="pt", padding=True)

        outputs = self.clip_model(**inputs)
        # end2 = time.time()
        # print("Time of model running is ", (end2 - start))
        self.similarity = outputs.logits_per_image  # this is the image-text similarity score
        self.memory[self.frame] = self.similarity
        np.save("./output/sim.npy", self.memory)
        # end3 = time.time()
        # print("Time of similarity calculation is ", (end3 - end1))
        #
        # probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        # end4 = time.time()
        # print("Time of softmax is ", (end4 - start))

    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            obs_size = 2 + 1
        return obs_size

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            tar_dir = self._tar_dir
            tar_locomotion_index = self._tar_locomotion_index
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_dir = self._tar_dir[env_ids]
            tar_locomotion_index = self._tar_locomotion_index[env_ids]

        obs = compute_heading_observations(root_states, tar_dir, tar_locomotion_index)
        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        self.rew_buf[:] = compute_heading_reward(root_pos, self._prev_root_pos,  root_rot, self._tar_dir,
                                                 self._tar_speed, self.dt, self.similarity)
        return

    def _reset_task(self, env_ids):
        n = len(env_ids)

        if n > 0:
            rand_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi

            change_steps = torch.randint(low=self._heading_change_steps_min, high=self._heading_change_steps_max,
                                         size=(n,), device=self.device, dtype=torch.int64)

            tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)

            tar_locomotion_index = torch.randint(low=0, high=3, size=(n,), device=self.device, dtype=torch.int64)

            self._tar_dir[env_ids] = tar_dir
            self._tar_facing_dir[env_ids] = tar_dir
            self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
            self._tar_locomotion_index[env_ids] = tar_locomotion_index
            self._tar_speed[env_ids] = self._corresponding_speeds[tar_locomotion_index]

        return

    def post_physics_step(self):
        super().post_physics_step()


@torch.jit.script
def compute_heading_observations(root_states, tar_dir, tar_locomotion_index):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    root_rot = root_states[:, 3:7]

    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_tar_dir = quat_rotate(heading_rot, tar_dir3d)
    local_tar_dir = local_tar_dir[..., 0:2]

    obs = torch.cat([local_tar_dir, tar_locomotion_index.view(-1, 1)], dim=-1)
    return obs


@torch.jit.script
def compute_heading_reward(root_pos, prev_root_pos, root_rot, tar_dir, tar_speed, dt, similarity):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    vel_err_scale = 0.25
    dir_err_scale = 2.0
    clip_err_scale = 2.0

    vel_reward_w = 0.3
    face_reward_w = 0.5
    clip_reward_w = 0.5

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

    # add one more reward for text_guided training
    clip_reward = torch.exp(-clip_err_scale * similarity)

    reward = vel_reward_w * vel_reward + clip_reward_w * clip_reward
    # reward = vel_reward_w * vel_reward + face_reward_w * move_dir_reward + clip_reward_w * clip_reward

    return reward
