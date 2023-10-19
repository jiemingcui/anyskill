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

# import env.tasks.humanoid as humanoid
# import env.tasks.humanoid_amp as humanoid_amp
# import env.tasks.humanoid_amp_task as humanoid_amp_task
# from utils import torch_utils

from isaacgym.torch_utils import *
from isaacgym import gymapi, gymutil, gymtorch
import time
from enum import Enum
from PIL import Image as Im
from matplotlib import pyplot as plt

from env.tasks.humanoid_location import HumanoidLocation
from utils import torch_utils
from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
# from transformers import CLIPProcessor, CLIPModel
import open_clip


TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2
RANGE = 8

class MovementType(Enum):
    RUN = 0
    WALK = 1
    CROUCH = 2

class IdleType(Enum):
    STAND = 7
    ROAR = 8
    CROUCH = 9

class HumanoidLocationConditioned(HumanoidLocation):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self._tar_change_steps_min = 310
        self._tar_change_steps_max = 320

        self._should_strike = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._stay_idle = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._tar_height = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)
        self._success = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)

        self.movement_type = MovementType[cfg["env"]["movementType"].upper()]
        self.idle_type = IdleType[cfg["env"]["idleType"].upper()]
        self.motions_dict = {
            IdleType.STAND: {
                MovementType.RUN: 7,
                MovementType.WALK: 3,
                MovementType.CROUCH: 3
            },
            IdleType.ROAR: {
                MovementType.RUN: 9,
                MovementType.WALK: 3,
                MovementType.CROUCH: 3
            },
            IdleType.CROUCH: {
                MovementType.RUN: 7,
                MovementType.WALK: 3,
                MovementType.CROUCH: 3
            }
        }

        self._strike_index = 0
        self._idle_index = self.idle_type.value

        self._enable_early_termination = False
        self._near_prob = 0

        self._tar_dist_max = 10.0

        self._total_episodes = None
        self._total_successes = 0

        # self._tar_locomotion_index = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)
        self._memory = {}
        self.clip_features = []
        self.motionclip_features = []
        # self._memory = torch.zeros([RANGE, 3, 256, 256], device=self.device, dtype=torch.float32)
        self._similarity = torch.zeros([RANGE], device=self.device, dtype=torch.float32)
        # self._corresponding_speeds = torch.tensor([6, 3.5, 3.5], device=self.device, dtype=torch.float32)

        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 256
        self.camera_props.height = 256
        self.camera_props.enable_tensors = True

        self.mlip_model, _, self.mlip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        self.frame = 0
        # self._similarity = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)

        return

    def _process(self, n_px):
        return Compose([
            # Resize(size, interpolation=InterpolationMode.BICUBIC),
            # CenterCrop(size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            #still don't know the reason of those mean and std
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])(n_px)

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
        super(HumanoidLocationConditioned, self).render()
        # for i in range(len(self.envs)):
        cam_pos, cam_target = self._adjust_camera()
        self.render_img(cam_pos, cam_target)


    def render_img(self, pos, target, env_id=None):
        self.frame += 1
        if self.frame > 150 and self.frame%5 == 1:
        # if self.frame > 150 and self.frame%60 == 1:
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
            self._similarity[self.frame%5] = self._calc_similarity(camera_image)
            # self._similarity[self.frame%60] = self._calc_similarity(camera_image)
            # self.memory[self.frame%30, :, :, :] = self._process(camera_image.convert("RGB"))

            self.gym.end_access_image_tensors(self.sim)

        # elif self.frame > 150 and self.frame%30 == RANGE:
        #     print("When the frame is {}, we start to compute the similarity".format(self.frame))
        #     self._similarity(video_seq=torch.tensor(self.memory))


    def _calc_similarity(self, image):
        # print("==================== compute the similarity of CLIP ====================")
        # image = self.render(sync_frame_time=False)
        raw_caption = self.cfg['env']['caption']
        caption = raw_caption.replace("_", " ")

        img = self.mlip_preprocess(image).unsqueeze(0)
        text = self.tokenizer([caption])

        image_features = self.mlip_model.encode_image(img)
        text_features = self.mlip_model.encode_text(text)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = image_features_norm @ text_features.T  # this is the image-text similarity score
        # similarity = 100.0 * image_features_norm @ text_features.T  # this is the image-text similarity score


        # end2 = time.time()
        # print("Time of model running is ", (end2 - start))

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rigid_state_tensor = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = rigid_state_tensor.shape[0] // self.num_envs
        rigid_state_tensor = rigid_state_tensor.view(self.num_envs, bodies_per_env, 13)
        state_embeds = rigid_state_tensor[0, :, :]

        # train_features_img = torch.cat((state_embeds, outputs.text_embeds), 1)
        # train_features_motion = torch.cat((outputs.image_embeds, outputs.text_embeds), 0)

        # count = int((self.frame - 150)/300)
        # self.memory[count] = self.similarity

        self.clip_features.append(image_features.data.cpu().numpy())
        self.motionclip_features.append(state_embeds.data.cpu().numpy())

        # train_features = torch.cat((outputs.image_embeds, outputs.text_embeds), 0)
        # similarity = outputs.logits_per_image  # this is the image-text similarity score
        # # count = int((self.frame - 150)/300)
        # # self.memory[count] = self.similarity
        # self.clip_features.append(train_features.data.cpu().numpy())
        # np.save("./output/train_feature1.npy", self.clip_features)
        #
        # self._memory[self.frame] = similarity
        # np.save("./output/sim_location.npy", self._memory)

        return similarity
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
        root_pos = self._humanoid_root_states[..., 0:2]
        prev_root_pos = self._prev_root_pos[..., 0:2]
        tar_pos = self._tar_pos

        pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]

        pos_err = torch.sum(pos_diff * pos_diff, dim=-1)

        idle_distance = self.motions_dict[self.idle_type][self.movement_type]
        if self.movement_type == MovementType.RUN:
            dist_mask = pos_err < idle_distance
            self._tar_height[dist_mask] = MovementType.WALK.value
            idle_distance = self.motions_dict[self.idle_type][MovementType.WALK]

        dist_mask = pos_err < idle_distance
        self._stay_idle[dist_mask] = 1

        dist_mask = pos_err >= idle_distance
        self._stay_idle[dist_mask] = 0

        dist_mask = pos_err < 1
        delta_root_pos = root_pos - prev_root_pos
        root_vel = torch.sum(delta_root_pos * delta_root_pos / self.dt, dim=-1)
        idle_chars = root_vel < 0.5
        idle_and_close = torch.logical_and(idle_chars, dist_mask)

        self._success[idle_and_close] = 1

        if env_ids is None:
            root_states = self._humanoid_root_states
            tar_pos = self._tar_pos
            tar_height = self._tar_height
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_pos = self._tar_pos[env_ids]
            tar_height = self._tar_height[env_ids]

        obs = compute_location_heading_observations(root_states, tar_pos, tar_height)
        return obs


    def _compute_reward(self, actions):
        # root_pos = self._humanoid_root_states[..., 0:3]
        # root_rot = self._humanoid_root_states[..., 3:7]
        # self.rew_buf[:] = compute_heading_reward(root_pos, self._prev_root_pos,  root_rot, self._tar_pos,
        #                                          self._tar_speed, self.dt, self._similarity)
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        self.rew_buf[:], self.rew_pos[:], self.rew_vel[:], self.rew_face[:], self.rew_clip[:] = compute_location_reward(root_pos, self._prev_root_pos, root_rot,
                                                  self._tar_pos, self._tar_speed,
                                                  self.dt, self._similarity)
        return


    def _reset_task(self, env_ids):
        n = len(env_ids)

        if self._total_episodes is not None:
            if n > 0:
                self._total_episodes += n
                self._total_successes += torch.sum(self._success[env_ids])

                print(
                    f'Current success rate: {self._total_successes * 100. / self._total_episodes}%, out of {self._total_episodes} episodes')
        else:
            self._total_episodes = 0

        char_root_pos = self._humanoid_root_states[env_ids, 0:2]
        rand_pos = self._tar_dist_max * (2.0 * torch.rand([n, 2], device=self.device) - 1.0)

        change_steps = torch.randint(low=self._tar_change_steps_min, high=self._tar_change_steps_max,
                                     size=(n,), device=self.device, dtype=torch.int64)

        self._should_strike[env_ids] = 0
        self._stay_idle[env_ids] = 0
        self._success[env_ids] = 0

        self._tar_pos[env_ids] = char_root_pos
        # self._tar_pos[env_ids] = char_root_pos + rand_pos
        self._tar_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps

        self._tar_height[env_ids] = self.movement_type.value

        return

    def post_physics_step(self):
        super().post_physics_step()

# @torch.jit.script
# def compute_heading_observations(root_states, tar_dir, tar_locomotion_index):
#     # type: (Tensor, Tensor, Tensor) -> Tensor
#     root_rot = root_states[:, 3:7]
#
#     tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
#     heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
#
#     local_tar_dir = quat_rotate(heading_rot, tar_dir3d)
#     local_tar_dir = local_tar_dir[..., 0:2]
#
#     obs = torch.cat([local_tar_dir, tar_locomotion_index.view(-1, 1)], dim=-1)
#     return obs
#
#
# @torch.jit.script
# def compute_heading_reward(root_pos, prev_root_pos, root_rot, tar_dir, tar_speed, dt, similarity):
#     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor) -> Tensor
#     vel_err_scale = 0.25
#     dir_err_scale = 2.0
#     clip_err_scale = 0.15
#
#     vel_reward_w = 0.1
#     face_reward_w = 0.5
#     clip_reward_w = 0.6
#
#     delta_root_pos = root_pos - prev_root_pos
#     root_vel = delta_root_pos / dt
#     tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
#
#     tar_vel_err = tar_speed - tar_dir_speed
#
#     # print(shapeof(tar_vel_err))
#     # print(tar_vel_err)
#
#     vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
#     speed_mask = tar_dir_speed <= 0
#     vel_reward[speed_mask] = 0
#
#     movement_dir = torch.nn.functional.normalize(root_vel[..., :2], dim=-1)
#     move_dir_err = tar_dir - movement_dir
#     move_dir_reward = torch.exp(-dir_err_scale * torch.norm(move_dir_err, dim=-1))
#
#     # similarity_err = similarity - 20
#     # add one more reward for text_guided training
#
#     sim_mask = similarity <= 22
#     similarity[sim_mask] = 0
#     similarity_bar = torch.mean(similarity)
#     clip_reward = torch.exp(clip_err_scale * similarity_bar)
#
#     # reward = vel_reward_w * vel_reward + clip_reward_w * clip_reward
#     reward = vel_reward_w * vel_reward + face_reward_w * move_dir_reward + clip_reward_w * clip_reward
#
#     return reward


@torch.jit.script
def compute_location_heading_observations(root_states, tar_pos, tar_height):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos_new = torch.zeros_like(tar_pos)
    tar_vec = tar_pos_new - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_vec, dim=-1)
    heading_theta = torch.atan2(tar_dir[..., 1], tar_dir[..., 0])
    tar_dir = torch.stack([torch.cos(heading_theta), torch.sin(heading_theta)], dim=-1)

    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_tar_dir = quat_rotate(heading_rot, tar_dir3d)
    local_tar_dir = local_tar_dir[..., 0:2]

    obs = torch.cat([local_tar_dir, tar_height.view(-1, 1)], dim=-1)
    return obs


@torch.jit.script
def compute_location_observations(root_states, tar_pos):
    # type: (Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos3d = torch.cat([tar_pos, torch.zeros_like(tar_pos[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_tar_pos = quat_rotate(heading_rot, tar_pos3d - root_pos)
    local_tar_pos = local_tar_pos[..., 0:2]

    obs = local_tar_pos
    return obs


# @torch.jit.script
def compute_location_reward(root_pos, prev_root_pos, root_rot, tar_pos, tar_speed, dt, similarity):
    # type: # (Tensor, Tensor, Tensor, Tensor, float, float, Tensor) -> Tensor
    dist_threshold = 0.5

    pos_err_scale = 0.5
    # vel_err_scale = 4.0
    # clip_err_scale = 2.0

    vel_err_scale = 0.25
    # dir_err_scale = 2.0
    clip_err_scale = 2.0

    pos_reward_w = 0.05
    vel_reward_w = 0.1
    face_reward_w = 0.05
    clip_reward_w = 0.8

    tar_pos_new = torch.zeros_like(tar_pos)
    pos_diff = tar_pos_new - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    tar_dir = tar_pos_new - root_pos[..., 0:2]
    # tar_dir = tar_pos - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    heading_rot = torch.ones_like(root_pos)
    # heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    dist_mask = pos_err < dist_threshold
    facing_reward[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    # sim_mask = similarity <= 22
    # similarity[sim_mask] = 0
    # similarity = similarity - 0.2
    # similarity = similarity - 0.22
    similarity_bar = torch.mean(similarity)
    # expand similarity_bar to match the shape of vel_reward
    similarity_bar = similarity_bar.expand_as(vel_reward)
    clip_reward = similarity_bar
    # clip_reward = torch.exp(-clip_err_scale * similarity_bar)

    # reward = vel_reward_w * vel_reward + clip_reward_w * clip_reward
    # reward = vel_reward_w * vel_reward + face_reward_w * move_dir_reward + clip_reward_w * clip_reward
    reward_pos = pos_reward_w * pos_reward # [1024, 1]
    reward_vel = vel_reward_w * vel_reward
    reward_face = face_reward_w * facing_reward
    reward_clip = clip_reward_w * clip_reward

    reward = reward_pos + reward_vel + reward_face + reward_clip

    return reward, reward_pos, reward_vel, reward_face, reward_clip