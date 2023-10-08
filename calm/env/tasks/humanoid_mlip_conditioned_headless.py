import torch
import open_clip
from isaacgym.torch_utils import *

from utils import torch_utils
from utils import anyskill
from Anyskill import *
from env.tasks.humanoid_heading import HumanoidHeading

TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2
RANGE = 1024

class HumanoidHeadingConditioned(HumanoidHeading):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._similarity = torch.zeros([RANGE], device=self.device, dtype=torch.float32)
        self._corresponding_speeds = torch.tensor([6, 3.5, 3.5], device=self.device, dtype=torch.float32)
        self._tar_locomotion_index = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)
        self._memory = {}
        self.clip_features = []
        self.motionclip_features = []

        self.mlip_model, _, self.mlip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.anyskill = anyskill.anytest()
        self.frame = 0
        return

    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            obs_size = 2 + 1
        return obs_size

    def _calc_similarity(self, state_embeds):
        # print("==================== compute the similarity of CLIP ====================")
        # image = self.render(sync_frame_time=False)
        raw_caption = self.cfg['env']['caption']
        caption = raw_caption.replace("_", " ")


        image_features = self.anyskill.get_motion_embedding(state_embeds)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        image_dim = image_features.shape[0]

        text = self.tokenizer([caption])
        text_features = self.mlip_model.encode_text(text).cuda()
        # text_features = text_features.repeat(image_dim, 1).cuda()
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = 100.0 * torch.matmul(image_features_norm, text_features_norm.permute(1,0))  # this is the image-text similarity score
        # similarity = 100.0 * image_features_norm @ text_features_norm.T  # this is the image-text similarity score

        # self.clip_features.append(image_features.data.cpu().numpy())
        # self.motionclip_features.append(state_embeds.data.cpu().numpy())
        #
        # np.save("./output/motion_feature_heading.npy", self.motionclip_features)
        # np.save("./output/image_feature_heading.npy", self.clip_features)
        #
        # self._memory[self.frame] = similarity
        # np.save("./output/sim_heading.npy", self._memory)
        return similarity.squeeze()


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
                                                 self._tar_speed, self.dt, self._similarity)
        return

    def _reset_task(self, env_ids):
        n = len(env_ids)

        if n > 0:
            rand_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi

            change_steps = torch.randint(low=self._heading_change_steps_min, high=self._heading_change_steps_max,
                                         size=(n,), device=self.device, dtype=torch.int64)

            tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)

            tar_locomotion_index = torch.randint(low=0, high=3, size=(n,), device=self.device, dtype=torch.int64)

            # rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
            # rigid_state_tensor = gymtorch.wrap_tensor(rigid_body_state)
            # bodies_per_env = rigid_state_tensor.shape[0] // self.num_envs
            # self.rigid_state_tensor = rigid_state_tensor.view(self.num_envs, bodies_per_env, 13)
            state_embeds = self._rigid_state_tensor[env_ids, :, :]

            self._tar_dir[env_ids] = tar_dir
            self._tar_facing_dir[env_ids] = tar_dir
            self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
            self._tar_locomotion_index[env_ids] = tar_locomotion_index
            self._tar_speed[env_ids] = self._corresponding_speeds[tar_locomotion_index]
            self._similarity[env_ids] = self._calc_similarity(state_embeds)
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
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor) -> Tensor
    vel_err_scale = 0.25
    dir_err_scale = 2.0
    clip_err_scale = 0.15

    vel_reward_w = 0.01
    face_reward_w = 0.01
    clip_reward_w = 0.98

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

    sim_mask = similarity <= 22
    similarity[sim_mask] = 0
    # similarity_bar = torch.mean(similarity)
    clip_reward = torch.exp(clip_err_scale * similarity)

    reward = vel_reward_w * vel_reward + face_reward_w * move_dir_reward + clip_reward_w * clip_reward

    return reward
