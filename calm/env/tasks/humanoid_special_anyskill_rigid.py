import torch
import time
import numpy as np
from isaacgym import gymtorch, gymapi

from env.tasks.humanoid_amp_getup import HumanoidAMPGetup
from isaacgym.torch_utils import *

class HumanoidSpecAnySKillRigid(HumanoidAMPGetup):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        
        self._recovery_episode_prob = cfg["env"]["recoveryEpisodeProb"]
        self._recovery_steps = cfg["env"]["recoverySteps"]
        self._fall_init_prob = cfg["env"]["fallInitProb"]
        self._tar_speed_min = cfg["env"]["tarSpeedMin"]
        self._tar_speed_max = cfg["env"]["tarSpeedMax"]
        self._heading_change_steps_min = cfg["env"]["headingChangeStepsMin"]
        self._heading_change_steps_max = cfg["env"]["headingChangeStepsMax"]
        self._enable_rand_heading = cfg["env"]["enableRandHeading"]
        self._motion_filename = cfg['env']['wandb_counter']
        self._reset_fall_env_ids = []

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._recovery_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self._build_target_tensors()
        self._generate_fall_states()
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._tar_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._tar_dir[..., 0] = 1.0
        self._tar_facing_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._tar_facing_dir[..., 0] = 1.0

        self._tar_speed = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        # self._tar_speed = torch.ones([self.num_envs], device=self.device, dtype=torch.float)
        self._heading_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
        self._similarity = torch.zeros([self.num_envs], device=self.device, dtype=torch.float32)
        self._punish_counter = torch.zeros([self.num_envs], device=self.device, dtype=torch.int)
        # self.torch_rgba_tensor = torch.zeros([self.num_envs, 224, 224, 3], device=self.device, dtype=torch.float32)
        self.eposide = 0

        self.save_inference_motion = cfg["env"]["enableMotionSave"]
        if self.save_inference_motion:
            file_name = "./calm/data/assets/shell.npy"
            file = np.load(file_name, allow_pickle=True)
            self.f = file.item()
            self.data_to_store = []
        return

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = torch.arange(self.num_envs * self.get_num_actors_per_env(), device=self.device,
                                     dtype=torch.int32)
        huamnoid_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(huamnoid_ids_int32), len(huamnoid_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def _reset_target(self, env_ids):
        n = len(env_ids)
        init_dist = 3 * torch.ones([n], dtype=self._target_states.dtype, device=self._target_states.device)
        init_theta = 0 * np.pi * torch.ones([n], dtype=self._target_states.dtype, device=self._target_states.device)
        # self._target_states[env_ids, 0] = init_dist * torch.cos(init_theta) + self._initial_humanoid_root_states[env_ids, 0]
        # self._target_states[env_ids, 1] = init_dist * torch.sin(init_theta) + self._initial_humanoid_root_states[env_ids, 1]
        # soccerball
        # self._target_states[env_ids, 2] = 0.1
        # pillar
        # self._target_states[env_ids, 2] = 0.9

        # chair
        self._target_states[env_ids, 0] = 0.448
        self._target_states[env_ids, 1] = 0.448
        self._target_states[env_ids, 2] = 0.448

        init_rot_theta = 0.5 * np.pi * torch.ones([n], dtype=self._target_states.dtype, device=self._target_states.device)
        # init_rot_theta = np.pi * torch.ones([n], dtype=self._target_states.dtype, device=self._target_states.device)
        axis = torch.tensor([1.0, 0.0, 0.0], dtype=self._target_states.dtype, device=self._target_states.device)
        init_rot = quat_from_angle_axis(init_rot_theta, axis)

        self._target_states[env_ids, 3:7] = init_rot
        self._target_states[env_ids, 7:10] = 0.0
        self._target_states[env_ids, 10:13] = 0.0

        return

    # def _draw_task(self):
    #     cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    #
    #     self.gym.clear_lines(self.viewer)
    #
    #     starts = self._humanoid_root_states[..., 0:3]
    #     ends = self._target_states[..., 0:3]
    #     verts = torch.cat([starts, ends], dim=-1).cpu().numpy()
    #
    #     for i, env_ptr in enumerate(self.envs):
    #         curr_verts = verts[i]
    #         curr_verts = curr_verts.reshape([1, 6])
    #         self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)
    #
    #     return

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._load_target_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_target(env_id, env_ptr)
        return

    def _load_target_asset(self):
        asset_root = "calm/data/assets/mjcf/"
        asset_file = "chair.urdf"
        # asset_file = "soccerball.urdf"
        # asset_file = "pillar.urdf"

        asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True

        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        # asset_options.max_angular_velocity = 100.0
        # chair
        asset_options.density = 50.0
        # asset_options.density = 50.0

        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # # # for chair

        self._target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        return

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]

        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]

        return

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0

        # for chair
        # default_pose.r = gymapi.Quat(0,0,0,1)
        default_pose.r = gymapi.Quat(0.5, 0.5, 0.5, 0.5)

        target_handle = self.gym.create_actor(env_ptr, self._target_asset, default_pose, "target", col_group,
                                              col_filter, segmentation_id)
        self._target_handles.append(target_handle)

        return

    def render_img(self, sync_frame_time=False):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[:, 0:3] if self.RENDER else self._humanoid_root_states[:, 0:3].cpu().numpy()
        # char_root_rot = self._humanoid_root_states[:, 3:7].cpu().numpy()
        self._cam_prev_char_pos[:] = char_root_pos

        start = time.time()
        for env_id in range(self.num_envs):
            cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
            cam_pos = torch.tensor([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z], device=self.device)
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
        # print("time of render {} frames' image: {}".format(env_id, (time.time() - start)))
        self.gym.end_access_image_tensors(self.sim)

        return self.torch_rgba_tensor.permute(0, 3, 1, 2)

    def render_headless(self, sync_frame_time=False):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[:, 0:3]
        # char_root_rot = self._humanoid_root_states[:, 3:7].cpu().numpy()
        self._cam_prev_char_pos[:] = char_root_pos

        start = time.time()
        for env_id in range(self.num_envs):
            cam_pos = torch.tensor([self.cam_pos.x, self.cam_pos.y, self.cam_pos.z], device=self.device)
            cam_delta = cam_pos - self._cam_prev_char_pos[env_id]

            target = gymapi.Vec3(char_root_pos[env_id, 0], char_root_pos[env_id, 1], 1.0)
            pos = gymapi.Vec3(char_root_pos[env_id, 0] + cam_delta[0],
                              char_root_pos[env_id, 1] + cam_delta[1],
                              cam_pos[2])

            pos_nearer = gymapi.Vec3(pos.x + 1.2, pos.y + 1.2, pos.z)
            self.gym.set_camera_location(self.camera_handles[env_id], self.envs[env_id], pos_nearer, target)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        # todo: THE CAMERA VIEW CHANGE STEP BY STEP
        for env_id in range(self.num_envs):
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_id],
                                                                      self.camera_handles[env_id],
                                                                      gymapi.IMAGE_COLOR)
            self.torch_rgba_tensor[env_id] = gymtorch.wrap_tensor(camera_rgba_tensor)[:, :,
                                             :3].float()  # [224,224,3] -> IM -> [env,224,224,3]
        # print("time of render {} frames' image: {}".format(env_id, (time.time() - start)))
        self.gym.end_access_image_tensors(self.sim)

        return self.torch_rgba_tensor.permute(0, 3, 1, 2)


    def get_task_obs_size(self):
        obs_size = 0  # dim for obj_state
        return obs_size

    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obs = humanoid_obs
        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return

    def get_obs_size(self):
        obs_size = super().get_obs_size()
        return obs_size + 0 # without text observation

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        self._update_recovery_count()
        return

    def __del__(self):
        if self.save_inference_motion:
            self.data_to_store = np.array(self.data_to_store)
            self.f['is_local'] = False
            self.f['rotation']['arr'] = self.data_to_store[:, :, 3:7]
            self.f['root_translation']['arr'] = self.data_to_store[:, 0, 0:3]
            self.f['global_velocity']['arr'] = self.data_to_store[:, :, 7:10]
            self.f['global_angular_velocity']['arr'] = self.data_to_store[:, :, 10:13]
            self.f['position'] = self.data_to_store[:, :, 0:3]
            np.save("./output/motions/" + self._motion_filename + ".npy", self.f)

    def post_physics_step(self):
        super().post_physics_step()
        self.extras["state_embeds"] = self._rigid_state_tensor[:, :, :]

        if self.save_inference_motion:
            data = self._rigid_body_state.view(self.num_envs, self._rigid_body_state.shape[0] // self.num_envs, 13).cpu().numpy()
            self.data_to_store.append(data[0][0:16])

    def _generate_fall_states(self):
        if not hasattr(self, "_target_states"):
            self._build_target_tensors()
        max_steps = 150
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._reset_target(env_ids)
        root_states = self._initial_humanoid_root_states[env_ids].clone()
        root_states[..., 3:7] = torch.randn_like(root_states[..., 3:7])
        root_states[..., 3:7] = torch.nn.functional.normalize(root_states[..., 3:7], dim=-1)
        self._humanoid_root_states[env_ids] = root_states
        
        huamnoids_env_ids_int32 = self._humanoid_actor_ids[env_ids]
        env_ids_int32 = torch.arange(self.num_envs * 2, device=self.device, dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(huamnoids_env_ids_int32), len(huamnoids_env_ids_int32))

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
        print("comes here")
        num_envs = env_ids.shape[0]
        recovery_probs = to_torch(np.array([self._recovery_episode_prob] * num_envs), device=self.device)
        recovery_mask = torch.bernoulli(recovery_probs) == 1.0
        terminated_mask = (self._terminate_buf[env_ids] == 1)
        mlip_mask = self._punish_counter[env_ids] > 8 # true for terminate
        object_mask = self._target_states[env_ids, 2] < 0.8 # true for terminate
        # print("Due to similarity, we need terminate {} envs.".format((mlip_mask==True).sum()))

        # print(object_mask.sum().item())
        filter_recovery_mask = torch.logical_and(recovery_mask, torch.logical_not(mlip_mask))
        recovery_mask = torch.logical_and(filter_recovery_mask, terminated_mask)

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

        object_ids = env_ids[object_mask]
        if len(object_ids) > 0:
            self._reset_target(object_ids)

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
        self.eposide += 1
        is_recovery = self._recovery_counter > 0
        self.reset_buf[is_recovery] = 0
        self._terminate_buf[is_recovery] = 0
        self._punish_counter[is_recovery] = 0
        return

    def compute_anyskill_reward(self, img_features_norm, text_features_norm, corresponding_id):
        similarity = torch.einsum('ij,ij->i', img_features_norm, text_features_norm[corresponding_id])

        # similarity_m = (100.0 * torch.matmul(img_features_norm, text_features_norm.permute(1, 0))).squeeze()
        # rows = torch.arange(corresponding_id.size(0))
        # similarity_raw = similarity_m[rows, corresponding_id]
        if self.RENDER:
            clip_reward_w = 800
        else:
            clip_reward_w = 3600
            # clip_reward_w = 30000
        delta = similarity - self._similarity
        punish_mask = delta < 0
        self._punish_counter[punish_mask] += 1

        # # delta
        # # print("clip_reward_w: {}".format(clip_reward_w))
        # clip_reward = clip_reward_w * delta

        # # value
        clip_reward = 0.8 * similarity

        # # global
        # alpha = 0.5
        # similarity.unsqueeze(1)
        # delta.unsqueeze(1)
        # clip_reward = torch.zeros_like(similarity)
        # for i in range(1024):
        #     proj = (torch.dot(similarity[i], delta[i]) / torch.norm(delta[i], p=2)**2) * delta
        #     term1 = alpha * proj * similarity[i]
        #     term2 = (1-alpha) * delta[i]
        #     clip_reward[i] = 1 - 0.5 * torch.norm(term1 + term2, p=2)**2

        # # CLIP socre
        # mask = similarity < 0
        # print((mask==True).sum())
        # similarity[mask] = 0
        # clip_score = 2.5 * similarity

        # # RCLIP score
        # # clip_score - E(clip_score)
        # # self._exp_sim()

        self._similarity = similarity
        return clip_reward, delta, similarity

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
        # print("tar_dir: ", tar_dir)
        self._tar_facing_dir[env_ids] = face_tar_dir
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        return


def compute_aux_reward(root_pos, prev_root_pos, tar_dir, tar_speed, dt):
    vel_err_scale = 0.25
    dir_err_scale = 2.0

    vel_reward_w = 0.1
    face_reward_w = 0.1

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

    # heading_rot = torch.ones_like(root_pos)
    # facing_dir = torch.zeros_like(root_pos)
    # facing_dir[..., 0] = 1.0
    # facing_dir = quat_rotate(heading_rot, facing_dir)
    # facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    # facing_reward = torch.clamp_min(facing_err, 0.0)
    #
    # dist_mask = pos_err < dist_threshold
    # facing_reward[dist_mask] = 1.0

    reward = vel_reward_w * vel_reward
    # reward = face_reward_w * move_dir_reward
    # reward = vel_reward_w * vel_reward + face_reward_w * move_dir_reward

    return reward

