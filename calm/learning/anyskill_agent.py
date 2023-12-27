import copy
from gym import spaces
import numpy as np
import os
import yaml
import wandb

from rl_games.common import a2c_common

import torch
from torch.nn.functional import cosine_similarity

import learning.common_agent as common_agent
import learning.calm_agent as calm_agent
import learning.calm_models as calm_models
import learning.calm_network_builder as calm_network_builder
from utils import anyskill


class AnyskillAgent(common_agent.CommonAgent):
    def __init__(self, base_name, config):
        with open(os.path.join(os.getcwd(), config['llc_config']), 'r') as f:
            llc_config = yaml.load(f, Loader=yaml.SafeLoader)
            llc_config_params = llc_config['params']
            self._latent_dim = llc_config_params['config']['latent_dim']
        
        super().__init__(base_name, config)

        self._task_size = self.vec_env.env.task.get_task_obs_size()
        
        self._llc_steps = config['llc_steps']
        llc_checkpoint = config['llc_checkpoint']
        assert(llc_checkpoint != "")
        self._build_llc(llc_config_params, llc_checkpoint)

        self._latent_steps_min = 30
        self._latent_steps_max = 90

        self.anyskill = anyskill.anytest()
        self.mlip_encoder = anyskill.FeatureExtractor()
        self.text_file = config['text_file']
        self.RENDER = config['render']

        return

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs = self.obs['obs']
        self._llc_actions = torch.zeros([self._llc_steps, 1024, 28], device=self.device, dtype=torch.float32)
        rewards = 0.0
        disc_rewards = 0.0
        done_count = 0.0
        terminate_count = 0.0
        for t in range(self._llc_steps): #low-level controller sample 5
            llc_actions = self._compute_llc_action(obs, actions)

            obs, aux_rewards, curr_dones, infos = self.vec_env.step(llc_actions)
            obs[..., self.obs_shape[0] - self._task_size:] = self._text_latents
            self.obs['obs'] = obs

            if self.RENDER:
                images = self.vec_env.env.task.render_img()
                image_features = self.mlip_encoder.encode_images(images)
            else:
                state_embeds = infos['state_embeds'][:, :15, :3]
                image_features = self.anyskill.get_motion_embedding(state_embeds)

            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            anyskill_rewards, similarity = self.vec_env.env.task.compute_anyskill_reward(image_features_norm, self._text_latents,
                                                                             self._latent_text_idx)
            curr_rewards = anyskill_rewards + aux_rewards
            self._llc_actions[t] = llc_actions
            rewards += curr_rewards
            done_count += curr_dones
            terminate_count += infos['terminate']
            
            amp_obs = infos['amp_obs']
            curr_disc_reward = self._calc_disc_reward(amp_obs)
            disc_rewards += curr_disc_reward

        rewards /= self._llc_steps
        disc_rewards /= self._llc_steps

        dones = torch.zeros_like(done_count)
        dones[done_count > 0] = 1.0
        terminate = torch.zeros_like(terminate_count)
        terminate[terminate_count > 0] = 1.0
        infos['terminate'] = terminate
        infos['disc_rewards'] = disc_rewards

        wandb.log({"similarity": similarity.mean().item()})
        wandb.log({"reward/anyskill_reward": anyskill_rewards.mean().item()})
        wandb.log({"reward/aux_reward": aux_rewards.mean().item()})


        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos, self._llc_actions
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos, self._llc_actions

    def cast_obs(self, obs):
        obs = super().cast_obs(obs)
        self._llc_agent.is_tensor_obses = self.is_tensor_obses
        return obs

    def preprocess_actions(self, actions):
        clamped_actions = torch.clamp(actions, -1.0, 1.0)
        if not self.is_tensor_obses:
            clamped_actions = clamped_actions.cpu().numpy()
        return clamped_actions

    def _reset_latents(self, env_ids):
        n = len(env_ids)
        z, z_text_idx = self._sample_latents(n)
        self._text_latents[env_ids] = z
        self._latent_text_idx[env_ids] = z_text_idx

        if (self.vec_env.env.task.viewer):
            self._change_char_color(env_ids)

        return

    def _sample_latents(self, n):
        z, z_text_idx = self.model.a2c_network.sample_text_embeddings(n, self.text_features, self.text_weights)
        return z, z_text_idx

    def _update_latents(self):
        new_latent_envs = self._latent_reset_steps <= self.vec_env.env.task.progress_buf

        need_update = torch.any(new_latent_envs)
        if (need_update):
            new_latent_env_ids = new_latent_envs.nonzero(as_tuple=False).flatten()
            self._reset_latents(new_latent_env_ids)
            self._latent_reset_steps[new_latent_env_ids] += torch.randint_like(self._latent_reset_steps[new_latent_env_ids],
                                                                               low=self._latent_steps_min,
                                                                               high=self._latent_steps_max)
            if (self.vec_env.env.task.viewer):
                self._change_char_color(new_latent_env_ids)

        return

    def play_steps(self):
        self.set_eval()
        '''
            s, a<-z, r<-(s:img, z )
        '''
        epinfos = []
        done_indices = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self._update_latents()
            self.obs = self.env_reset(done_indices)

            # update to real obs contains body states and text_latents
            obs = self.obs['obs']
            obs[..., self.obs_shape[0] - self._task_size:] = self._text_latents
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos, _llc_actions = self.env_step(res_dict['actions'])

            shaped_rewards = self.rewards_shaper(rewards)
            # self.experience_buffer.update_data('text_latents', n, self._text_latents)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            self.experience_buffer.update_data('disc_rewards', n, infos['disc_rewards'])

            style_rewards = self._calc_style_reward(res_dict['actions'])
            self.experience_buffer.update_data('style_rewards', n, style_rewards)

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_disc_rewards = self.experience_buffer.tensor_dict['disc_rewards']
        mb_style_rewards = self.experience_buffer.tensor_dict['style_rewards']
        # mb_text_latents = self.experience_buffer.tensor_dict['text_latents']

        mb_rewards = self._combine_rewards(mb_rewards, mb_disc_rewards, mb_style_rewards)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        return batch_dict
    
    def _load_config_params(self, config):
        super()._load_config_params(config)
        
        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']
        self._style_reward_w = config['style_reward_w']
        return

    def _get_mean_rewards(self):
        rewards = super()._get_mean_rewards()
        rewards *= self._llc_steps
        return rewards

    def _setup_action_space(self):
        super()._setup_action_space()
        self.actions_num = self._latent_dim
        return

    def init_tensors(self):
        super().init_tensors()

        del self.experience_buffer.tensor_dict['actions']
        del self.experience_buffer.tensor_dict['mus']
        del self.experience_buffer.tensor_dict['sigmas']

        batch_shape = self.experience_buffer.obs_base_shape
        # self.experience_buffer.tensor_dict['text_latents'] = torch.zeros(batch_shape + (512,),
        #                                                             dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['actions'] = torch.zeros(batch_shape + (self._latent_dim,),
                                                                    dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['mus'] = torch.zeros(batch_shape + (self._latent_dim,),
                                                                dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['sigmas'] = torch.zeros(batch_shape + (self._latent_dim,),
                                                                   dtype=torch.float32, device=self.ppo_device)
        
        self.experience_buffer.tensor_dict['disc_rewards'] = torch.zeros_like(self.experience_buffer.tensor_dict['rewards'])
        self.experience_buffer.tensor_dict['style_rewards'] = torch.zeros_like(self.experience_buffer.tensor_dict['rewards'])
        self.tensor_list += ['disc_rewards', 'style_rewards']

        batch_shape = self.experience_buffer.obs_base_shape
        self._latent_reset_steps = torch.zeros(batch_shape[-1], dtype=torch.int32, device=self.ppo_device)


        texts, texts_weights = load_texts(self.text_file)
        self.text_features = self.mlip_encoder.encode_texts(texts)
        self.text_weights = torch.tensor(texts_weights, device=self.device)
        self._text_latents = torch.zeros((batch_shape[-1], 512), dtype=torch.float32,
                                         device=self.ppo_device)
        self._latent_text_idx = torch.zeros((batch_shape[-1],), dtype=torch.long, device=self.ppo_device)

        return

    def _get_motion_encoding(self):
        all_encoded_demo_amp_obs = []
        for motion_id in range(self.vec_env.env.task._motion_lib._motion_weights.shape[0]):
            motion_amp_obs = self.vec_env.env.task.fetch_amp_obs_demo_per_id(32, motion_id)[-1].view(32, self.vec_env.env.task._num_amp_obs_enc_steps, self.vec_env.env.task._num_amp_obs_per_step)

            preproc_amp_obs = self._llc_agent._preproc_amp_obs(motion_amp_obs)
            encoded_demo_amp_obs = self._llc_agent.model.a2c_network.eval_enc(preproc_amp_obs)
            all_encoded_demo_amp_obs.append(encoded_demo_amp_obs)
        all_encoded_demo_amp_obs = torch.stack(all_encoded_demo_amp_obs, dim=0)

        return all_encoded_demo_amp_obs

    def _build_llc(self, config_params, checkpoint_file):
        network_params = config_params['network']

        network_builder = calm_network_builder.CALMBuilder()

        network_builder.load(network_params)

        network = calm_models.ModelCALMContinuous(network_builder)

        llc_agent_config = self._build_llc_agent_config(config_params, network)

        self._llc_agent = calm_agent.CALMAgent('llc', llc_agent_config)

        self._llc_agent.restore(checkpoint_file)
        print("Loaded LLC checkpoint from {:s}".format(checkpoint_file))
        self._llc_agent.set_eval()

        # enc_amp_obs = self._llc_agent._fetch_amp_obs_demo(128)
        # if len(enc_amp_obs) == 2:
        #     enc_amp_obs = enc_amp_obs[0]
        #
        # preproc_enc_amp_obs = self._llc_agent._preproc_amp_obs(enc_amp_obs)
        # self.encoded_motion = self._llc_agent.model.a2c_network.eval_enc(amp_obs=preproc_enc_amp_obs).unsqueeze(0)
        self.encoded_motion = self._get_motion_encoding()

        return

    def _build_llc_agent_config(self, config_params, network):
        llc_env_info = copy.deepcopy(self.env_info)
        obs_space = llc_env_info['observation_space']
        obs_size = obs_space.shape[0]
        obs_size -= self._task_size
        llc_env_info['observation_space'] = spaces.Box(obs_space.low[:obs_size], obs_space.high[:obs_size])

        config = config_params['config']
        config['network'] = network
        config['num_actors'] = self.num_actors
        config['features'] = {'observer': self.algo_observer}
        config['env_info'] = llc_env_info
        config['minibatch_size'] = 1
        config['amp_batch_size'] = 32
        config['amp_minibatch_size'] = 1
        config['enable_eps_greedy'] = False
        config['vec_env'] = self.vec_env

        return config

    def _compute_llc_action(self, obs, actions):
        llc_obs = self._extract_llc_obs(obs)
        processed_obs = self._llc_agent._preproc_obs(llc_obs)

        z = torch.nn.functional.normalize(actions, dim=-1)
        mu, _ = self._llc_agent.model.a2c_network.eval_actor(processed_obs, z)
        llc_action = mu
        llc_action = self._llc_agent.preprocess_actions(llc_action)

        return llc_action

    def _extract_llc_obs(self, obs):
        obs_size = obs.shape[-1]
        llc_obs = obs[..., :obs_size - self._task_size]
        return llc_obs

    def _calc_disc_reward(self, amp_obs):
        disc_reward = self._llc_agent._calc_disc_rewards(amp_obs)
        return disc_reward

    def _calc_style_reward(self, action):
        z = torch.nn.functional.normalize(action, dim=-1)
        style_reward = torch.max((cosine_similarity(z.unsqueeze(1), self.encoded_motion, dim=-1) + 1) / 2, dim=1)[0]
        return style_reward.unsqueeze(-1)

    def _combine_rewards(self, task_rewards, disc_rewards, style_rewards):
        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._disc_reward_w * disc_rewards + self._style_reward_w * style_rewards

        return combined_rewards

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['disc_rewards'] = batch_dict['disc_rewards']
        train_info['style_rewards'] = batch_dict['style_rewards']
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)

        disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
        self.writer.add_scalar('info/disc_reward_mean', disc_reward_mean.item(), frame)
        self.writer.add_scalar('info/disc_reward_std', disc_reward_std.item(), frame)

        style_reward_std, style_reward_mean = torch.std_mean(train_info['style_rewards'])
        self.writer.add_scalar('info/style_reward_mean', style_reward_mean.item(), frame)
        self.writer.add_scalar('info/style_reward_std', style_reward_std.item(), frame)
        return


    def _change_char_color(self, env_ids):
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        self.vec_env.env.task.set_char_color(rand_col, env_ids)
        return

def load_texts(text_file):
    ext = os.path.splitext(text_file)[1]
    assert ext == ".yaml"
    weights = []
    texts = []
    with open(os.path.join(os.getcwd(), text_file), 'r') as f:
        text_config = yaml.load(f, Loader=yaml.SafeLoader)

    text_list = text_config['texts']
    for text_entry in text_list:
        curr_text = text_entry['text']
        curr_weight = text_entry['weight']
        assert(curr_weight >= 0)
        weights.append(curr_weight)
        texts.append(curr_text)
    return texts, weights