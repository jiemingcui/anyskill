from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn


class SpecAnyskillBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

            if self.is_continuous:
                if not self.space_config['learn_sigma']:
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)

            return
        
        def forward(self, obs_dict):
            mu, sigma, value, states = super().forward(obs_dict)
            norm_mu = torch.tanh(mu)
            return norm_mu, sigma, value, states

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value

        def sample_text_embeddings(self, n, text_embeddings, weights):
            device = next(self.critic_mlp.parameters()).device
            if not hasattr(self, "all_norm_latents"):
                self.all_text_embeddings = text_embeddings

            if self.all_text_embeddings.device != device:
                self.all_text_embeddings = self.all_norm_latents.to(device)

            if not hasattr(self, "text_weights"):
                self.text_weights = weights

            z_text_idx = torch.multinomial(self.text_weights, num_samples=n, replacement=True)
            z = self.all_text_embeddings[z_text_idx, :]
            # z = torch.normal(mean=z, std=0.0)
            z = torch.nn.functional.normalize(z, dim=-1)
            return z, z_text_idx

    def build(self, name, **kwargs):
        net = SpecAnyskillBuilder.Network(self.params, **kwargs)
        return net

