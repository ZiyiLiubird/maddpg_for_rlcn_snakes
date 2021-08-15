import os
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from replay_buffer import ReplayBuffer
from common import soft_update, hard_update, device
from algo.network import Actor, Critic
from misc import onehot_from_logits, gumbel_softmax
# torch.nn.functional.gumbel_softmax

class DDPGAgent:
    
    def __init__(self, obs_dim, act_dim, num_in_critic, num_agent, args, index):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_in_critic = num_in_critic
        self.num_agent = num_agent
        self.args = args
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.index = index
        self.output_activation = args.output_activation
        self.eps = args.epsilon
        self.decay_speed = args.epsilon_speed
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.model_episode = args.model_episode


        
        self.actor = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        self.critic = Critic(num_in_critic, num_agent, args).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)
        
        self.actor_target = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        self.critic_target = Critic(num_in_critic, num_agent, args).to(self.device)
        hard_update(self.actor, self.actor_target)
        hard_update(self.critic, self.critic_target)
        
        self.c_loss = None
        self.a_loss = None

    def choose_action(self, obs, use_gumbel=False, evaluation=False):

        if use_gumbel:
            obs = torch.Tensor([obs]).to(self.device)
            action = self.actor(obs)
            action = torch.nn.functional.gumbel_softmax(action, hard=True).cpu().detach().numpy()[0]
            # action = gumbel_softmax(action, self.device, hard=True).cpu().detach().numpy()[0]
        else:
            p = np.random.random()
            if p > self.eps or evaluation:
                obs = torch.Tensor([obs]).to(self.device)
                action = self.actor(obs).cpu().detach().numpy()[0]
            else:
                action = self.random_action()

            self.eps *= self.decay_speed
        return action

    def random_action(self):
        if self.output_activation == 'tanh':
            return np.random.uniform(low=-1, high=1, size=(1, self.act_dim))
        return np.random.uniform(low=0, high=1, size=(1, self.act_dim))

    def get_loss(self):
        return self.c_loss, self.a_loss

    def load_model(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        model_actor_path = os.path.join(base_path, "actor_" + str(self.index) + "_" + str(episode) + ".pth")
        model_critic_path = os.path.join(base_path, "critic_" + str(self.index) + "_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            critic = torch.load(model_critic_path, map_location=device)
            self.actor.load_state_dict(actor)
            self.critic.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

    def save_model(self, run_dir, episode, score):
        base_path = os.path.join(run_dir, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(self.index) + "_" + str(episode) + "_" + str(score) + ".pth")
        torch.save(self.actor.state_dict(), model_actor_path)

        model_critic_path = os.path.join(base_path, "critic_" + str(self.index) + "_" + str(episode) + ".pth")
        torch.save(self.critic.state_dict(), model_critic_path)
