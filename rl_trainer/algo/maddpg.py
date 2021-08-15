import os
from rl_trainer.misc import onehot_from_logits
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
from algo.ddpg_agent import DDPGAgent
from misc import onehot_from_logits, gumbel_softmax

class MADDPG:

    def __init__(self, obs_dim, act_dim, num_agent, args):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.args = args
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.model_episode = args.model_episode
        self.eps = args.epsilon
        self.decay_speed = args.epsilon_speed
        self.output_activation = args.output_activation
        self.niter = 0
        self.a_clip = args.a_clip
        self.c_clip = args.c_clip
        self.use_gumbel = args.use_gumbel
        
        self.num_in_critic = 0
        self.num_in_actor = obs_dim
        for i in range(num_agent):
            self.num_in_critic += obs_dim
            self.num_in_critic += act_dim
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)
        
        self.agents = [DDPGAgent(obs_dim,
                                 act_dim,
                                 self.num_in_critic,
                                 self.num_agent,
                                 args,
                                 index) for index in range(num_agent)]

    @property
    def policies(self):
        return [a.actor for a in self.agents]

    @property
    def target_policies(self):
        return [a.actor_target for a in self.agents]
    
    def update(self, agent_i):
        
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batches()
        curr_agent = self.agents[agent_i]
        curr_agent.critic_optimizer.zero_grad()

        state_batch = state_batch.reshape(self.batch_size, self.num_agent, -1)
        action_batch = action_batch.reshape(self.batch_size, self.num_agent, -1)
        reward_batch = reward_batch.reshape(self.batch_size, self.num_agent, 1)
        next_state_batch = next_state_batch.reshape(self.batch_size, self.num_agent, -1)
        done_batch = done_batch.reshape(self.batch_size, self.num_agent, 1)

        for i in range(self.num_agent):
            if i == 0:
                obs = state_batch[:, 0, :][None, :, :]
                acs = action_batch[:, 0, :][None, :, :]
                rews = reward_batch[:, 0, :][None, :, :]
                next_obs = next_state_batch[:, 0, :][None, :, :]
                dones = done_batch[:, 0, :][None, :, :]
            else:
                obs = np.vstack((obs, state_batch[:, i, :][None, :, :]))
                acs = np.vstack((acs, action_batch[:, i, :][None, :, :]))
                rews = np.vstack((rews, reward_batch[:, i, :][None, :, :]))
                next_obs = np.vstack((next_obs, next_state_batch[:, i, :][None, :, :]))
                dones = np.vstack((dones, done_batch[:, i, :][None, :, :]))
        
        obs = torch.Tensor(obs).to(self.device)
        acs = torch.Tensor(acs).to(self.device)
        rews = torch.Tensor(rews).to(self.device)
        next_obs = torch.Tensor(next_obs).to(self.device)
        dones = torch.Tensor(dones).to(self.device)

        all_target_acts = [onehot_from_logits(pi(nobs)) for pi, nobs in zip(self.target_policies, next_obs)]
        target_vf_in = torch.cat((*next_obs, *all_target_acts), dim=1)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.critic_target(target_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))
        
        vf_in = torch.cat((*obs, *acs), dim=1)
        actual_value = curr_agent.critic(vf_in)
        vf_loss = torch.nn.MSELoss()(actual_value, target_value.detach())
        vf_loss.backward()
        clip_grad_norm_(curr_agent.critic.parameters(), self.c_clip)
        curr_agent.critic_optimizer.step()
        
        curr_agent.actor_optimizer.zero_grad()
        
        curr_pol_out = curr_agent.actor(obs[agent_i])
        curr_pol_vf_in = torch.nn.functional.gumbel_softmax(curr_pol_out, hard=True)
        # curr_pol_vf_in = gumbel_softmax(curr_pol_out,  device=self.device, hard=True)
        all_pol_acs = []
        for i, pi, ob in zip(range(self.num_agent), self.policies, obs):
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in)
            else:
                all_pol_acs.append(onehot_from_logits(pi(ob)))
        vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
        clip_grad_norm_(curr_agent.actor.parameters(), self.a_clip)
        curr_agent.actor_optimizer.step()
        curr_agent.a_loss = pol_loss.item()
        curr_agent.c_loss = vf_loss.item()
        return curr_agent.c_loss, curr_agent.a_loss
        
        
    def update_all_targets(self):
        
        for a in self.agents:
            soft_update(a.critic, a.critic_target, self.tau)
            soft_update(a.actor, a.actor_target, self.tau)
        self.niter += 1

    def choose_action(self, obs, evaluation=False):

        actions = np.zeros((3, 4))
        for i in range(self.num_agent):
            obs_agent = np.expand_dims(obs[i], axis=0)
            actions[i] = self.agents[i].choose_action(obs_agent, self.use_gumbel)
        
        return actions


    def load_model(self, run_dir, episode):
        pass
    
    def save_model(self, run_dir, episode, score):
        for agent in self.agents:
            agent.save_model(run_dir, episode, score)
    
    def loss(self):
        flag = True
        for i in range(self.num_agent):
            if self.agents[i].c_loss is None or self.agents[i].a_loss is None:
                flag = False
        
        return flag


    
    

        

        