# import gym
import torch
import torch.nn as nn
import torch.optim as optim
from parameters import Gamma, EPISODES
import ddos_env

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared layers
        self.common = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic head
        self.critic = nn.Linear(128, 1)

    def forward(self, x):

        x = self.common(x)

        policy = self.actor(x)
        # policy = torch.nan_to_num(policy)
        policy  = policy

        value = self.critic(x)
        # value = torch.nan_to_num(value)
        value = value

        return policy, value

def select_action(policy):
    action_probs = torch.distributions.Categorical(policy)
    action = action_probs.sample()
    return action.item(), action_probs.log_prob(action)

def train(env, model, optimizer, gamma=Gamma, num_episodes=EPISODES):
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        log_probs = []
        values = []
        rewards = []
        done = False
        while not done:
            policy, value = model(state)

            action, log_prob = select_action(policy)

            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        values = torch.cat(values)

        log_probs = torch.stack(log_probs)

        advantage = returns - values.squeeze()

        actor_loss = -(log_probs * advantage).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        