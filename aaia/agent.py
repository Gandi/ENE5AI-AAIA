import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
import numpy as np

# Policy / Value NN params
STATE_SIZE = 8

HIDDEN_SIZE = 512
DROPOUT_PROB = 0.2

# PPOAgent params
LR = 0.00001
GAMMA = 0.9
CLIP_RATIO = 0.1
VF_COEF = 0.8
ENTROPY_COEF = 0.1
OPTIMIZATION_EPOCHS = 1
DECAYING_EXPLORATION_NOISE_COEF = 0.5


class Policy(nn.Module):

    def __init__(self, env):
        super(Policy, self).__init__()
        action_size_allocation_cpu = env.action_space['CPU']['Allocation'].n
        action_size_allocation_ram = env.action_space['RAM']['Allocation'].n

        self.fcin = nn.Linear(STATE_SIZE, HIDDEN_SIZE)
        self.fc1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        self.fcout_reallocating_cpu = nn.Linear(HIDDEN_SIZE,
                                                1)
        self.fcout_reallocating_ram = nn.Linear(HIDDEN_SIZE,
                                                1)

        self.fcout_allocation_cpu = nn.Linear(HIDDEN_SIZE,
                                              action_size_allocation_cpu)
        self.fcout_allocation_ram = nn.Linear(HIDDEN_SIZE,
                                              action_size_allocation_ram)

        self.dropout = nn.Dropout(p=DROPOUT_PROB)

    def forward(self, x):
        x = F.relu(self.fcin(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        logits_reallocating_cpu = self.fcout_reallocating_cpu(x)
        logits_reallocating_ram = self.fcout_reallocating_ram(x)
        probs_reallocating_cpu = F.sigmoid(logits_reallocating_cpu)
        probs_reallocating_ram = F.sigmoid(logits_reallocating_ram)

        logits_allocation_cpu = self.fcout_allocation_cpu(x)
        logits_allocation_ram = self.fcout_allocation_ram(x)
        probs_allocation_cpu = F.softmax(logits_allocation_cpu, dim=-1)
        probs_allocation_ram = F.softmax(logits_allocation_ram, dim=-1)

        return probs_reallocating_cpu,\
            probs_reallocating_ram,\
            probs_allocation_cpu,\
            probs_allocation_ram


class Value(nn.Module):

    def __init__(self):
        super(Value, self).__init__()
        self.fcin = nn.Linear(STATE_SIZE, HIDDEN_SIZE)
        self.fc1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fcout = nn.Linear(HIDDEN_SIZE, 1)

        self.dropout = nn.Dropout(p=DROPOUT_PROB)

    def forward(self, x):
        x = F.relu(self.fcin(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        state_value = self.fcout(x)
        return state_value


class PPOAgent:

    def __init__(self,
                 env,
                 lr=LR,
                 gamma=GAMMA,
                 clip_ratio=CLIP_RATIO,
                 vf_coef=VF_COEF,
                 entropy_coef=ENTROPY_COEF,
                 optimization_epochs=OPTIMIZATION_EPOCHS):
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.optimization_epochs = optimization_epochs
        self.policy = Policy(env)
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                           lr=self.lr)
        self.value = Value()
        self.value_optimizer = optim.Adam(self.value.parameters(),
                                          lr=self.lr)

    def select_action(self,
                      state,
                      exploration_noise=0,
                      coef_margin_cpu=0,
                      coef_margin_ram=0):
        probs_reallocating_cpu,\
            probs_reallocating_ram,\
            probs_allocation_cpu,\
            probs_allocation_ram = self.policy(state)

        requested_cpu = state[0]
        requested_ram = state[1]
        predicted_cpu = state[6]
        predicted_ram = state[7]

        reallocating_noise_cpu = np.random.normal(0, exploration_noise)
        reallocating_noise_ram = np.random.normal(0, exploration_noise)
        allocation_noise_cpu = np.random.normal(0, exploration_noise)
        allocation_noise_ram = np.random.normal(0, exploration_noise)

        dist_reallocating_cpu = Bernoulli(probs_reallocating_cpu)
        dist_reallocating_ram = Bernoulli(probs_reallocating_ram)

        reallocating_cpu = dist_reallocating_cpu.sample()
        reallocating_cpu = torch.clamp(
            reallocating_cpu * (1 + reallocating_noise_cpu),
            min=0,
            max=1).round()
        reallocating_ram = dist_reallocating_ram.sample()
        reallocating_ram = torch.clamp(
            reallocating_ram * (1 + reallocating_noise_ram),
            min=0,
            max=1).round()

        dist_allocation_cpu = Categorical(probs_allocation_cpu)
        dist_allocation_ram = Categorical(probs_allocation_ram)

        if reallocating_cpu:
            allocation_cpu = dist_allocation_cpu.sample()
            allocation_cpu = torch.clamp(
                allocation_cpu * (1 + allocation_noise_cpu),
                min=min(predicted_cpu + coef_margin_cpu * requested_cpu,
                        requested_cpu),
                max=min(predicted_cpu + 2 * coef_margin_cpu * requested_cpu,
                        requested_cpu)).round()
        else:
            allocation_cpu = reallocating_cpu
        if reallocating_ram:
            allocation_ram = dist_allocation_ram.sample()
            allocation_ram = torch.clamp(
                allocation_ram * (1 + allocation_noise_ram),
                min=min(predicted_ram + coef_margin_ram * requested_ram,
                        requested_ram),
                max=min(predicted_ram + 2 * coef_margin_ram * requested_ram,
                        requested_ram)).round()
        else:
            allocation_ram = reallocating_ram

        action = {
            'CPU':
            {
                'Reallocating': reallocating_cpu.int().item(),
                'Allocation': allocation_cpu.int().item()
            },
            'RAM':
            {
                'Reallocating': reallocating_ram.int().item(),
                'Allocation': allocation_ram.int().item()
            }
        }

        log_prob = dist_reallocating_cpu.log_prob(reallocating_cpu) +\
            dist_reallocating_ram.log_prob(reallocating_ram) +\
            dist_allocation_cpu.log_prob(allocation_cpu) +\
            dist_allocation_ram.log_prob(allocation_ram)

        return action, log_prob

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns

    def compute_baseline(self, states):
        values = self.value(states)
        return values.squeeze()

    def update_policy(self,
                      states,
                      actions,
                      log_probs,
                      returns):
        states = torch.stack(states).detach()

        actions = torch.stack(actions).detach()
        reallocating_cpu = actions[:, 0]
        reallocating_ram = actions[:, 1]
        allocation_cpu = actions[:, 2]
        allocation_ram = actions[:, 3]

        log_probs = torch.stack(log_probs).detach()

        for _ in range(self.optimization_epochs):

            new_probs_reallocating_cpu,\
                new_probs_reallocating_ram,\
                new_probs_allocation_cpu,\
                new_probs_allocation_ram = self.policy(states)

            new_dist_reallocating_cpu = Bernoulli(new_probs_reallocating_cpu)
            new_dist_reallocating_ram = Bernoulli(new_probs_reallocating_ram)

            new_dist_allocation_cpu = Categorical(new_probs_allocation_cpu)
            new_dist_allocation_ram = Categorical(new_probs_allocation_ram)

            new_log_probs_reallocating_cpu = new_dist_reallocating_cpu\
                .log_prob(reallocating_cpu)
            new_log_probs_reallocating_ram = new_dist_reallocating_ram\
                .log_prob(reallocating_ram)
            new_log_probs_allocation_cpu = new_dist_allocation_cpu\
                .log_prob(allocation_cpu)
            new_log_probs_allocation_ram = new_dist_allocation_ram\
                .log_prob(allocation_ram)

            new_log_probs = new_log_probs_reallocating_cpu +\
                new_log_probs_reallocating_ram +\
                new_log_probs_allocation_cpu +\
                new_log_probs_allocation_ram

            baseline = self.compute_baseline(states)
            advantages = returns - baseline

            ratio = torch.exp(new_log_probs - log_probs)
            obj = ratio * advantages.view(-1, 1)
            obj_clipped = ratio.clamp(1 - self.clip_ratio,
                                      1 + self.clip_ratio)\
                * advantages.view(-1, 1)
            policy_loss = -torch.min(obj, obj_clipped).mean()

            value_loss = F.smooth_l1_loss(baseline, returns)

            entropy = (new_dist_reallocating_cpu.entropy().mean() +
                       new_dist_reallocating_ram.entropy().mean() +
                       new_dist_allocation_cpu.entropy().mean() +
                       new_dist_allocation_ram.entropy().mean()) / 4

            loss = policy_loss +\
                self.vf_coef * value_loss -\
                self.entropy_coef * entropy

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()

            return loss, policy_loss, value_loss, entropy


def state_to_tensor(state):
    state_vec = np.array([state['CPU']['Requested'],
                          state['RAM']['Requested'],
                          state['CPU']['Used'],
                          state['RAM']['Used'],
                          state['CPU']['Allocated'],
                          state['RAM']['Allocated'],
                          state['CPU']['Predicted'],
                          state['RAM']['Predicted']])
    return torch.FloatTensor(state_vec)


def action_to_tensor(action):
    action_vec = np.array([action['CPU']['Reallocating'],
                           action['RAM']['Reallocating'],
                           action['CPU']['Allocation'],
                           action['RAM']['Allocation']])
    return torch.FloatTensor(action_vec)


def train(env,
          episodes,
          min_time_steps,
          max_time_steps,
          used_cpu_data,
          predicted_cpu_data,
          used_ram_data,
          predicted_ram_data,
          model_policy_path=None,
          plot_path=None):
    print('==== PPO agent training ====')

    agent = PPOAgent(env)

    episode_rewards = []
    losses = []
    policy_losses = []
    value_losses = []
    entropies = []

    for episode in range(episodes):
        states = []
        actions = []
        rewards = []
        log_probs = []

        state = env.reset()

        exploration_noise = DECAYING_EXPLORATION_NOISE_COEF *\
            max((1 - episode / episodes), 0)

        iteration_count = random.randint(min_time_steps, max_time_steps)
        i_start = random.randint(0, max_time_steps - iteration_count)

        for i in range(i_start, i_start + iteration_count - 1):
            state_vec = state_to_tensor(state)
            states.append(state_vec)

            action, log_prob = agent.select_action(
                state=state_vec,
                exploration_noise=exploration_noise,
                coef_margin_cpu=env.coef_margin_cpu,
                coef_margin_ram=env.coef_margin_ram)
            action_vec = action_to_tensor(action)
            actions.append(action_vec)
            log_probs.append(log_prob)

            next_used_cpu = used_cpu_data[i]
            next_used_ram = used_ram_data[i]
            next_predicted_cpu = predicted_cpu_data[i]  # predicted next
            next_predicted_ram = predicted_ram_data[i]  # predicted next
            next_state, reward, _ =\
                env.step(action=action,
                         next_used_cpu=next_used_cpu,
                         next_used_ram=next_used_ram,
                         next_predicted_cpu=next_predicted_cpu,
                         next_predicted_ram=next_predicted_ram)
            rewards.append(reward)

            state = next_state

        returns = agent.compute_returns(rewards)

        loss,\
            policy_loss,\
            value_loss,\
            entropy = agent.update_policy(states, actions, log_probs, returns)

        normalized_total_reward = sum(rewards) / iteration_count
        episode_rewards.append(normalized_total_reward)
        losses.append(loss.item())
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        entropies.append(entropy.item())

        if episode % 10 == 0:
            print(f'Episode {episode}, ' +
                  f'Normalized total reward: {normalized_total_reward}')

    env.close()

    if model_policy_path:
        torch.save(agent.policy.state_dict(), model_policy_path)

    if plot_path:
        plt.figure()
        plt.xlabel('Episode')
        plt.ylabel('Normalized total reward')
        plt.title('PPO agent training')
        plt.plot(episode_rewards, color='tab:blue',
                 label='Normalized total reward')
        plt.savefig(plot_path)

        plt.figure()
        plt.xlabel('Episode')
        plt.ylabel('Total oss')
        plt.title('PPO agent training')
        plt.plot(losses, color='tab:blue', label='Total loss')
        plt.savefig(plot_path.replace('reward', 'total_loss'))

        plt.figure()
        plt.xlabel('Episode')
        plt.ylabel('Policy loss')
        plt.title('PPO agent training')
        plt.plot(policy_losses, color='tab:blue', label='Policy loss')
        plt.savefig(plot_path.replace('reward', 'policy_loss'))

        plt.figure()
        plt.xlabel('Episode')
        plt.ylabel('Value loss')
        plt.title('PPO agent training')
        plt.plot(value_losses, color='tab:blue', label='Value loss')
        plt.savefig(plot_path.replace('reward', 'value_loss'))

        plt.figure()
        plt.xlabel('Episode')
        plt.ylabel('Entropy')
        plt.title('PPO agent training')
        plt.plot(entropies, color='tab:blue', label='Entropy')
        plt.savefig(plot_path.replace('reward', 'entropy'))

    return agent
