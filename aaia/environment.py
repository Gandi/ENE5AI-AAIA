import gymnasium as gym
from gymnasium.spaces import Dict, Discrete


# Reward params
REWARD_UNDERALLOCATION_CPU = -10
REWARD_UNDERALLOCATION_RAM = -125
REWARD_ALLOCATION_MIN_CPU = 0
REWARD_ALLOCATION_MIN_RAM = 0
REWARD_ALLOCATION_MAX_CPU = 100
REWARD_ALLOCATION_MAX_RAM = 100
REWARD_RELLOCATING_CPU = -1
REWARD_RELLOCATING_RAM = -75


class ResourceAllocationEnv(gym.Env):

    def __init__(self,
                 requested_cpu,
                 requested_ram,
                 coef_margin_cpu,
                 coef_margin_ram):
        super(ResourceAllocationEnv, self).__init__()

        self.requested_cpu = requested_cpu
        self.requested_ram = requested_ram
        self.used_cpu = requested_cpu
        self.used_ram = requested_ram
        self.predicted_cpu = requested_cpu
        self.predicted_ram = requested_ram

        self.coef_margin_cpu = coef_margin_cpu
        self.coef_margin_ram = coef_margin_ram

        self.state_space = Dict(
            {
                'CPU': Dict(
                    {
                        'Requested': Discrete(self.requested_cpu + 1),
                        'Used': Discrete(self.requested_cpu + 1),
                        'Allocated': Discrete(self.requested_cpu + 1),
                        'Predicted': Discrete(self.requested_cpu + 1)
                    }
                ),
                'RAM': Dict(
                    {
                        'Requested': Discrete(self.requested_ram + 1),
                        'Used': Discrete(self.requested_ram + 1),
                        'Allocated': Discrete(self.requested_ram + 1),
                        'Predicted': Discrete(self.requested_ram + 1)
                    }
                )
            }
        )

        self.action_space = Dict(
            {
                'CPU': Dict(
                    {
                        'Reallocating': Discrete(2),
                        'Allocation': Discrete(self.requested_cpu + 1)
                    }
                ),
                'RAM': Dict(
                    {
                        'Reallocating': Discrete(2),
                        'Allocation': Discrete(self.requested_ram + 1)
                    }
                )
            }
        )

        self.reallocating_cpu = False
        self.reallocating_ram = False
        self.allocated_cpu = self.requested_cpu
        self.allocated_ram = self.requested_ram

    def step(self,
             action,
             next_used_cpu,
             next_used_ram,
             next_predicted_cpu,
             next_predicted_ram):
        assert self.action_space.contains(action), 'Invalid action'

        self.used_cpu = next_used_cpu
        self.used_ram = next_used_ram
        self.predicted_cpu = next_predicted_cpu
        self.predicted_ram = next_predicted_ram

        self.reallocating_cpu = action['CPU']['Reallocating']
        self.reallocating_ram = action['RAM']['Reallocating']

        if self.reallocating_cpu:
            self.allocated_cpu = action['CPU']['Allocation']
        if self.reallocating_ram:
            self.allocated_ram = action['RAM']['Allocation']

        reward = self.calculate_reward()

        state = {
            'CPU': {
                'Requested': self.requested_cpu,
                'Used': self.used_cpu,
                'Allocated': self.allocated_cpu,
                'Predicted': self.predicted_cpu
            },
            'RAM': {
                'Requested': self.requested_ram,
                'Used': self.used_ram,
                'Allocated': self.allocated_ram,
                'Predicted': self.predicted_ram
            }
        }

        return state, reward, {}

    def reset(self):
        self.used_cpu = self.requested_cpu
        self.used_ram = self.requested_ram

        self.reallocating_cpu = False
        self.reallocating_ram = False
        self.allocated_cpu = self.requested_cpu
        self.allocated_ram = self.requested_ram
        self.predicted_cpu = self.requested_cpu
        self.predicted_ram = self.requested_ram

        state = {
            'CPU': {
                'Requested': self.requested_cpu,
                'Used': self.used_cpu,
                'Allocated': self.allocated_cpu,
                'Predicted': self.allocated_cpu
            },
            'RAM': {
                'Requested': self.requested_ram,
                'Used': self.used_ram,
                'Allocated': self.allocated_ram,
                'Predicted': self.allocated_ram
            }
        }

        return state

    def calculate_reward(self):
        reward_cpu = 0
        reward_ram = 0

        margin_cpu = self.coef_margin_cpu * self.requested_cpu

        if self.allocated_cpu <= self.used_cpu:
            reward_cpu += REWARD_UNDERALLOCATION_CPU
        elif self.allocated_cpu <= self.used_cpu + margin_cpu:
            reward_cpu +=\
                (self.allocated_cpu - self.used_cpu) / margin_cpu *\
                (REWARD_ALLOCATION_MAX_CPU - REWARD_UNDERALLOCATION_CPU) +\
                REWARD_UNDERALLOCATION_CPU
        else:  # self.allocated_cpu <= self.requested_cpu
            reward_cpu +=\
                (self.allocated_cpu - (self.used_cpu + margin_cpu)) /\
                (self.requested_cpu - (self.used_cpu + margin_cpu)) *\
                (REWARD_ALLOCATION_MIN_CPU - REWARD_ALLOCATION_MAX_CPU) +\
                REWARD_ALLOCATION_MAX_CPU

        if self.reallocating_cpu:
            reward_cpu += REWARD_RELLOCATING_CPU

        margin_ram = self.coef_margin_ram * self.requested_ram

        if self.allocated_ram <= self.used_ram:
            reward_ram += REWARD_UNDERALLOCATION_RAM
        elif self.allocated_ram <= self.used_ram + margin_ram:
            reward_ram +=\
                (self.allocated_ram - self.used_ram) / margin_ram *\
                (REWARD_ALLOCATION_MAX_RAM - REWARD_UNDERALLOCATION_RAM) +\
                REWARD_UNDERALLOCATION_RAM
        else:  # self.allocated_ram <= self.requested_ram
            reward_ram +=\
                (self.allocated_ram - (self.used_ram + margin_ram)) /\
                (self.requested_ram - (self.used_ram + margin_ram)) *\
                (REWARD_ALLOCATION_MIN_RAM - REWARD_ALLOCATION_MAX_RAM) +\
                REWARD_ALLOCATION_MAX_RAM

        if self.reallocating_ram:
            reward_ram += REWARD_RELLOCATING_RAM

        return reward_cpu + reward_ram

    def render(self, mode='human'):
        cpu_usage_pct = self.used_cpu / self.requested_cpu * 100
        ram_usage_pct = self.used_ram / self.requested_ram * 100
        print('CPU used/requested: \t' +
              f'{self.used_cpu}\t/{self.requested_cpu}\t' +
              f'{round(cpu_usage_pct)}%')
        print('RAM used/requested: \t' +
              f'{self.used_ram}\t/{self.requested_ram}\t' +
              f'{round(ram_usage_pct)}%')
        cpu_allocation_pct = self.allocated_cpu / self.requested_cpu * 100
        ram_allocation_pct = self.allocated_ram / self.requested_ram * 100
        print('CPU allocated/requested: ' +
              f'{self.allocated_cpu}\t/{self.requested_cpu}\t' +
              f'{round(cpu_allocation_pct)}%')
        print('RAM allocated/requested: ' +
              f'{self.allocated_ram}\t/{self.requested_ram}\t' +
              f'{round(ram_allocation_pct)}%')
