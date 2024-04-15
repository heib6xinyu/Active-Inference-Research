import numpy as np

class ReplayBuffer():
    
    def __init__(self, max_size, input_shape, n_actions):
    # @param    max_size: the maximum memory space for the agent
    # @param    input_shape: the observation dimension
    # @param    n_actions: number of component of the action?
        self.mem_size = max_size
        self.mem_cntr = 0 #index of the position of the first available memory
        self.state_memory = np.zeros((self.mem_size, *input_shape)) # * unpack input_shape
        #For example, if self.mem_size = 1000 and input_shape = (84, 84, 3), then the line of code would be equivalent to:
        #self.state_memory = np.zeros((1000, 84, 84, 3))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape)) #result after our action
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)

    
    def store_transition(self, state, action, reward, state_, done):
    # @param    state: current state
    # @param    action: action the agent take
    # @param    reward: reward from the environment
    # @param    state_: next state
    # @param    done: terminal flag
        #first, figure out where our first available memory is
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1  # so memory counter is coutinuous from the very beginning of the experiment, but will overwrite the old ones when mem_size is exceeded?


    def sample_buffer(self, batch_size):
    # @param    batch_size: sample how many buffer
    # @return   states, actions, rewards, states_, done: the sampled data
        max_mem = min(self.mem_cntr, self.mem_size) 
        #how many memory is already stored, if mem_size is the smaller, there will be no empty memory space (or in other words, memory started to overwrite)

        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones