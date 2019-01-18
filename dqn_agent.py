# In this agent file we want to implement Prioritised Experience Replay
import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
E = 5e-2                # small constant to add to td_error
ALPHA = 0.6             # PER hyperparameter 1
BETA  = 0.6             # PER hyperparameter 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
       
        # calculate td-error relative to current experience tuple
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values_state = self.qnetwork_local(state)
            action_values_next_state = self.qnetwork_local(next_state)
            
        self.qnetwork_local.train()
        
        max_q_next_state = np.max(action_values_next_state.cpu().data.numpy())
        expected_array = action_values_state.cpu().data.numpy()
        expected = expected_array.item(action)
        
        td_error = np.array(reward + GAMMA*max_q_next_state-expected)
        abs_td_error = torch.abs(torch.from_numpy(td_error))+E
        abs_td_error = abs_td_error.float().unsqueeze(0).to(device)
       
        # need to turn these back into numpy arrays before they get added to the replay buffer
        # this is because the replay buffer will operate on them on the basis that they are numpy arrays
        state = state.cpu().data.numpy()
        next_state = next_state.cpu().data.numpy()
        abs_td_error = abs_td_error.cpu().data.numpy()
        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, abs_td_error, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, abs_td_errors, dones, indices, weights = experiences
       
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Latest abs TD-errors. They will be transformed into proper priorities in the ReplayBuffer class
        priorities = torch.abs(Q_targets - Q_expected) + E
        
        # Compute loss
        loss = F.mse_loss(Q_expected*weights, Q_targets*weights)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        # update the latest batch priorities in the replay buffer
        self.memory.update_priorities(indices.data.cpu().numpy(), priorities.data.cpu().numpy())
        
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "priority", "done"])
        self.seed = random.seed(seed)
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.pos = 0
        self.buffer_size = buffer_size
    
    def add(self, state, action, reward, next_state, priority, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, priority, done)
        self.memory.append(e)
        
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.buffer_size
        
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
    
    def sample(self):
        """Sample a batch of experiences from memory based on their priority."""
        if len(self.memory) == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** ALPHA
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        experiences = [self.memory[idx] for idx in indices]
        
        total    = len(self.memory)
        weights  = (total * probs[indices]) ** (-BETA)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        priorities = torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        indices = torch.from_numpy(np.vstack([index for index in indices]).astype(np.uint8)).int().to(device)
        weights = torch.from_numpy(np.vstack([weight for weight in weights]).astype(np.float32)).float().to(device)
        return (states, actions, rewards, next_states, priorities, dones, indices, weights)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)