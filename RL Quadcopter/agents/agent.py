import random
from collections import namedtuple, deque

class PEReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, e = 0.01, a = 0.6):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.probs = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.count = 0
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
        self.e = e
        self.a = a
        

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if self.count < self.buffer_size: 
            self.memory.append(self.experience(state, action, reward, next_state, done))
            self.probs.append(np.float32(1.0)) # ((abs(reward) + self.e) ** self.a)
            self.count += 1
        else:
            self.memory.popleft()
            self.probs.popleft()
            self.memory.append(self.experience(state, action, reward, next_state, done))
            self.probs.append(np.float32(1.0)) # ((abs(reward) + self.e) ** self.a)
        
        
    def normalize_probs(self):
        self.probs = list(self.probs / sum(self.probs))
        
    def get_experience_by_index(self, index):
        return self.memory[index]
        
    def get_probs_by_index(self, index):
        return self.probs[index]

    def sample_indices(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        idx = np.random.choice(range(len(self.memory)), size=self.batch_size, p=self.probs)
        return idx

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
from keras import layers, models, optimizers
from keras import backend as K

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, batch_size, steps):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.steps = steps
        self.batch_size = batch_size
        self.ACTOR_LEARNING_RATE = 0.00001

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.steps, 6,), name='states')

        # Add hidden layers
        net = layers.Conv1D(filters=32, kernel_size = 3, padding='same', kernel_initializer=layers.initializers.glorot_normal(seed=None))(states)
        net = layers.normalization.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Conv1D(filters=32, kernel_size = 3, padding='valid', kernel_initializer=layers.initializers.glorot_normal(seed=None))(net)
        net = layers.normalization.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.MaxPooling1D(pool_size=2)(net)
        #net = layers.Dropout(0.25)(net)
        
        net = layers.Conv1D(filters=64, kernel_size = 3, padding='same', kernel_initializer=layers.initializers.glorot_normal(seed=None))(net)
        net = layers.normalization.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Conv1D(filters=64, kernel_size = 3, padding='valid', kernel_initializer=layers.initializers.glorot_normal(seed=None))(net)
        net = layers.normalization.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.MaxPooling1D(pool_size=2)(net)
        #net = layers.Dropout(0.25)(net)
        
        net = layers.Conv1D(filters=128, kernel_size = 3, padding='same', kernel_initializer=layers.initializers.glorot_normal(seed=None))(net)
        net = layers.normalization.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Conv1D(filters=128, kernel_size = 3, padding='valid', kernel_initializer=layers.initializers.glorot_normal(seed=None))(net)
        net = layers.normalization.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.MaxPooling1D(pool_size=2)(net)
        net = layers.Flatten()(net)
        net = layers.Dense(units=512, activation='relu', kernel_initializer=layers.initializers.glorot_normal(seed=None))(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size,
            name='raw_actions')(net)
        raw_actions = layers.normalization.BatchNormalization()(raw_actions)
        raw_actions = layers.Activation('sigmoid')(raw_actions)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x, range, low: ((x * range) + low), arguments={"range": self.action_range, "low": self.action_low},
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(self.ACTOR_LEARNING_RATE)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        actor_gradients = list(map(lambda x: (x / self.batch_size), updates_op))
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=actor_gradients)
            
class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.CRITIC_LEARNING_RATE = 0.0001

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=512, kernel_initializer=layers.initializers.RandomUniform(minval=-0.003, maxval=0.003))(states)
        net_states = layers.normalization.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dense(units=256, kernel_initializer=layers.initializers.RandomUniform(minval=-0.003, maxval=0.003))(net_states)
        net_states = layers.normalization.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dense(units=128, kernel_initializer=layers.initializers.RandomUniform(minval=-0.003, maxval=0.003))(net_states)
        net_states = layers.normalization.BatchNormalization()(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=256, kernel_initializer=layers.initializers.RandomUniform(minval=-0.003, maxval=0.003))(actions)
        net_actions = layers.normalization.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)
        net_actions = layers.Dense(units=128, kernel_initializer=layers.initializers.RandomUniform(minval=-0.003, maxval=0.003))(net_actions)
        net_actions = layers.normalization.BatchNormalization()(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.
        

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)
        
        # Add more layers to the combined network if needed
        net = layers.Dense(units=512, kernel_initializer=layers.initializers.RandomUniform(minval=-0.003, maxval=0.003))(net)
        net = layers.normalization.BatchNormalization()(net)
        net = layers.Activation('relu')(net)


        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, kernel_initializer=layers.initializers.RandomUniform(minval=-0.003, maxval=0.003), 
            name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(self.CRITIC_LEARNING_RATE)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
            
import numpy as np
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.drop_noise = False
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        if self.drop_noise: 
            return 0
        
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
        
class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.save_file_name = "saved_actor_models/weights.actor.best.hdf5"
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.batch_size = 64
        self.steps = task.action_repeat

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.batch_size, self.steps)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.batch_size, self.steps)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 1.
        self.exploration_sigma = 0.1
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        
        self.memory = PEReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters
        
        self.score = -1
        self.best_score = -1
        self.count = 0
        self.adding_noise_period = 1000
        self.stop_noise_num = 3000
        print("The noise is stopped after each {} steps and added again after each {} episodes. ".format(self.stop_noise_num, self.adding_noise_period))
        
    def save_actor_model(self):
        print("\nSaving the actor...")
        self.actor_local.model.save(self.save_file_name)
        print("\nDone!")
        
    def load_actor_model(self):
        print("\nLoading the actor...")
        self.actor_local.model = models.load_model(self.save_file_name)
        print("\nDone!")

    def reset_episode(self, remaining_episode=0):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        
        if (remaining_episode % self.adding_noise_period) == 0:
            if self.best_score < self.score: 
                self.best_score = self.score
            else:
                self.count = 0
                self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
            
        return state

    def step(self, action, reward, next_state, done):
        self.count += 1
        self.score = self.score + 1.0 / self.count * (reward - self.score)
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            self.memory.normalize_probs()
            experiences_indices = self.memory.sample_indices()
            self.learn(experiences_indices)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.steps, 6])
        action = self.actor_local.model.predict(state)[0]
        
        if self.count == self.stop_noise_num:
            self.noise.drop_noise = True
            print("\nStop annoying it!")
        
        return list(action + self.noise.sample())

    def learn(self, experiences_indices):
        """Update policy and value parameters using given batch of experience tuples."""
        p_b_weights = np.array([1.0 for idx in range(len(experiences_indices))]).astype(np.float32)
        # if len(p_b_weights) > 0:
            # for idx in range(len(p_b_weights)):
                # prob = self.memory.get_probs_by_index(experiences_indices[idx])
                # p_b_weights[idx] = (1.0 / prob)
        
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([self.memory.get_experience_by_index(id).state for id in experiences_indices if self.memory.get_experience_by_index(id) is not None])
        actions = np.array([self.memory.get_experience_by_index(id).action for id in experiences_indices if self.memory.get_experience_by_index(id) is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([self.memory.get_experience_by_index(id).reward for id in experiences_indices if self.memory.get_experience_by_index(id) is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([self.memory.get_experience_by_index(id).done for id in experiences_indices if self.memory.get_experience_by_index(id) is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([self.memory.get_experience_by_index(id).next_state for id in experiences_indices if self.memory.get_experience_by_index(id) is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        
        actions_next = self.actor_target.model.predict_on_batch(next_states.reshape(-1, self.steps, 6))
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets, sample_weight=p_b_weights)

        # Train actor model (local)
        actions_for_training = self.actor_local.model.predict_on_batch(states.reshape(-1, self.steps, 6))
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions_for_training, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states.reshape(-1, self.steps, 6), action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)