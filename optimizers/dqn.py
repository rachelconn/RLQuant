import random
from collections import deque
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model

from data_structures.replay_buffer import ReplayBuffer

def create_lstm_network(env, *_, lr, layer_nodes, time_steps):
    """ Builds an LSTM DQN network fitting the environment's state/action spaces.
    """
    state_size = env.observation_space.shape[0]
    nA = env.action_space.n

    model = Sequential()
    model.add(LSTM(layer_nodes[0], input_shape=(time_steps, state_size)))
    for num_nodes in layer_nodes[1:]:
        model.add(Dense(num_nodes, activation='relu'))
    model.add(Dense(nA, activation='linear'))
    model.compile(loss=Huber(), optimizer=Adam(lr=lr), loss_weights=1.0)

    return model

def load_dqn_model(model_location, target_model_location):
    model = load_model(model_location)
    target_model = load_model(target_model_location)
    return model, target_model

class DQN:
    def __init__(
        self,
        env,
        *_,
        model=None,
        target_model=None,
        layer_nodes=[128, 128],
        epsilon=0.99,
        lr=0.0003,
        gamma,
        time_steps,
        replay_buffer,
        update_target_every
    ):
        # Set params
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.nA = self.env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.time_steps = time_steps
        self.update_target_every = update_target_every
        self.until_target_update = self.update_target_every

        # Create replay buffer
        self.replay_buffer = replay_buffer

        # Use or create model
        if model is None or target_model is None:
            self.model = create_lstm_network(lr=lr, layer_nodes=layer_nodes, time_steps=time_steps)
            self.target_model = create_lstm_network(lr=lr, layer_nodes=layer_nodes, time_steps=time_steps)
        else:
            self.model = model
            self.target_model = target_model


    def update_target_model(self):
        """ Copies weights from model to target_model """
        self.target_model.set_weights(self.model.get_weights())

    def take_epsilon_greedy_action(self, state):
        epsilon_chance = self.epsilon / self.nA
        a_star = np.argmax(self.model(np.array([state]))[0])
        probs = np.repeat(epsilon_chance, self.nA)
        probs[a_star] = 1 - self.epsilon + epsilon_chance
        return np.random.choice(self.nA, p=probs)

    def add_episode_to_replay_buffer(self):
        # Reset the environment
        s = self.env.reset()
        done = False
        steps = deque((0,) * self.state_size for _ in range(self.time_steps - 1))
        steps.append(s)

        while not done:
            # Add step to replay buffer
            s_steps = list(steps)
            a = self.take_epsilon_greedy_action(s_steps)
            s_prime, r, done, _ = self.env.step(a)

            steps.popleft()
            steps.append(s_prime)
            s_prime_steps = list(steps)

            # Use selective replay buffer - experience from buying/selling is more important than taking neutral position
            if a != 1 or random.randrange(10) == 0:
                self.replay_buffer.add_sample((s_steps, a, r, s_prime_steps, done))

            s = s_prime

    def train_minibatch(self, batch_size, can_update_target=False):
        # Train from replay buffer
        def get_y(sample):
            _, _, replay_r, replay_s_prime, terminal = sample
            y = replay_r
            if not terminal:
                q_s_prime = np.max(self.target_model(np.array([replay_s_prime]))[0])
                y += self.gamma * q_s_prime
            return y

        batch = self.replay_buffer.sample_minibatch(batch_size)
        train_x = []
        train_y = []
        for sample in batch:
            sample_s, sample_a, sample_r, _, _ = sample
            train_x.append(sample_s)
            y = self.model(np.array([sample_s]))[0].numpy()
            y[sample_a] = get_y(sample)
            train_y.append(y)
        train_x = np.array(train_x)
        train_y = np.array(train_y)

        # SGD
        self.model.train_on_batch(train_x, train_y)

        # Update target network as needed
        if can_update_target:
            self.until_target_update -= 1
            if self.until_target_update == 0:
                self.update_target_model()
                self.until_target_update = self.update_target_every
