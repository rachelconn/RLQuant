import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Softmax, Input, LSTM, SimpleRNN, Embedding, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model

def actor_loss():
    def loss(advantage, predicted_output):
        """
            The policy gradient loss function.
            Note that you are required to define the Loss^PG
            which should be the integral of the policy gradient
            The "returns" is the one-hot encoded (return - baseline) value for each action a_t
            ('0' for unchosen actions).

            args:
                advantage: advantage of each action a_t (one-hot encoded).
                predicted_output: Predicted actions (action probabilities).

            Use:
                K.log: Element-wise log.
                K.sum: Sum of a tensor.
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        predicted_output = K.clip(predicted_output, 1e-8, 1-1e-8)
        log_probs = K.log(predicted_output)
        losses = -K.sum(advantage * log_probs, -1)
        # losses = K.print_tensor(losses, 'actor loss:')
        return losses

    return loss

def critic_loss():
    def loss(advantage, predicted_output):
        """
            The integral of the critic gradient

            args:
                advantage: advantage of each action a_t (one-hot encoded).
                predicted_output: Predicted state value.

            Use:
                K.sum: Sum of a tensor.
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        # diffs = K.sum((advantage + predicted_output) - predicted_output, -1)
        # losses = K.mean(K.square(diffs))
        # losses = K.mean(K.sum(K.square(advantage), 1)) * predicted_output
        # losses = K.sum(0 * predicted_output + advantage)
        losses = -K.sum(advantage * predicted_output, -1)
        # losses = K.print_tensor(losses, 'critic loss:')
        return losses

    return loss

def load_a2c_lstm_model(actor_location, critic_location):
    """ Loads A2C LSTM networks from folders specified """
    actor = load_model(actor_location, custom_objects={'loss': actor_loss()})
    critic = load_model(actor_location, custom_objects={'loss': critic_loss()})
    return actor, critic

def build_a2c_networks(env, *_, lr, layer_nodes=[64, 64], time_steps):
    """ Builds an LSTM A2C network fitting the environment's state/action spaces.
    """
    state_size = env.observation_space.shape[0]
    nA = env.action_space.n

    actor = Sequential()
    actor.add(LSTM(layer_nodes[0], input_shape=(time_steps, state_size)))
    for num_nodes in layer_nodes[1:]:
        actor.add(Dense(num_nodes, activation='tanh'))
    actor.add(Dense(nA, activation='tanh'))
    actor.add(Softmax())
    actor.compile(loss=actor_loss(), optimizer=Adam(lr=lr), loss_weights=1.0)

    critic = Sequential()
    critic.add(LSTM(layer_nodes[0], input_shape=(time_steps, state_size)))
    for num_nodes in layer_nodes[1:]:
        critic.add(Dense(num_nodes, activation='tanh'))
    critic.add(Dense(1, activation='linear', name='state_value'))
    critic.compile(loss=critic_loss(), optimizer=Adam(lr=lr), loss_weights=1.0)

    return actor, critic

class A2C_LSTM():
    """ A2C optimizer.
        If this doesn't suffice to properly learn the environment, there are a couple potential improvements:
            - Include entropy in loss calculations
            - Use n-step bootstrapping instead of 1-step
            - Dynamic learning rate
    """
    def __init__(self, env, *_, lr=0.001, gamma=0.99, time_steps=4, actor=None, critic=None):
        # Environment data
        self.env = env
        self.nA = self.env.action_space.n
        self.episodes_run = 0

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.time_steps = time_steps

        # Use provided network or build a default one
        if actor is None or critic is None:
            self.actor, self.critic = build_a2c_networks(env, lr=self.lr, time_steps=time_steps)
        else:
            self.actor, self.critic = actor, critic

    def get_policy(self, state):
        test = np.array([state])
        ret = self.actor(np.array([state]))
        return ret[0].numpy()

    def choose_action(self, state):
        return np.random.choice(self.nA, p=self.get_policy(state))

    def generate_episode(self, render=False):
        """ Generates an episode and returns the trajectory """
        total_r = 0

        s = self.env.reset()
        done = False
        episode = []
        while not done:
            # Create time steps for each state transition
            steps = [s]
            for i in range(1, self.time_steps):
                if len(episode) >= i:
                    steps.append(episode[-i][0][-1])
                else:
                    steps.append((0,) * len(s))
            steps.reverse()

            a = self.choose_action(steps)
            s_prime, r, done, _ = self.env.step(a)
            episode.append((steps, a, r))

            s = s_prime
            total_r += r

            if render:
                self.env.render()

        self.env.close()
        self.episodes_run += 1

        return episode

    def train_on_episode(self, render=False):
        """ Trains on a single episode """
        episode = self.generate_episode(render=render)

        # Estimate state values
        x_train = np.array([s for s, _, _ in episode])
        state_values = [prediction[0] for prediction in self.critic.predict(x_train)]

        # Go through episode determining action encodings and advantages
        deltas = np.empty((len(episode), self.nA))
        for i, ((s, a, r), v_s) in enumerate(zip(episode, state_values)):
            v_s_prime = state_values[i + 1] if i < len(episode) - 1 else 0
            advantage = r + self.gamma * v_s_prime - v_s

            # delta: advantage for taken action, 0 for other actions
            delta = np.zeros(self.env.action_space.n)
            delta[a] = advantage
            deltas[i] = delta

        y_train = {
            'policy': deltas,
            'state_value': deltas,
        }

        # Update actor critic
        self.actor.train_on_batch(x=x_train, y=deltas)
        self.critic.train_on_batch(x=x_train, y=deltas)

    def train(self, n, *_, render_every=None):
        """ Trains on n episodes """
        for i in range(1, n + 1):
            render = render_every and i % render_every == 0
            self.train_on_episode(render=render)

    def get_models(self):
        """ Returns the actor-critic network for serialization or other external use """
        return self.actor, self.critic
