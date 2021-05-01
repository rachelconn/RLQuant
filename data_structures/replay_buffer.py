import random

class ReplayBuffer:
    def __init__(self, n):
        self.buf = []
        self.samples = 0
        self.next_pos = 0
        self.n = n

    def add_sample(self, transition):
        if (self.samples < self.n):
            self.buf.append(transition)
        else:
            self.buf[self.next_pos] = transition

        self.next_pos = (self.next_pos + 1) % self.n
        self.samples = min(self.samples + 1, self.n)

    def sample_minibatch(self, k):
        # return np.random.choice(self.buf, min(self.samples, k))
        return random.sample(self.buf, min(self.samples, k))
