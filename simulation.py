import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

class ElfGame:
    def __init__(self, strategy, elfLoss=True, elfCount=12):
        self.strategy = strategy
        self.sunny = lambda: np.random.rand() > 1/3
        self.elfLoss = elfLoss
        self.elves = elfCount
        self.reset()
    
    def reset(self):
        self.elfCount = self.elves
        self.money = 0
        self.t = 0
        self.history = []
        self.done = False

    def take_action(self, w):
        sunny = self.sunny()
        elves = w * self.elfCount
        rewards = np.array([10, 20 * sunny, 50 * sunny])
        reward = np.dot(elves, rewards)
        if not sunny:
            elvesLost = elves[-1]
        else:
            elvesLost = 0
        return reward, sunny, elvesLost

    def step(self):
        if self.done: return 'Done'
        w = self.strategy[self.t]
        reward, sunny, elvesLost = self.take_action(w)
        self.history.append([self.t, sunny,  elvesLost, reward, self.money])
        if self.elfLoss: self.elfCount -= elvesLost
        self.money += reward
        self.t += 1
        if self.t == len(self.strategy):
            self.done = True
            self.history = np.array(self.history)
        return 0
    
    def run(self):
        self.reset()
        while True:
            if self.step() == 'Done': break
    
    def price_history(self):
        if not self.done: return 'Must be done'
        return self.history[:, 3]

def run_sim_set_t(S, t, n=1_000, elfLoss=True, elfCount=12, plothist=False):
    usedS = S[t:]
    Env = ElfGame(usedS, elfLoss=elfLoss, elfCount=elfCount)
    results = np.array([])
    for _ in range(n):
        Env.run()
        results = np.append(results, Env.money)

    E = np.mean(results)
    VAR = np.var(results)

    if plothist:
        plt.hist(results, bins=20)
    return np.array([E, VAR])