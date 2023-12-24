import numpy as np

def run_eval_set_t(S, t, elfLoss=True, elfCount=12):
    ElfStats = ElfGameStats(S, elfLoss=elfLoss)
    E = elfCount * ElfStats.EDc(t)
    VAR = elfCount**2 * ElfStats.VarDc(t)
    return np.array([E, VAR]), ElfStats

def evaluateStrategy(S):
    ElfStats = ElfGameStats(S)
    E = 12 * ElfStats.EDc(0)
    VAR = 12**2 * ElfStats.VarDc(0)
    return np.array([E, VAR])

class LOCATION:
    NF = 0
    FF = 1
    FFB = 2

class ElfGameStats:
    def __init__(self, strategy, elfLoss=True):
        self.strategy = np.array(strategy) # shape = (25, 3)
        
        self.EDc_cache = {}
        self.VarDc_cache = {}

        self.elfLoss = elfLoss

    def Re(self, Loc):
        match Loc:
            case LOCATION.NF:
                res = 10
            case LOCATION.FF:
                res = 20
            case LOCATION.FFB:
                res = 50
            case _:
                raise Exception('Not a valid location')
        return res

    def ED(self, t):
        ReArr = np.array([self.Re(L) for L in [LOCATION.NF, LOCATION.FF, LOCATION.FFB]])
        w = self.strategy[t]
        res = 2/3 * np.dot(w, ReArr) + 1/3 * w[0] * ReArr[0]
        return res
    
    def ED_sq(self, t):
        ReArr = np.array([self.Re(L) for L in [LOCATION.NF, LOCATION.FF, LOCATION.FFB]])
        w = self.strategy[t]
        res = 2/3 * (np.dot(w, ReArr))**2 + 1/3 * (w[0] * ReArr[0])**2
        return res
    
    def VarD(self, t):
        res = self.ED_sq(t) - self.ED(t)**2
        return res

    def EM(self, t):
        S_t_FFB =  self.strategy[t, 2]
        res = (3 - S_t_FFB) / 3
        return res
    
    def EM_sq(self, t):
        S_t_FFB =  self.strategy[t, 2]
        res = (3 - 2 * S_t_FFB + S_t_FFB**2) / 3
        return res
    
    def VarM(self, t):
        S_t_FFB =  self.strategy[t, 2]
        res = 2/9 * S_t_FFB**2
        return res

    def EDc(self, t):
        if t in self.EDc_cache: return self.EDc_cache[t]
        if t == 23: return self.ED(23)
        res = self.ED(t) + self.EM(t) * self.EDc(t+1)
        return res

    def a_t(self, t):
        ReArr = np.array([self.Re(L) for L in [LOCATION.NF, LOCATION.FF, LOCATION.FFB]])
        w = self.strategy[t]
        res = (w[1] * ReArr[1] + w[2] * ReArr[2]) / w[2]
        return res

    def VarDc(self, t):
        if t == 23: return self.VarD(23)
        if self.strategy[t, 2] == 0:
            res = self.VarD(t) + self.VarDc(t+1)
        else:
            part1 = self.VarD(t)
            part2 = self.EM_sq(t) * (self.VarDc(t+1) + self.EDc(t+1)**2)
            part3 = (self.EM(t) * self.EDc(t+1))**2
            part4 = 2 * self.a_t(t) * self.EDc(t+1) * self.VarM(t)
            res = part1 + part2 - part3 + part4
        return res