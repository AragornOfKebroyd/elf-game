import numpy as np

def run_eval_set_t(S, t, elfLoss=True, elfCount=12):
    ElfStats = ElfGameStats(S, elfLoss=elfLoss)
    E = elfCount * ElfStats.V_A(t)
    VAR = elfCount**2 * ElfStats.R_A(t)
    return np.array([E, VAR]), ElfStats

class LOCATION:
    NF = 0
    FF = 1
    FFB = 2

def evaluateStrategy(S):
    ElfStats = ElfGameStats(S)
    E = 12 * ElfStats.V_A(0)
    VAR = 12**2 * ElfStats.R_A(0)
    return np.array([E, VAR])

class ElfGameStats:
    def __init__(self, strategy, elfLoss=True):
        self.strategy = strategy # shape = (25, 3)
        # E(X) #
        self.Ecache = {}
        self.Vcache = {}
        self.VAcache = {}
        self.EAcache = {}

        # Var(X) #
        self.Varcache = {}
        self.VarAcache = {}
        self.Rcache = {}
        self.RAcache = {}
        self.caches = {
            'E': self.Ecache, 'V': self.Vcache, 'VA': self.VAcache, 'EA': self.EAcache, 'Var': self.Varcache, 'VarA': self.VarAcache, 'R': self.Rcache, 'RA': self.RAcache
        }

        self.elfLoss = elfLoss

    ##### Expected Value #####
    
    def L(self, Loc):
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

    def E(self, L):
        # check cache
        if L in self.Ecache: return self.Ecache[L]

        # Logic
        match L:
            case LOCATION.NF:
                res = 10
            case LOCATION.FF:
                res = 40 / 3
            case LOCATION.FFB:
                res = 100 / 3
            case _:
                raise Exception('Not a valid location')
        
        # save cache
        self.Ecache[L] = res
        return res

    def V(self, t):
        # check cache
        if t in self.Vcache: return self.Vcache[t]

        # Logic
        w = self.strategy[t]
        e = np.array([self.E(L) for L in [LOCATION.NF, LOCATION.FF, LOCATION.FFB]])
        res = np.dot(w, e)

        # save cache
        self.Vcache[t] = res
        return res

    def V_A(self, t):
        # check cache
        if t in self.VAcache: return self.VAcache[t]

        # Logic
        if t == 23: return self.V(23)
        w = self.strategy[t]
        e = np.array([self.E_A(L, t) for L in [LOCATION.NF, LOCATION.FF, LOCATION.FFB]])
        res = np.dot(w, e)

        # Save cache
        self.VAcache[t] = res
        return res
    
    def E_A(self, L, t):
        # check cache
        if (L,t) in self.EAcache: return self.EAcache[(L, t)]

        # Logic
        match L:
            case LOCATION.NF | LOCATION.FF:
                res = self.E(L) + self.V_A(t+1)
            case LOCATION.FFB:
                if self.elfLoss:
                    res =  self.E(L) + self.V_A(t+1) * 2/3
                else:
                    res = self.E(L) + self.V_A(t+1)

            case _:
                raise Exception('Not a valid location')
                
        # save cache
        self.EAcache[(L, t)] = res
        return res
    
    ##### Variance #####
    def Var(self, L):
        # check cache
        if L in self.Varcache: return self.Varcache[L]

        # Logic
        match L:
            case LOCATION.NF:
                res = 0
            case LOCATION.FF:
                res = 800 / 9
            case LOCATION.FFB:
                res = 5000 / 9
            case _:
                raise Exception('Not a valid location')
        
        # save cache
        self.Varcache[L] = res
        return res
    
    def Var_A(self, L, t):
        # check cache
        if (L, t) in self.VarAcache: return self.VarAcache[(L, t)]

        # Logic
        match L:
            case LOCATION.NF | LOCATION.FF:
                res = self.R_A(t+1)
            case LOCATION.FFB:
                res = 2/3 * self.R_A(t+1) + 2/9 * (self.V_A(t+1) + 50)**2
            case _:
                raise Exception('Not a valid location')
        
        # save cache
        self.VarAcache[(L, t)] = res
        return res
    
    def R(self, t): # correct
        # check cache
        if t in self.Rcache: return self.Rcache[t]

        # Logic
        w = self.strategy[t]
        e = np.array([self.L(L) for L in [LOCATION.NF, LOCATION.FF, LOCATION.FFB]])
        C1 = np.dot(w,e)
        C2 = w[0] * e[0]
        
        res = 2/3 * C1**2 + 1/3 * C2**2 - (2/3 * C1 + 1/3 * C2)**2
        # save cache
        self.Rcache[t] = res
        return res

    def R_A(self, t): # unsure
        # check cache
        if t in self.RAcache: return self.RAcache[t]

        # Logic
        if t == 23: return self.R(23)
        w = self.strategy[t]
        if w[-1] == 0:
            res = self.R(t) + self.R_A(t+1)
        else:
            e = np.array([self.L(L) for L in [LOCATION.NF, LOCATION.FF, LOCATION.FFB]])
            M1 = np.dot(w, e)
            M2 = w[0] * e[0]
            EM = (3-w[-1])/3
            EMSQ = (w[-1]**2-2*w[-1]+3)/3
            a = (M1 - M2) / w[-1]

            # print(f"t:{t}\n",self.R(t), (self.R_A(t+1) + self.V_A(t+1)**2), a)
            res = self.R(t) + EMSQ * (self.R_A(t+1) + self.V_A(t+1)**2) - EM**2 * self.V_A(t+1)**2 + 2 * a * self.V_A(t+1) * (EMSQ-EM**2)
        return res
