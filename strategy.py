import numpy as np
import sys
# allow to import from files in this dir
sys.path.append('/optimisation')
from objective_functions import evaluateStrategy
from objective_functions import ElfGameStats

class STRATEGYTYPE:
    FULL = 1
    IMPLIED = 2

def evaluateStrategyWeights(W):
    S = Strategy(W, type=STRATEGYTYPE.IMPLIED, strict=True)
    return evaluateStrategy(S)

class Strategy:
    def __init__(self, W=None, type=STRATEGYTYPE.FULL, strict=False):
        self.initialised = False
        if W is None:
            self.weights = np.empty(shape=(25,3))
        else:
            W = np.array(W)
            match type:
                case STRATEGYTYPE.FULL:
                    self.full_init(W)
                case STRATEGYTYPE.IMPLIED:
                    self.implied_init(W, strict=strict)

    def full_init(self, day_weights):
        self.weights = day_weights
        self.initialised = True

    def implied_init(self, day_weights, strict=False): # shapes = (16), (8,2)
        # check validity of inputs
        two_loc_days, three_loc_days = day_weights[:16], np.reshape(day_weights[16:], (8,2))
        invalid_2 = two_loc_days[np.logical_or(two_loc_days > 1, two_loc_days < 0)]
        invalid_3 = three_loc_days[np.sum(three_loc_days, axis=1) > 1]
        invalid_3a = three_loc_days[three_loc_days < 0]
        if invalid_2.size or invalid_3.size or invalid_3a.size:
            if strict:
                print(two_loc_days, three_loc_days)
                raise Exception('Constraints are not held')

        # create weights
        weights = np.zeros(shape=(24,3))
        weights[:16,0] =  two_loc_days
        weights[:16,1]  = 1 - two_loc_days
        weights[16:,:2] = three_loc_days
        weights[16:,2] = 1 - np.sum(three_loc_days, axis=1)
        self.weights = weights
        self.initialised = True
    
    def __getitem__(self, i, j=None):
        return self.weights[i, :]

    def __str__(self) -> str:
        if not self.initialised:
            return 'Not Initialised'
        return 'Strategy:\n' + np.array2string(self.weights) + '\n'
    
    def __len__(self):
        return len(self.weights)