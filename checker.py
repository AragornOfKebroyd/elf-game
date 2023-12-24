import sys
# allow to import from files in this dir
sys.path.append('/optimisation')

from simulation import run_sim_set_t
from strategy import Strategy, STRATEGYTYPE
from objective_functions import ElfGameStats, run_eval_set_t

import time
import matplotlib.pyplot as plt
import numpy as np

def test_t(S, t, n=10_000, elfLoss=True,elfCount=12, plot=False):
    start = time.time()    
    predicted, Stats = run_eval_set_t(S, t, elfLoss=elfLoss, elfCount=elfCount)
    end = time.time()
    elapsed = round((end - start) * 1000, 3)
    print(f'predicted in {elapsed}ms')
    simulated = run_sim_set_t(S, t, n, elfLoss=elfLoss, elfCount=elfCount, plothist=plot)
    error = np.abs(predicted - simulated)
    perc = 100 * error / predicted
    print(f"##### t={t} #####")
    print("predicted:\n", np.round(predicted, decimals=4))
    print("simulated:\n", np.round(simulated, decimals=4))
    print("absolue error:\n", np.round(error, decimals=4))
    print("percentage error:\n", np.round(perc, decimals=4))
    print("\n")
    return Stats

def check_values(ElfStats=None):
    if ElfStats == None:
        W = np.repeat(np.array([[0.5, 0, 0.5], [0.8, 0.1, 0.1], [0.5, 0.3, 0.2], [0, 0.1, 0.9], [0, 0, 1]]), repeats=(1, 1, 17, 4, 1), axis=0) 
        S = Strategy(W, type=STRATEGYTYPE.FULL)
        ElfStats = ElfGameStats(S)
    
    print('Expected Value')
    print('1x:', ElfStats.EDc(0))
    print('12x:', 12 * ElfStats.EDc(0))
    print('\nVariance')
    print('1x:', ElfStats.VarDc(0))
    print('12x:', 144 * ElfStats.VarDc(0))

def check_string(strategy_string):
    array = np.array([np.float64(i) for i in strategy_string.replace('[', '').replace(']', '').split(', ')])
    print(array)
    S = Strategy(array, type=STRATEGYTYPE.IMPLIED)
    print(S)

def main():
    W = np.repeat(np.array([[0.5, 0.5, 0], [0.8, 0.2, 0], [0, 1, 0], [0.2, 0.5, 0.3], [0, 0.6, 0.4]]), repeats=(1, 1, 20, 1, 1), axis=0) # matches the overleaf doc
    S = Strategy(W, type=STRATEGYTYPE.FULL)
    E = ElfGameStats(S)

    check_values(E)

    # for t in range(23, -1, -1):
    #     test_t(S, t=t, elfLoss=True, elfCount=1)
    test_t(S,t=0,n=100_000, plot=True)
    
    plt.show()

if __name__ == '__main__':
    main()