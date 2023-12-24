import numpy as np
import sys
# allow to import from files in this dir
sys.path.append('/optimisation')
from simulation import run_sim_set_t
from strategy import Strategy, STRATEGYTYPE
from objective_functions import ElfGameStats, run_eval_set_t
import time
import matplotlib.pyplot as plt

def test_t(S, t, n=10_000, elfLoss=True,elfCount=12, plot=False):
    start = time.time()    
    predicted, Stats = run_eval_set_t(S, t, elfLoss=elfLoss, elfCount=elfCount)
    end = time.time()
    elapsed = round((end - start) * 1000, 3)
    print(f'predicted in {elapsed}ms')
    simulated = run_sim_set_t(S, t, n, elfLoss=elfLoss, elfCount=elfCount, plothist=plot)
    error = np.abs(predicted - simulated)
    perc = 100 * error / simulated
    print(f"##### t={t} #####")
    print("predicted:\n", np.round(predicted, decimals=4))
    print("simulated:\n", np.round(simulated, decimals=4))
    print("absolue error:\n", np.round(error, decimals=4))
    print("percentage error:\n", np.round(perc, decimals=4))
    print("\n")
    return Stats

def check_values():
    W = np.repeat(np.array([[0.5, 0, 0.5], [0.8, 0.1, 0.1], [0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.3, 0.6, 0.1]]), repeats=(1, 1, 20, 1, 1), axis=0) # matches the overleaf doc
    S = Strategy(W, type=STRATEGYTYPE.FULL)
    ElfStats = ElfGameStats(S)

    # rounding errors means you need this
    assert round(ElfStats.V(1) - 32 / 3, 10) == 0
    assert round(ElfStats.V_A(23) - 64 / 3, 10) == 0
    assert round(ElfStats.V_A(22) - 568 / 15, 10) == 0

    print(ElfStats.caches)


def checker(thingyString):
    array = np.array([np.float64(i) for i in thingyString.replace('[', '').replace(']', '').split(', ')])
    print(array)
    S = Strategy(array, type=STRATEGYTYPE.IMPLIED)
    print(S)

def main():
    # W = np.repeat(np.array([[0.5, 0.5, 0], [0.8, 0.2, 0], [0, 1, 0], [0.2, 0.5, 0.3], [0, 0.6, 0.4]]), repeats=(1, 1, 20, 1, 1), axis=0) # matches the overleaf doc
    W = np.repeat(np.array([[0, 1, 0], [0, 0, 1]]), repeats=(22, 2), axis=0) # matches the overleaf doc
    S = Strategy(W, type=STRATEGYTYPE.FULL)
    # W = [np.full(fill_value=0.8, shape=(16)), np.tile(np.array([0, 0.6]), (8,1))]
    # S = Strategy(W, type=STRATEGYTYPE.IMPLIED)
    # print(S)
    E = ElfGameStats(S)

    print(12 * E.V_A(0), 12**2 * E.R_A(0))

    # for t in range(23, -1, -1):
    #     test_t(S, t=t, elfLoss=True, elfCount=1)
    test_t(S,t=0,n=100_000, plot=True)
    
    plt.show()

def vartesting():
    W = np.repeat(np.array([[0.2, 0.8, 0], [0.4, 0.6, 0], [0.5, 0.1, 0.4], [0.2, 0.3, 0.5], [0.3, 0.3, 0.4]]), repeats=(1, 1, 20, 1, 1), axis=0) # matches the overleaf doc
    S = Strategy(W, type=STRATEGYTYPE.FULL)
    E = ElfGameStats(S)
    test_t(S, t=21, elfLoss=True, elfCount=1, n=100_000)
    # main()
    VARD21 = 968/9
    ED21 = 59/3
    ED21SQ = 1483/3
    EM21 = 13/15
    EM21SQ = 59/75
    VARD22 = 1922/9
    ED22 = 68/3
    ED22SQ = 2182/3
    EM22 = 5/6
    EM22SQ = 3/4
    ED23 = 61/3
    ED23SQ = 1691/3
    VARM22 = EM22SQ - EM22**2
    VARM21 = EM21SQ - EM21**2

    ED22C=E.V_A(22)

    a21, b21 = 55,-28
    a22, b22 = 62,-29

    VARD22C = VARD22 + EM22SQ * ED23SQ - EM22**2 * ED23**2 + 2 * a22 * ED23 * VARM22
    print(VARD22C)

    VARD21C = VARD21 + EM21SQ * (VARD22C+ED22C**2) - EM21**2 * ED22C**2 + 2 * a21 * ED22C * VARM21
    print(VARD21C)
    # print(f"t:{21}\n",VARD21, (VARD22C+ED22C**2), a21)

if __name__ == '__main__':
    main()
    #checker('[0.0026836119927549375, 0.00460298246930467, 9.798804908258489e-05, 0.0007009796282884185, 0.007116068453005664, 0.0033817372801654647, 0.0011128869216209091, 0.002231874386629833, 0.009273196619023054, 0.0014074035385497387, 0.0011748473594124225, 0.00013955072978929218, 0.0030508153245082737, 0.0010850591839891646, 0.0005579195637372604, 0.0002064159956647287, 0.9975013445750811, 0.9996244465740849, 0.9988043036619813, 0.9996494509091487, 0.977715528143953, 0.9995935977664655, 0.40192578441839244, 0.9994928889840242, 0.003464050139814395, 0.9999342460546202, 0.0006208409202767941, 0.060862290648657225, 0.0006345116565247687, 0.00025907529201493583, 0.00020763495017887636, 0.0011375309606806467]')