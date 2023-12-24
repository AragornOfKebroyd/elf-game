import sys
# allow to import from files in this dir
sys.path.append('/optimisation')

from strategy import Strategy, STRATEGYTYPE
from NSGAII import main as findParetoFront, INITYPE
from plotter import loadandshow_withlabels, loadandshow, plot_with_labels
from objective_functions import evaluateStrategyWeights

import numpy as np
import matplotlib.pyplot as plt
import time

plt.ion()

def slimdata(pareto_points):
    BINCOUNT = 100
    evaluated = np.apply_along_axis(evaluateStrategyWeights, 1, pareto_points)
    expected_value = evaluated[:,0]
    var = evaluated[:,1]

    edges = np.histogram_bin_edges(expected_value, bins=BINCOUNT)

    bins = np.digitize(expected_value, bins=edges)

    slimmed_pareto_points = np.zeros(shape=(0,32))
    exps = np.array([[]])
    vars = np.array([[]])

    with_bins = np.column_stack((bins, evaluated, pareto_points))
    sorted_indices = np.argsort(with_bins[:,1])
    with_bins = with_bins[sorted_indices]

    for bin_i in range(1,BINCOUNT+2):
        bin = with_bins[with_bins[:,0] == bin_i]
        if bin.size == 0: continue
        min = np.argmin(bin[:,2])
        minpoint = bin[min][1:]
        pareto_point = minpoint[2:]
        exp = minpoint[0]
        var = minpoint[1]
        slimmed_pareto_points = np.append(slimmed_pareto_points, [pareto_point], axis=0)
        exps = np.append(exps, exp)
        vars = np.append(vars, var)

    return slimmed_pareto_points

def find_pareto_points(LOWER,UPPER,STEP, GENS):
    AIMS = np.arange(LOWER,UPPER,STEP) # 3800, 3900, ... 4400, 4500 # 3800 is baseline all elves to FF every day is 3800

    pareto_points = np.zeros(shape=(0,32))
   
    for aim in AIMS:
        print("AIM:", aim)
        pareto_front = findParetoFront(wanted=aim, gens=GENS, init=INITYPE.RANDOM)
    
        pareto_points = np.append(pareto_points, pareto_front, axis=0)

    # Save
    timestr = time.strftime("%m%d-%H%M")
    np.save(f'./plots/pareto-points-{timestr}', pareto_points)

def slim_pareto(fname):
    pareto_points = np.load(f'./plots/{fname}.npy')
    slimmed = slimdata(pareto_points)
    plot_with_labels(slimmed)

    timestr = time.strftime("%m%d-%H%M")
    np.save(f'./plots/slimmed-points-{timestr}', slimmed)

def get_point(fname, i):
    pareto_points = np.load(f'./plots/{fname}.npy')
    return pareto_points[i]

def save_points(fname):
    loadandshow_withlabels(fname)
    plt.show()

    np.set_printoptions(suppress=True)
    while True:
        i = int(input('Input index:\n>>> '))
        strategy = get_point(fname, i)
        strat = Strategy(strategy, type=STRATEGYTYPE.IMPLIED)
        print(strat)
        yn = input('Save (Y/N):\n>>> ')
        if yn.lower() == 'y':
            filename = input('Enter filename')
            np.save(f'./saved_starts/{filename}', strategy)

def run():
    # will take a long time to run
    LOWER = 3600
    UPPER = 4600
    STEP = 100
    GENS = 200
    find_pareto_points(LOWER, UPPER, STEP, GENS)

class RUNMODE:
    RUN = 1
    LOADANDSHOW = 2
    SLIMDATA = 3
    SAVEPOINTS = 4

if __name__ == '__main__':
    RMODE = RUNMODE.RUN

    match RMODE:
        case RUNMODE.RUN:
            run()
        case RUNMODE.LOADANDSHOW:
            fname = 'pareto-points-1220-2137'
            loadandshow(fname)
        case RUNMODE.SLIMDATA:
            fname = 'pareto-points-1220-2137'
            slim_pareto(fname)
        case RUNMODE.SAVEPOINTS:
            fname = 'slimmed-points-1220-2204'
            save_points(fname)