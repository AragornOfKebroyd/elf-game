import numpy as np
import matplotlib.pyplot as plt
import sys
# allow to import from files in this dir
sys.path.append('/optimisation')
from strategy import evaluateStrategyWeights, Strategy, STRATEGYTYPE
from NSGAII import main as findParetoFront, INITYPE
import time

plt.ion() # interactive mode

def plot(pareto_points):
    evaluated = np.apply_along_axis(evaluateStrategyWeights, 1, pareto_points)
    expected_value = evaluated[:,0]
    var = evaluated[:,1]
    std = np.sqrt(var)
    # if trunc:
    #     upto = input(expected_value.shape)

    #     joined = np.column_stack((expected_value, std))
    #     joined = np.random.shuffle(joined)
    #     truncated = joined[:upto]
    #     expected_value = truncated[:,0]
    #     std = truncated[:,1]
    plt.xlabel('Standard Deviation')
    plt.ylabel('Expected Value')
    plt.scatter(std, expected_value, alpha=0.01)
    plt.show()
    input()

def slimdata(pareto_points):
    BINCOUNT = 100
    evaluated = np.apply_along_axis(evaluateStrategyWeights, 1, pareto_points)
    expected_value = evaluated[:,0]
    var = evaluated[:,1]
    std = np.sqrt(var)

    # plt.scatter(std, expected_value)
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

    #print(slimmed_pareto_points)
    #print(expected_val)
    #print(std)
    # stds = np.sqrt(vars)
    # plt.scatter(stds, exps, c="red")
    # plt.show()

    return slimmed_pareto_points

def find_pareto_points(LOWER,UPPER,STEP, GENS):
    AIMS = np.arange(LOWER,UPPER,STEP) # 3800, 3900, ... 4400, 4500 # 3800 is baseline all elves to FF every day is 3800

    pareto_points = np.zeros(shape=(0,32))
    # todo allow a way to get the best ones from this big front
    # todo round to nearest 12th for first 16 days, then some more rounding for others
    # check the eval is working correctly
    # run for a long time

    for aim in AIMS:
        print("AIM:", aim)
        pareto_front = findParetoFront(wanted=aim, gens=GENS, init=INITYPE.RANDOM)
    
        pareto_points = np.append(pareto_points, pareto_front, axis=0)

    # Save
    timestr = time.strftime("%m%d-%H%M")
    np.save(f'./plots/pareto-points-{timestr}', pareto_points)

    # plot(pareto_points)

def loadandshow(fname):
    pareto_points = np.load(f'./plots/{fname}.npy')
    plot(pareto_points)

def plot_with_labels(pareto_points):
    fig, ax = plt.subplots(1,1)

    evaluated = np.apply_along_axis(evaluateStrategyWeights, 1, pareto_points)
    expected_value = evaluated[:,0]
    var = evaluated[:,1]
    std = np.sqrt(var)

    plt.xlabel('Standard Deviation')
    plt.ylabel('Expected Value')    
    labels = [str(i) for i in range(len(std))]

    sc = plt.scatter(std, expected_value)

    annot = ax.annotate("", xy=(0,0), xytext=(5,5),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = " ".join([labels[n] for n in ind["ind"]])
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)
        
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    # plt.show()
    input()

def slimpareto(fname):
    pareto_points = np.load(f'./plots/{fname}.npy')
    slimmed = slimdata(pareto_points)
    plot_with_labels(slimmed)

    timestr = time.strftime("%m%d-%H%M")
    np.save(f'./plots/slimmed-points-{timestr}', slimmed)

def get_point(fname, i):
    pareto_points = np.load(f'./plots/{fname}.npy')
    return pareto_points[i]

def save_points(fname):
    pareto_points = np.load(f'./plots/{fname}.npy')
    plot_with_labels(pareto_points)
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

if __name__ == '__main__':
    # will take a long time to run
    LOWER = 3600
    UPPER = 4600
    STEP = 100
    GENS = 1000
    # find_pareto_points(LOWER, UPPER, STEP, GENS)
    fname = 'pareto-points-1220-2137'
    # loadandshow(fname)
    # slimpareto(fname)
    slimname = 'slimmed-points-1220-2204'
    # pareto_points = np.load(f'./plots/{slimname}.npy')
    # plot_with_labels(pareto_points)
    save_points(slimname)