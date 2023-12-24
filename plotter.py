import sys
# allow to import from files in this dir
sys.path.append('/optimisation')

from objective_functions import evaluateStrategyWeights

import numpy as np
import matplotlib.pyplot as plt

def loadandshow(fname):
    all = np.load(f'./plots/{fname}.npy')
    plot(all)

def plot(pareto_points):
    evaluated = np.apply_along_axis(evaluateStrategyWeights, 1, pareto_points)
    expected_value = evaluated[:,0]
    var = evaluated[:,1]
    std = np.sqrt(var)

    plt.xlabel('Standard Deviation')
    plt.ylabel('Expected Value')
    plt.scatter(std, expected_value, alpha=0.01)

def loadandshow_withlabels(fname):
    all = np.load(f'./plots/{fname}.npy')
    plot_with_labels(all)

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

if __name__ == '__main__':
    loadandshow()
