import numpy as np
import matplotlib.pyplot as plt
import sys
# allow to import from files in this dir
sys.path.append('/optimisation')
from strategy import evaluateStrategyWeights

def test():
    pareto_front = np.load('./pareto_fronts/paretoFront-4187-20231218-225122.npy')

    evaluated = np.apply_along_axis(evaluateStrategyWeights, 1, pareto_front)

    expected_value = evaluated[:,0]
    var = evaluated[:,1]
    std = np.sqrt(var)
    plt.xlabel('Expected Value')
    plt.ylabel('Standard Deviation')
    plt.scatter(expected_value, std)
    plt.show()

def loadandshow():
    all = np.load('./plots/front1.npy')
    total_ex, total_std = all
    plt.xlabel('Standard Deviation')
    plt.ylabel('Expected Value')
    plt.scatter(total_std, total_ex)
    plt.show()

if __name__ == '__main__':
    loadandshow()
