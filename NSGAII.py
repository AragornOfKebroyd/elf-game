import sys
# allow to import from files in this dir
sys.path.append('/optimisation')

from objective_functions import evaluateStrategyWeights
from strategy import STRATEGYTYPE

import numpy as np
from deap import algorithms, base, creator, tools
import random
import time

class INITYPE:
    ZEROS = 0
    RANDOM = 1
    BIASEDRANDOM = 2

# Define the objective functions and constraints
def evalFunction(individual):
    ExpectedValue, Variance = evaluateStrategyWeights(individual, type=STRATEGYTYPE.IMPLIED, strict=False)
    Diff = abs(WANTED - ExpectedValue)
    return Diff, Variance

def feasible(individual):
    # Implement constraints for x17 + x18 <= 1, x19 + x20 <= 1, etc.
    constraints = []

    for i, val in enumerate(individual[16:-1]):
        if i % 2 == 1: continue
        constraints.append(val + individual[i+17] <= 1)

    if False in constraints:
        return False
    return True

def initialise():
    match INIT:
        case INITYPE.ZEROS:
            return 0
        case INITYPE.RANDOM:
            return random.random()
        case INITYPE.BIASEDRANDOM:
            return abs(random.random() - random.random())

def defineProblem():
    # Define the problem and individuals
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -10.0))  # Minimize the second objective, maximize the first
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    # Define the optimization problem
    toolbox = base.Toolbox()

    toolbox.register("individual", tools.initRepeat, creator.Individual, initialise, n=32)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20.0, indpb=1.0/32)
    toolbox.register("select", tools.selNSGA2)

    return toolbox

def main(wanted, init=INITYPE.BIASEDRANDOM, gens=200):
    global WANTED
    WANTED = wanted

    global INIT
    INIT = init

    toolbox = defineProblem()

    # Create the initial population
    population = toolbox.population(n=100)

    # Set up the statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", min)
    stats.register("max", max)
    stats.register("mean", np.mean, axis=0)

    # Create a Pareto front object to store the non-dominated feasible solutions
    pareto_front = tools.ParetoFront()

    # Define the evaluation function with storing the Pareto front
    def evaluate_with_pareto_front(individual):
        # Evaluate the objectives
        fitness_values = evalFunction(individual)
        
        # Check if the individual satisfies constraints
        if feasible(individual):
            # If constraints are satisfied, update the Pareto front
            individual.fitness.values = fitness_values
            pareto_front.update([individual])

        return fitness_values

    toolbox.register("evaluate", evaluate_with_pareto_front)
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 1e+10))

    # Run the optimization
    population, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.2, ngen=gens, stats=stats, halloffame=None)

    # Print the final population
    final_population = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    # print("Final Population:", final_population)

    # Access the optimal solution from the Pareto front
    # optimal_solution = final_population[0]

    # print("Optimal Solution:", optimal_solution)
    # print("Optimal Objectives:", evalFunction(optimal_solution))

    print(final_population, pareto_front)

    timestr = time.strftime("%Y%m%d-%H%M")
    np.save(f'./pareto_fronts/paretoFront-{timestr}-{WANTED}-{gens}', pareto_front)

    return pareto_front


if __name__ == '__main__':
    main()