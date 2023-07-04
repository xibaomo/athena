import pdb
import multiprocessing
import random
import numpy as np
import yaml
from deap import base, creator, tools, algorithms

class GAMinConfig(object):
    def __init__(self, cf):
        self.yamlDict = yaml.load(open(cf), Loader=yaml.FullLoader)
        self.root = "GA_MINIMIZER"

    def getNumGenerations(self):
        return self.yamlDict[self.root]['NUM_GENERATION']
    def getPopulation(self):
        return self.yamlDict[self.root]['POPULATION']
    def getCrossProb(self):
        return self.yamlDict[self.root]['CROSS_PROB']
    def getMutateProb(self):
        return self.yamlDict[self.root]['MUTATE_PROB']
# Define the genetic algorithm
def ga_minimize(objfunc, params, num_variables, lb=0.,ub=1.,population_size=200, num_generations=50, cross_prob=0.5,mutation_rate=0.1):
    def evaluate(ind):
        return objfunc(ind,params),
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin,lowerbound=0.,upperbound=1.)
    # creator.create("Individual", list, fitness=creator.Float, lowerbound=0.0, upperbound=1.0)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.uniform, lb, ub/num_variables)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=num_variables)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=mutation_rate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    population = toolbox.population(n=population_size)

    # pdb.set_trace()
    best_perf = np.inf
    for generation in range(num_generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=cross_prob, mutpb=mutation_rate)

        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
            # pdb.set_trace()
            if ind.fitness.values[0] < best_perf:
                best_perf = ind.fitness.values[0]

        population = toolbox.select(offspring, k=population_size)

        if (generation+1) % 100 == 0:
            print("generation: {}, best result so far: {}".format(generation+1,best_perf))

    best_solution = tools.selBest(population, k=1)[0]
    best_fitness = best_solution.fitness.values[0]

    delattr(creator, "FitnessMin")
    delattr(creator, "Individual")
    return best_solution, best_fitness

if __name__ == "__main__":

    # Set the random seed for reproducibility
    random.seed(0)
    # Define the objective function to be optimized
    def objective_function(x,pms):
        s = (x[0]-pms[0])**2 + (x[1]-pms[1])**2 + (x[2]-pms[2])**2
        return s
    # Define the parameters for the genetic algorithm
    population_size = 200
    num_generations = 50
    num_variables = 3
    mutation_rate = 0.01

    # Run the genetic algorithm
    # best_solution, best_fitness = genetic_algorithm(population_size, num_generations, num_variables, mutation_rate)

    params = [0.2,0.4,0.9]
    best_solution,best_fitness = ga_minimize(objective_function,params,3,mutation_rate=0.1)
    # Print the best solution and its fitness value
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)
