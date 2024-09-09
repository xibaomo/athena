import pdb
import multiprocessing
import random
import numpy as np
import yaml
from deap import base, creator, tools, algorithms
import time

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
class Result(object):
    def __init__(self,x_,fun_):
        # pdb.set_trace()
        self.x=x_
        self.fun=fun_
def ga_minimize(objfunc, num_variables, bounds=[], is_parallel=True,population_size=200, num_generations=50, cross_prob=0.5,mutation_rate=0.2):
    # def evaluate(ind):
    #     return objfunc(ind),
    def create_bounded_float(low, up):
        # pdb.set_trace()
        return random.uniform(low, up)

    def bounded_mutate(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = create_bounded_float(bounds[i][0], bounds[i][1])

        return individual,

    # Custom crossover function that respects bounds
    def cx_bounded(ind1, ind2):
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
                # Enforce bounds after swapping
                ind1[i] = max(bounds[i][0], min(ind1[i], bounds[i][1]))
                ind2[i] = max(bounds[i][0], min(ind2[i], bounds[i][1]))
        return ind1, ind2

    # pdb.set_trace()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    n = len(bounds)
    if n>0 and n == num_variables:
        # for i in range(len(bounds)):
        #     toolbox.register(f"attr_float_{i}", create_bounded_float, bounds[i][0], bounds[i][1])
        # attr_floats = [getattr(toolbox, f"attr_float_{i}") for i in range(num_variables)]
        toolbox.register("attr_float", create_bounded_float)
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         [lambda: create_bounded_float(bounds[i][0], bounds[i][1]) for i in range(num_variables)])

        # toolbox.register("individual", tools.initCycle, creator.Individual, attr_floats, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    else:
        # toolbox.register("attribute", random.uniform, lb, ub)
        # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=num_variables)
        print("Error! Must give bounds for each variable in form of [[lb,ub],[lb,ub],...]")

    # toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objfunc)
    toolbox.register("mate", cx_bounded)
    # toolbox.register("mate", tools.cxBlend, alpha=cross_prob)
    # toolbox.register("mutate", tools.mutGaussian,mu=0,sigma=1,indpb=mutation_rate)
    toolbox.register("mutate",bounded_mutate,indpb=mutation_rate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    if is_parallel:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
    population = toolbox.population(n=population_size)

    result = algorithms.eaSimple(population, toolbox, cxpb=cross_prob, mutpb=mutation_rate, ngen=num_generations, verbose=True)

    if is_parallel:
        pool.close()
        pool.join()

    best_solution = tools.selBest(population, k=1)[0]
    best_fitness = best_solution.fitness.values[0]

    delattr(creator, "FitnessMin")
    delattr(creator, "Individual")

    res = Result(best_solution,best_fitness)
    return res

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
