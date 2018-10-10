'''
Created on Oct 10, 2018

@author: fxua
'''
from apps.app import App 
from modules.basics.common.logger import *
from apps.gaopt.gaoptconf import GaOptConfig
import os 
import copy 
import random 
import numpy as np 
from deap import base
from deap import creator
from deap import tools
class GaOptimizer(App):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(GaOptimizer,self).__init__()
        self.config = GaOptConfig()
        self.bestFitness = None
        self.getBest = None
        return
    
    def prepare(self):
        npm = self.config.getNumParams()
        ub  = self.config.getUpperBounds()
        lb  = self.config.getLowerBounds()
        if len(ub) != npm or len(lb) != npm:
            Log(LOG_FATAL) << "Length of bounds inconsistent with num_params"
            
        if self.config.getObjectiveType() == 0: #minimization
            creator.create("FitnessMin",base.Fitness,weights=(-1.0,))
            creator.create("Individual",list,fitness=creator.FitnessMin)
            self.getBest = min
        elif self.config.getObjectiveType() == 1: #maximization
            creator.create("FitnessMax",base.Fitness,weights=(1.0,))
            creator.create("Individual",list,fitness=creator.FitnessMax)
            self.getBest = max
        else:
            Log(LOG_FATAL) << "Wrong type of objective: %s" % self.config.getObjectiveType()
            
        self.toolbox = base.Toolbox()
        self.toolbox.register("params",random.randint,min(lb),max(ub))
        self.toolbox.register("individual",tools.initRepeat,
                              creator.individual,self.toolbox.params,npm)
        # population is defined as a list of individuals
        self.toolbox.register("population",tools.initRepeat,list,self.toolbox.individual)
        
        # Operators
#         self.toolbox.register("evaluate",function)
        self.toolbox.register("mate",tools.cxTwoPoint)
        self.toolbox.register("mutate",tools.mutUniformInt,low=min(lb),up=max(ub),
                              indpb=self.config.getIndProb())
        self.toolbox.register("select",tools.selTournament,tournsize=self.config.getTournamentSize())
        
            
        return
    
    def execute(self):
        pop = self.toolbox.population(n=self.config.getPopulationSize())
        fitnesses = self.evaluatePopulation(pop)
        for ind,fit in zip(pop,fitnesses):
            ind.fitness.values = fit
        fits = [ind.fitness.values[0] for ind in pop]
        self.bestFitness = self.getBest(pop)
        Log(LOG_INFO) << "Best of Adams and Eves: %i" % self.bestFitness
        g = 0
        
        while g < self.config.getNumGenerations():
            g+=1
            Log(LOG_INFO) << "-- Generation %i -- " % g
            offspring = self.toolbox.select(pop,len(pop))
            offspring = list(map(self.toolbox.clone,offspring))
            for child1,child2 in zip(offspring[::2],offspring[1::2]):
                if random.random() < self.config.getCrossProb():
                    self.toolbox.mate(child1,child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < self.config.getMutateProb():
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.evaluatePopulation(invalid_ind)
            for ind,fit in zip(invalid_ind,fitnesses):
                ind.fitness.values=fit
                
            pop[:] = offspring
            fits = [ind.fitness.values[0]  for ind in pop]
            bestFit,idx = self.getBest((val,idx) for (idx,val) in enumerate(fits))
            Log(LOG_INFO) << "Best fitness of generation: %s" % bestFit
            Log(LOG_INFO) << "Winner of generation: %s" % pop[idx]
            
            if self.config.getObjectiveType() == 0:
                if bestFit < self.bestFitness:
                    self.bestFitness = bestFit
                    self.winner = pop[idx]
            if self.config.getObjectiveType() == 1:
                if bestFit > self.bestFitness:
                    self.bestFitness = bestFit
                    self.winner = pop[idx]
            Log(LOG_INFO) << "Historical best fitness: %s" % self.bestFitness
            Log(LOG_INFO) << "Historical winner: %s" % self.winner
        return
    
    def finish(self):
        return
    
    def writeToYaml(self,individual):
        ym = self.config.getYamlModifier()
        if not os.path.isfile(ym):
            Log(LOG_FATAL) << "File not found: " + ym
        ymer = os.path.splitext(ym)
        user_defined_module = __import__(ymer)
        user_defined_module.writeToYaml(individual,self.config.getYamlTemplate())
        return
    
    def evaluatePopulation(self,pop):
        fitnesses = np.random.uniform(size=len(pop))
        return fitnesses
    
    