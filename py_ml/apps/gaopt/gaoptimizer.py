from apps.app import App
from modules.basics.common.logger import *
from apps.gaopt.gaoptconf import GaOptConfig
import modules.basics.common.utils as utils
import os 
import sys
import copy 
import random 
import numpy as np 
from deap import base
from deap import creator
from deap import tools
from time import sleep
from numpy import isfinite
READ_INTERVAL = 5 # 5 sec
WINNER_FOLDER = "historical_winner_" + str(os.getpid())
WKFOLDER = "workground_" + str(os.getpid())
JOBFOLDER_PREFIX="job_"
class GaOptimizer(App):
    def __init__(self):
        super(GaOptimizer,self).__init__()
        self.config = GaOptConfig()
        self.cxpb = self.config.getCrossProb()
        self.mupb = self.config.getMutateProb()
        self.bestFitness = None 
        self.WORST = None
        if self.config.getObjectiveType() == 0:
            self.WORST = 1.e200
        if self.config.getObjectiveType() == 1:
            self.WORST = 0.

        os.mkdir(WINNER_FOLDER)
        return
    
    @classmethod
    def getInstance(cls):
        return cls()
    
    def prepare(self):
        return
    
    def initHyperParams(self,pcls,lb,ub):
        prop = []
        for i in range(len(lb)):
            if type(lb[i]) is int:
                p = np.random.randint(lb[i],ub[i])
            if type(lb[i]) is float:
                p = np.random.uniform(lb[i],ub[i])
            prop.append(p)
            
        part = pcls(prop)
        return part
    
    def execute(self):
        npm = self.config.getNumParams()
        ub = self.config.getUpperBounds()
        lb = self.config.getLowerBounds()
        if len(ub) != npm or len(lb) != npm:
            Log(LOG_FATAL) <<  "Length of bounds inconsistent with num_params"
            
        if self.config.getObjectiveType() == 0: # minimization
            creator.create("FitnessMin",base.Fitness,weights=(-1.0,))
            creator.create("Individual",list,fitness=creator.FitnessMin)
            self.getBest = min
        elif self.config.getObjectiveType() == 1: # maximization
            creator.create("FitnessMax",base.Fitness,weights=(1.0,))
            creator.create("Individual",list,fitness=creator.FitnessMax)
            self.getBest = max
        else:
            Log(LOG_FATAL) << "Wrong type of objective: %s" % self.config.getObjectiveType()
            
        self.toolbox = base.Toolbox()
        
        self.toolbox.register("params", random.randint,min(lb),max(ub))
#         self.toolbox.register("individual",tools.initRepeat,creator.Individual,
#                               self.toolbox.params,npm)
        self.toolbox.register("individual",self.initHyperParams,creator.Individual,
                              lb=lb, ub=ub)
        self.toolbox.register("population",tools.initRepeat,list,self.toolbox.individual)
        
        # Operators
        #self.toolbox.register("evaluate",function)
        self.toolbox.register("mate",tools.cxTwoPoint)
        self.toolbox.register("mutate",tools.mutUniformInt,low=lb,up=ub,
                              indpb=self.config.getIndProb())

        self.toolbox.register("select",tools.selTournament,
                              tournsize=self.config.getTournamentSize())
        ###########################################################
        pop = self.toolbox.population(n = self.config.getPopulationSize())
        
        fitnesses = self.evaluatePopulation(pop)
        
        for ind,fit in zip(pop,fitnesses):
            ind.fitness.values = fit,
        fits = [ind.fitness.values[0] for ind in pop]
        bestFit,idx = self.getBest((val,idx) for (idx,val) in enumerate(fits))
        self.bestFitness = bestFit 
        self.winner = pop[idx]
        
        Log(LOG_INFO) << "Best of Adams and Eves: %f" % self.bestFitness
        
        g = 0
        
        while g < self.config.getNumGenerations():
            g+=1
            Log(LOG_INFO) << "-- Generation %i --" % g
            
            offspring = self.toolbox.select(pop,len(pop))
            offspring = list(map(self.toolbox.clone,offspring))
            # crossover
            for child1,child2 in zip(offspring[::2],offspring[1::2]):
                if random.random() < self.cxpb:
                    self.toolbox.mate(child1,child2)
                    if child1 in pop:
                        i1 = pop.index(child1)
                        o1 = offspring.index(child1) 
                        Log(LOG_DEBUG) << "Found child1 in parents: %s" % pop[i1]
                        offspring[o1] = pop[i1]
                    else:
                        Log(LOG_INFO) << "New species: %s" % str(child1)
                        del child1.fitness.values
                    
                    if child2 in pop:
                        i2 = pop.index(child2)
                        o2 = offspring.index(child2)
                        Log(LOG_DEBUG) << "Found child2 in parents: %s" % pop[i2]
                        offspring[o2] = pop[i2]
                    else:
                        Log(LOG_INFO) << "New species: %s" % str(child2)
                        del child2.fitness.values
    
            # mutate
            for mutant in offspring:
                if random.random() < self.mupb:
                    self.toolbox.mutate(mutant)
                    if mutant in pop:
                        im = pop.index(mutant)
                        om = offspring.index(mutant)
                        offspring[om] = pop[im]
                    else:
                        Log(LOG_INFO) << "New species: %s" % str(mutant)
                        del mutant.fitness.values
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            fitnesses = self.evaluatePopulation(invalid_ind)
            for ind,fit in zip(invalid_ind,fitnesses):
                ind.fitness.values = fit,
            
            pop[:] = offspring
#             utils.tabulateList(pop)
            fits = [ind.fitness.values[0] for ind in pop]
            bestFit,idx = self.getBest((val,idx) for (idx,val) in enumerate(fits))
            Log(LOG_INFO) << "Best fitness of generation: %s" % bestFit
            Log(LOG_INFO) << "Winner of generation: %s" % str(pop[idx])

            if (self.config.getObjectiveType() == 0 and bestFit < self.bestFitness) \
                or (self.config.getObjectiveType() == 1 and bestFit > self.bestFitness):

                self.bestFitness = bestFit
                self.winner = pop[idx]
                src = WKFOLDER+"/" + JOBFOLDER_PREFIX + "_".join(str(w) for w in self.winner)
                dst = WINNER_FOLDER + "/"
                cmd = "cp -rf " + src + "/* "  + dst
                os.system(cmd)

            Log(LOG_INFO) << "Historical best fitness: %s " % self.bestFitness
            Log(LOG_INFO) << "Historical best kid: %s" % str(self.winner)

        return
    
    def finish(self):
        return
    
    def writeToYaml(self,individual,ym_modifier,yaml_template):
        if not os.path.isfile(ym_modifier):
            Log(LOG_FATAL) << "File not found: " + ym_modifier
        
        ym = ym_modifier 
        ymer,_ = os.path.splitext(ym)
        newpath = os.getcwd()+'/'+os.path.dirname(ym)
        sys.path.append(newpath)
        user_defined_module = __import__(os.path.basename(ymer))
        user_defined_module.writeToYaml(individual,yaml_template)
        
        return
    
    def checkBounds(self,pop):
        ub = self.config.getUpperBounds()
        lb = self.config.getLowerBounds()
        sb = self.config.getSumBound()

        numJobs=0
        for ind in pop:
            job_index = "_".join(str(i) for i in ind)
            for i,u,l in zip(ind,ub,lb):
                if i > u or i < l:
                    self.allResults[job_index] = self.WORST
                    numJobs+=1
                    break
            if sum(ind)  > sb:
                self.allResults[job_index] = self.WORST
                numJobs+=1
        Log(LOG_INFO) << "Out-of-bound samples: %d" % numJobs
        return  numJobs

    def evaluatePopulation(self,pop):
        self.allResults = {}
        self.checkBounds(pop)

        workFolder = WKFOLDER
        
        if not os.path.isdir(workFolder):
            os.mkdir(workFolder)
            
        numJobs = self.createJobs(workFolder, pop)
        Log(LOG_INFO) << "CWD after creating jobs: " + os.getcwd()
        self.runJobs(workFolder)
        Log(LOG_INFO) << "CWD after submitting jobs: " + os.getcwd()
        
        while True:
            isFinish = self.readJobs(workFolder, numJobs)
            if isFinish:
                break
            sleep(READ_INTERVAL)

        fitnesses=[]
        for ind in pop:
            job_index = "_".join(str(i) for i in ind)
            fitnesses.append(self.allResults[job_index])

        return fitnesses
    
    def createJobs(self,workFolder,pop):
        os.chdir(workFolder)
        root = os.getcwd()
        os.system("rm -rf *")

        numJobs=0
        for ind in pop:
            job_index = "_".join(str(i) for i in ind)
            if self.allResults.get(job_index) is not None:
                continue # skip if already has result
            
            wf = JOBFOLDER_PREFIX + job_index
            
            if os.path.isdir(wf):
                continue # skip if folder exists
            
            numJobs+=1
            os.mkdir(wf)
            os.chdir(wf)
            for f in self.config.getDataFiles():
                src = "../../" + f 
                if not os.path.isfile(src):
                    Log(LOG_FATAL) << "File not found: " + f
                os.system("ln -sf " + src + " .")
                
            yaml_template = "../../" + self.config.getYamlTemplate()
            ym_modifier = "../../"+ self.config.getYamlModifier()
            self.writeToYaml(ind,ym_modifier,yaml_template)
            
            cmd = "cp ../../" + self.config.getSGEJobFile() + " ."
            os.system(cmd)
            
            # rename job name with pid
            cmd = "sed -i \'s/athena_job/pid_" + str(os.getpid()) + "/\' " + self.config.getSGEJobFile()
            os.system(cmd)    
            
            os.chdir(root)
        
        os.chdir("../")
        return numJobs
    
    def runJobs(self,workFolder):
        os.chdir(workFolder)
        root = os.getcwd()
        allfiles = [x for x in os.listdir(".") if os.path.isdir(x)]
        for fd in allfiles:
            os.chdir(fd)
            cmd = "qsub -S /bin/bash " + self.config.getSGEJobFile()
            os.system(cmd)
            
            os.chdir(root)
            
        os.chdir("../")
        
        Log(LOG_INFO) << "All jobs are submitted to SGE"
        return
    
    def readJobs(self,workFolder,numJobs):
        nFinish = 0
        os.chdir(workFolder)
        root = os.getcwd()
        
        allfolders = [ x for x in os.listdir(".") if os.path.isdir(x)]
        for fd in allfolders:
            key = fd[len(JOBFOLDER_PREFIX):]
            Log(LOG_DEBUG) << "Checking %s" % key
            
            if self.allResults.get(key) is not None:
                nFinish+=1
                continue
            
            os.chdir(fd)
            logs = [ x for x in os.listdir(".") if x.startswith("log_")]
            
            res,n = self.parseLogs(logs)
            Log(LOG_DEBUG) << "Finished logs: %d" % n
            
            if n == self.config.getNItersPerKid():
                self.allResults[key] = res
                nFinish+=1
            else:
                Log(LOG_INFO) << "Not finished: %s" % key
                
            os.chdir(root)
            
        os.chdir("../")
        
        Log(LOG_INFO) << "CWD after reading jobs: %s" % os.getcwd()
        
        Log(LOG_INFO) << "%d out of %d finished" % (nFinish,numJobs)
        
        return nFinish == numJobs
    
    def parseLogs(self,logs):
        kws = self.config.getResultKeywords()
        allres = []
        for log in logs:
            res = []
            with open(log,'r') as f:
                for line in f:
                    line = line.strip()
                    for kw in kws:
                        if line.find(kw) >= 0:
                            idx=line.rfind(":")
                            res.append(float(line[idx+1:]))
            if len(res) == len(kws):
                allres.append(res)
                
        if len(allres) == 0:
            return None,0
        avg = np.mean([x[0] for x in allres])
        
        return avg,len(allres)
        
    
        
        
        
    