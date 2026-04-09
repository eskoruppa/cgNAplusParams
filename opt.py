#!/usr/bin/env python3

import sys, glob, os
import numpy as np
import random
import time
import multiprocessing
from deap import base, creator, tools

from cgnaplusparams import cgnaplus2rbp, rbp_conf
from cgnaplusparams import visualize_chimerax
from cgnaplusparams import curvature, distance

nbp = 340
TARGET_DISTANCE = 1.0
TARGET_CURVATURE = 0.07
base_fn = 'Distance/test'

NTERM = 100
POP_SIZE = 1000
NGEN = 2000
CXPB = 0.5
MUTPB = 0.75
NHOF = 3
TOURSIZE = 7
INDPB = 0.1
MAX_INDPB = 0.3
STAGNATION_THRESHOLD = 10
MUTATION_MULTIPLIER = 1.5
POP_BOOST_THRESHOLD = 100
POP_BOOST = 0.1

current_indpb = INDPB

BASE_MAPPING = ['A', 'C', 'G', 'T']

# DEAP setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("attr_int", random.randint, 0, 3)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=nbp)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# cost function
def evaluate(individual):
    seq = ''.join(BASE_MAPPING[int(gene)] for gene in individual)
    result = cgnaplus2rbp(seq, include_stiffness=False)
#    kappa = curvature(base_fn, seq, shape_params=result["gs"], cg=1)
#    cost = abs((kappa - TARGET_CURVATURE)/TARGET_CURVATURE)
    dist = distance(base_fn, seq, shape_params=result["gs"], cg=1)
    cost = abs((dist - TARGET_DISTANCE)/TARGET_DISTANCE)
    return (cost,)

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=TOURSIZE)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=INDPB)

def main():
    # create multiprocessing pool
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    toolbox.register("map", pool.map)

#    random.seed(42)
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(NHOF)
    
    no_improvement = 0
    best_cost_so_far = float('inf')

    t1 = time.time()
    
    for gen in range(NGEN):
        offspring = toolbox.select(pop, POP_SIZE)
        offspring = list(map(toolbox.clone, offspring))
        
        # crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        # mutation        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # replace worst population
        if no_improvement >= POP_BOOST_THRESHOLD:
            print(f"Boosting population diversity by {POP_BOOST*100}% (gen {gen+1}, stagnation {no_improvement})")
            num_to_replace = int(POP_SIZE * POP_BOOST)
            sorted_pop = sorted(offspring, key=lambda x: x.fitness.values[0])
            worst_indices = [i for i, ind in enumerate(offspring) if ind in sorted_pop[:num_to_replace]]
            
            for idx in worst_indices:
                offspring[idx] = toolbox.individual()
                offspring[idx].fitness.values = toolbox.evaluate(offspring[idx])
        
        pop[:] = offspring
        hof.update(pop)
        
        current_best = hof[0].fitness.values[0]
        if current_best < best_cost_so_far - 1e-6:
            best_cost_so_far = current_best
            no_improvement = 0
            current_indpb = INDPB
        else:
            no_improvement += 1

            # boost individual mutation rate
            if no_improvement >= STAGNATION_THRESHOLD:
                current_indpb = min(current_indpb * MUTATION_MULTIPLIER, MAX_INDPB)
                toolbox.unregister("mutate")
                toolbox.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=current_indpb)
                print(f"Increased mutation rate indpb={current_indpb:.3f} (gen {gen+1}, stagnation {no_improvement})")

        print(f"Gen {gen+1}: Best cost={current_best:.6f}, No imp={no_improvement}")
        
        if no_improvement >= NTERM:
            print(f"Early stop at gen {gen+1}")
            break
    
    t2 = time.time()
    
    pool.close()
    pool.join()
    
    final_pop_costs = [ind.fitness.values[0] for ind in pop]
    hof_costs = [ind.fitness.values[0] for ind in hof]
    final_costs = final_pop_costs + hof_costs
    print("Final costs (top 10 best):", sorted(final_costs)[:10])
    
    best_ind = hof[0]
    seq = ''.join(BASE_MAPPING[int(gene)] for gene in best_ind)
    best_cost = best_ind.fitness.values[0]

    result = cgnaplus2rbp(seq, include_stiffness=False)
    kappa = curvature(base_fn, seq, shape_params=result["gs"], cg=1)
    dist = distance(base_fn, seq, shape_params=result["gs"], cg=1)

    print(f"\nBest: {seq}")
    print(f"kappa = {kappa:.6f}")
    print(f"distance = {dist:.6f}")
    print(f"cost = {best_cost:.6f}")

    conf = rbp_conf(result["gs"])
    visualize_chimerax(base_fn, seq, shape_params=result["gs"], cg=1)
    
    print(f"Time taken: {t2 - t1:.5f} seconds total")

if __name__ == "__main__":
    main()
