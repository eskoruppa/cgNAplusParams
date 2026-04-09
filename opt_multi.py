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

nbp = 360
TARGET_CURVATURE = 0.051
TARGET_DISTANCE = 2.0
base_fn = 'Multi/test'

NTERM = 100
POP_SIZE = 3000
NGEN = 2000
CXPB = 0.5
MUTPB = 0.75
NHOF = 3
INDPB = 0.1
MAX_INDPB = 0.3
STAGNATION_THRESHOLD = 10
MUTATION_MULTIPLIER = 1.5

current_indpb = INDPB

BASE_MAPPING = ['A', 'C', 'G', 'T']

# Multi-objective setup
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

toolbox.register("attr_int", random.randint, 0, 3)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=nbp)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    seq = ''.join(BASE_MAPPING[int(gene)] for gene in individual)
    result = cgnaplus2rbp(seq, include_stiffness=False)
    kappa = curvature(base_fn, seq, shape_params=result["gs"], cg=1)
    dist = distance(base_fn, seq, shape_params=result["gs"], cg=1)
    
    obj1 = abs((kappa - TARGET_CURVATURE)/TARGET_CURVATURE)
    obj2 = abs((dist - TARGET_DISTANCE)/TARGET_DISTANCE)
    
    return (obj1, obj2)

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selNSGA2)  # NSGA-II selection
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=current_indpb)

def main():
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=POP_SIZE)
    
    # Initial evaluation
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof = tools.HallOfFame(NHOF)
    pareto_front = tools.ParetoFront()
    
    no_improvement = 0
    best_cost_so_far = float('inf')
    t1 = time.time()
    
    for gen in range(NGEN):
        # SIMPLIEST NSGA-II: Select parents from current population
        offspring = toolbox.select(pop, POP_SIZE)
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover + mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # ELITIST: Select best from COMBINED parents + offspring pool
        pop[:] = toolbox.select(pop + offspring, POP_SIZE)
        
        hof.update(pop)
        pareto_front.update(pop)
        
        current_best1 = min(ind.fitness.values[0] for ind in pareto_front)
        current_best2 = min(ind.fitness.values[1] for ind in pareto_front)
        if current_best1 < best_cost_so_far - 1e-6:
            best_cost_so_far = current_best1
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
        print(f"Gen {gen+1}: Best cost: obj1={current_best1:.6f}, obj2={current_best2:.6f}, Pareto={len(pareto_front)}, No imp={no_improvement}")
        
        if no_improvement >= NTERM:
            print(f"Early stop at gen {gen+1}")
            break
    
    t2 = time.time()
    pool.close()
    pool.join()
    
    # Pareto front results
    print("\n=== PARETO FRONT ===")
    for i, ind in enumerate(pareto_front):
        obj1, obj2 = ind.fitness.values
        print(f"{i+1:2d}: obj1(kappa)={obj1:.6f}, obj2(dist)={obj2:.6f}")
    
    # Final population stats
    final_pop_costs = [ind.fitness.values[0] for ind in pop]
    hof_costs = [ind.fitness.values[0] for ind in hof]
    final_costs = final_pop_costs + hof_costs
    print("Final costs (top 10 best):", sorted(final_costs)[:10])
    
    # Best by kappa error (obj1)
    best_ind = min(hof, key=lambda ind: ind.fitness.values[0])
    seq = ''.join(BASE_MAPPING[int(gene)] for gene in best_ind)
    
    result = cgnaplus2rbp(seq, include_stiffness=False)
    kappa = curvature(base_fn, seq, shape_params=result["gs"], cg=1)
    dist = distance(base_fn, seq, shape_params=result["gs"], cg=1)
    
    print(f"\nBest by kappa: {seq}")
    print(f"  kappa={kappa:.6f} (target={TARGET_CURVATURE}, err={best_ind.fitness.values[0]:.6f})")
    print(f"  distance={dist:.6f}  (target={TARGET_DISTANCE}, err={best_ind.fitness.values[1]:.6f})")
    
    conf = rbp_conf(result["gs"])
    visualize_chimerax(base_fn, seq, shape_params=result["gs"], cg=1)
    
    print(f"\nTime taken: {t2 - t1:.5f} seconds total")

if __name__ == "__main__":
    main()
