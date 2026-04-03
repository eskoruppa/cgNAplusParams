#!/usr/bin/env python3

import sys, glob, os
import numpy as np
import random
import time
import multiprocessing
from deap import base, creator, tools  # No algorithms needed

from cgnaplusparams import cgnaplus2rbp, rbp_conf
from cgnaplusparams import visualize_chimerax
from cgnaplusparams import curvature

nbp = 147
TARGET_CURVATURE = 0.08
base_fn = 'Curvature/test'

NTERM = 50
POP_SIZE = 1000
NGEN = 1000
CXPB = 0.5
MUTPB = 0.75
NHOF = 3
TOURSIZE = 7
INDPB = 0.08
MAX_INDPB = 0.3
STAGNATION_THRESHOLD = 20
MUTATION_MULTIPLIER = 1.5
POP_BOOST_THRESHOLD = 30
POP_BOOST = 0.1

current_indpb = INDPB

BASE_MAPPING = ['A', 'C', 'G', 'T']

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_int", random.randint, 0, 3)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=nbp)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    seq = ''.join(BASE_MAPPING[int(gene)] for gene in individual)
    result = cgnaplus2rbp(seq, include_stiffness=False)
    kappa = curvature(base_fn, seq, shape_params=result["gs"], cg=1)
    fitness = 1.0 / (1 + abs(kappa - TARGET_CURVATURE))
    return (fitness,)

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=TOURSIZE)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=INDPB)

def main():
    # Create multiprocessing pool and tell DEAP to use it
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    toolbox.register("map", pool.map)

#    random.seed(42)
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(NHOF)
    
    no_improvement = 0
    best_fitness_so_far = 0

    t1 = time.time()
    
    for gen in range(NGEN):
        offspring = toolbox.select(pop, POP_SIZE)
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Parallel evaluation here
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)  # uses pool.map
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if no_improvement >= POP_BOOST_THRESHOLD:
            print(f"Boosting population diversity by {POP_BOOST*100}% (gen {gen+1}, stagnation {no_improvement})")
            # Sort by fitness, replace worst POP_BOOST
            num_to_replace = int(POP_SIZE * POP_BOOST)
            sorted_pop = sorted(offspring, key=lambda x: x.fitness.values[0])
            worst_indices = [i for i, ind in enumerate(offspring) if ind in sorted_pop[:num_to_replace]]
            
            # Replace worst with fresh random + IMMEDIATELY evaluate
            for idx in worst_indices:
                offspring[idx] = toolbox.individual()
                offspring[idx].fitness.values = toolbox.evaluate(offspring[idx])  # EVALUATED!
        
        pop[:] = offspring
        hof.update(pop)
        
        current_best = hof[0].fitness.values[0]
        if current_best > best_fitness_so_far + 1e-6:
            best_fitness_so_far = current_best
            no_improvement = 0
            current_indpb = INDPB
        else:
            no_improvement += 1

            # boost individual mutation rate
            if no_improvement >= STAGNATION_THRESHOLD:
                current_indpb = min(current_indpb * 1.5, MAX_INDPB)
                # Re-register with new rate
                toolbox.unregister("mutate")
                toolbox.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=current_indpb)
                print(f"Increased mutation rate indpb={current_indpb:.3f} (gen {gen+1}, stagnation {no_improvement})")

        print(f"Gen {gen+1}: Best={current_best:.4f}, No imp={no_improvement}")
        
        if no_improvement >= NTERM:
            print(f"Early stop at gen {gen+1}")
            break
    
    t2 = time.time()
    
    # Clean up pool
    pool.close()
    pool.join()
    
    # Results
    final_pop_fitness = [ind.fitness.values[0] for ind in pop]  # Current population
    hof_fitness = [ind.fitness.values[0] for ind in hof]
    final_fitness = final_pop_fitness + hof_fitness
    print("Final fitness (top 10):", sorted(final_fitness,reverse=True)[:10])
    
    best_ind = hof[0]
    seq = ''.join(BASE_MAPPING[int(gene)] for gene in best_ind)
    best_fitness = best_ind.fitness.values[0]

    result = cgnaplus2rbp(seq, include_stiffness=False)
    kappa = curvature(base_fn, seq, shape_params=result["gs"], cg=1)

    print("Best (seq, kappa, fit): ", seq, kappa, best_fitness)
    
    conf = rbp_conf(result["gs"])
    visualize_chimerax(base_fn, seq, shape_params=result["gs"], cg=1)
    
    print(f"Time taken: {t2 - t1:.5f} seconds total")

if __name__ == "__main__":
    main()
