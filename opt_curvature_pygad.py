#!/usr/bin/env python3

import sys,glob,os
import numpy as np
import pygad

from cgnaplusparams import cgnaplus2rbp, rbp_conf
from cgnaplusparams import visualize_chimerax
from cgnaplusparams import curvature
import time

nbp = 40
TARGET_CURVATURE = 0.07

base_fn = 'Curvature/test'

BASE_MAPPING = ['A', 'C', 'G', 'T']

def single_fitness(solution):

    seq = decode_solution(solution)
    result = cgnaplus2rbp(seq,include_stiffness=False)
    kappa = curvature(base_fn, seq, shape_params=result["gs"],cg=1)

    return 1.0 / (1 + abs(kappa - TARGET_CURVATURE))

def fitness_func(ga_instance, solution, solution_idx):

    # Decode GA solution -> sequence
    seq = decode_solution(solution)
    result = cgnaplus2rbp(seq,include_stiffness=False)
    kappa = curvature(base_fn, seq, shape_params=result["gs"],cg=1)
    return 1.0 / (1 + abs(kappa - TARGET_CURVATURE))

def decode_solution(solution):

    # Decode GA solution -> sequence
    seq_ints = np.rint(solution).astype(int)
    seq_ints = np.clip(seq_ints, 0, 3)
    seq = ''.join(BASE_MAPPING[i] for i in seq_ints)  # "ATCG...

    return seq

if __name__ == "__main__":

    ga = pygad.GA(
        fitness_func=fitness_func,
        num_genes=nbp,
        sol_per_pop=100,
        num_generations=200,
        num_parents_mating=40,
        gene_space=[0, 1, 2, 3],
    )

    t1 = time.time()

    ga.run()

    final_fitness = ga.last_generation_fitness
    print("Final population fitness:", final_fitness)

    solution, solution_fitness, solution_idx = ga.best_solution()
    seq = decode_solution(solution)
    print("Best:", seq, solution_fitness)

    result = cgnaplus2rbp(seq,include_stiffness=False)
    kappa = curvature(base_fn, seq, shape_params=result["gs"],cg=1)
    print("kappa: %g" % kappa)

    conf = rbp_conf(result["gs"])
    visualize_chimerax(base_fn, seq, shape_params=result["gs"],cg=1)

    t2 = time.time()

    print(f"Time taken: {t2 - t1:.5f} seconds total")
