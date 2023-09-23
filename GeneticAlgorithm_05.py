import numpy as np
import matplotlib.pyplot as plt
import random
from deap import base, algorithms
from deap import creator
from deap import tools
from algelitism import eaSimpleElitism
from graph_show import show_graph

inf = 100
D = (
    (0, 3, 1, 3, inf, inf),
    (3, 0, 4, inf, inf, inf),
    (1, 4, 0, inf, 7, 5),
    (3, inf, inf, 0, inf, 2),
    (inf, inf, 7, inf, 0, 4),
    (inf, inf, 5, 2, 4, 0)
)

startV = 0
LENGTH_D = len(D)
LENGTH_CHROM = len(D) * len(D[0])
POPULATION_SIZE = 500
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 30
HALL_OF_FAME_SIZE = 1

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("randomOrder", random.sample, range(LENGTH_D), LENGTH_D)
toolbox.register(
    "individualCreator",
    tools.initRepeat,
    creator.Individual,
    toolbox.randomOrder,
    LENGTH_D
)
toolbox.register(
    "populationCreator",
    tools.initRepeat,
    list,
    toolbox.individualCreator
)

population = toolbox.populationCreator(n=POPULATION_SIZE)


def dikstryFitness(individual):
    s = 0

    for n, path in enumerate(individual):
        path = path[:path.index(n)+1]

        si = startV

        for j in path:
            s += D[si][j]
            si = j

    return s,


def cxOrdered(ind1, ind2):
    for p1, p2 in zip(ind1, ind2):
        tools.cxOrdered(p1, p2)

    return ind1, ind2


def mutShuffleIndexes(individual, indpb):
    for ind in individual:
        tools.mutShuffleIndexes(ind, indpb)

    return individual,


toolbox.register("evaluate", dikstryFitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", cxOrdered)
toolbox.register("mutate", mutShuffleIndexes, indpb=1.0/LENGTH_CHROM/10)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)


def show(ax):
    ax.clear()
    show_graph(ax, hof.items[0])

    plt.draw()
    plt.gcf().canvas.flush_events()


plt.ion()
fig, ax = plt.subplots()

population, logbook = eaSimpleElitism(
    population,
    toolbox,
    cxpb=P_CROSSOVER / LENGTH_D,
    mutpb=P_MUTATION / LENGTH_D,
    ngen=MAX_GENERATIONS,
    halloffame=hof,
    stats=stats,
    callback=(show, (ax,)),
    verbose=True
)

maxFitnessValues, meanFitnessValues = logbook.select("min", "avg")

best = hof.items[0]

print(best)

plt.ioff()
plt.show()
