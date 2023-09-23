import matplotlib.pyplot as plt
import numpy as np
import random
import time
from deap import base, algorithms
from deap import creator
from deap import tools

ONE_MAX_LENGTH = 100
POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 50

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def oneMaxFitness(individual):
    return sum(individual),


toolbox = base.Toolbox()

toolbox.register("zeroOrOne", random.randint, 0, 1)
toolbox.register(
    "individualCreator",
    tools.initRepeat,
    creator.Individual,
    toolbox.zeroOrOne,
    ONE_MAX_LENGTH
)
toolbox.register(
    "populationCreator",
    tools.initRepeat,
    list,
    toolbox.individualCreator
)

population = toolbox.populationCreator(n=POPULATION_SIZE)


def select(population, n_pop):
    return population


toolbox.register("evaluate", oneMaxFitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)
stats.register("values", np.array)

population, logbook = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=P_CROSSOVER,
    mutpb=P_MUTATION,
    ngen=MAX_GENERATIONS,
    stats=stats,
    verbose=False
)

maxFitnessValues, meanFitnessValues, vals = logbook.select(
    "max",
    "avg",
    "values"
)


plt.ion()
fig, ax = plt.subplots()

line, = ax.plot(vals[0], 'o', markersize=1)
ax.set_ylim(40, 110)

for v in vals:
    line.set_ydata(v)

    plt.draw()
    plt.gcf().canvas.flush_events()

    time.sleep(0.5)

plt.ioff()
plt.show()
