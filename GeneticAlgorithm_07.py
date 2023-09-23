import numpy as np
import matplotlib.pyplot as plt
import random
import time
from deap import base, algorithms
from deap import creator
from deap import tools
from algelitism import eaSimpleElitism

LOW, UP = -5, 5
ETA = 20
LENGTH_CHROM = 2
POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.2
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 5

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def randomPoint(a, b):
    return [random.uniform(a, b), random.uniform(a, b)]


toolbox = base.Toolbox()
toolbox.register("randomPoint", randomPoint, LOW, UP)
toolbox.register(
    "individualCreator",
    tools.initIterate,
    creator.Individual,
    toolbox.randomPoint
)
toolbox.register(
    "populationCreator",
    tools.initRepeat,
    list,
    toolbox.individualCreator
)

population = toolbox.populationCreator(n=POPULATION_SIZE)


def himmelblau(individual):
    x, y = individual
    f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    return f,


toolbox.register("evaluate", himmelblau)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register(
    "mate",
    tools.cxSimulatedBinaryBounded,
    low=LOW,
    up=UP,
    eta=ETA
)
toolbox.register(
    "mutate",
    tools.mutPolynomialBounded,
    low=LOW,
    up=UP,
    eta=ETA,
    indpb=1.0 / LENGTH_CHROM
)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)


def show(ax, xgrid, ygrid, f):
    ptMins = [
        [3.0, 2.0],
        [-2.805118, 3.131312],
        [-3.779310, -3.283186],
        [3.584458, -1.848126]
    ]

    ax.clear()
    ax.contour(xgrid, ygrid, f)
    ax.scatter(*zip(*ptMins), marker='X', color='red', zorder=1)
    ax.scatter(*zip(*population), color='green', s=2, zorder=0)

    plt.draw()
    plt.gcf().canvas.flush_events()

    time.sleep(0.2)


x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xgrid, ygrid = np.meshgrid(x, y)

f_himmelbalu = (xgrid ** 2 + ygrid - 11) ** 2 + (xgrid + ygrid ** 2 - 7) ** 2

plt.ion()
fig, ax = plt.subplots()
fig.set_size_inches(5, 5)

ax.set_xlim(LOW - 3, UP + 3)
ax.set_ylim(LOW - 3, UP + 3)

population, logbook = eaSimpleElitism(
    population,
    toolbox,
    cxpb=P_CROSSOVER,
    mutpb=P_MUTATION,
    ngen=MAX_GENERATIONS,
    halloffame=hof,
    stats=stats,
    callback=(show, (ax, xgrid, ygrid, f_himmelbalu)),
    verbose=True
)

maxFitnessValues, meanFitnessValues = logbook.select("min", "avg")

best = hof.items[0]

print(best)

plt.ioff()
plt.show()

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Generation')
plt.ylabel('Maximum/Average fitness')
plt.title('Dependence of maximum and average fitness on generation')
plt.show()
