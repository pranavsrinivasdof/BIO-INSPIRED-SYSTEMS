import random

# Define the optimization problem: Example - minimize the function f(x) = x^2
def fitness_function(x):
    return x**2

# Step 1: Initialize Parameters
POPULATION_SIZE = 20
GENE_RANGE = (-10, 10)  # Genes represent values in the range [-10, 10]
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
GENERATIONS = 50

# Step 2: Initialize Population
def initialize_population(size, gene_range):
    return [random.uniform(*gene_range) for _ in range(size)]

# Step 3: Evaluate Fitness
def evaluate_fitness(population):
    return [fitness_function(individual) for individual in population]

# Step 4: Selection (Tournament Selection)
def select_parents(population, fitness):
    selected = []
    for _ in range(len(population)):
        i1, i2 = random.sample(range(len(population)), 2)
        selected.append(population[i1] if fitness[i1] < fitness[i2] else population[i2])
    return selected

# Step 5: Crossover (Single-point Crossover)
def crossover(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents) and random.random() < CROSSOVER_RATE:
            # Perform crossover
            point = random.random()
            child1 = parents[i] * point + parents[i + 1] * (1 - point)
            child2 = parents[i + 1] * point + parents[i] * (1 - point)
            offspring.extend([child1, child2])
        else:
            # No crossover
            offspring.extend([parents[i], parents[i + 1]])
    return offspring

# Step 6: Mutation
def mutate(population):
    for i in range(len(population)):
        if random.random() < MUTATION_RATE:
            population[i] += random.uniform(-1, 1)  # Small random change
    return population

# Step 7: Gene Expression (In this example, genes directly represent the solution)
# Not explicitly needed as our individuals are already directly usable in the fitness function.

# Main Evolution Loop
def gene_expression_algorithm():
    population = initialize_population(POPULATION_SIZE, GENE_RANGE)
    best_solution = None
    best_fitness = float('inf')

    for generation in range(GENERATIONS):
        fitness = evaluate_fitness(population)
        best_idx = fitness.index(min(fitness))

        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best_solution = population[best_idx]

        # Print progress
        print(f"Generation {generation + 1}: Best Solution = {best_solution}, Fitness = {best_fitness}")

        # Selection, Crossover, Mutation
        parents = select_parents(population, fitness)
        offspring = crossover(parents)
        population = mutate(offspring)

    return best_solution, best_fitness

# Run the Algorithm
best_solution, best_fitness = gene_expression_algorithm()
print(f"\nOptimal Solution Found: {best_solution} with Fitness = {best_fitness}")
