import random

# Step 1: Define the Problem
# This is a simple mathematical function to optimize. Let's use the function f(x) = x^2.
def fitness_function(x):
    return x**2

# Step 2: Initialize Parameters
population_size = 10
mutation_rate = 0.01 # 1% chance of mutation
crossover_rate = 0.7 # 70% chances of mating
num_generations = 20 # toal generation to traverse
gene_length = 5  # binary representation of numbers (5 bits for simplicity)

# Step 3: Create Initial Population
# Generate a random initial population of binary strings (which represent numbers)
def create_initial_population(size, gene_length):
    population = []
    for _ in range(size):
        individual = [random.randint(0, 1) for _ in range(gene_length)]
        population.append(individual)
    return population

# Step 4: Evaluate Fitness
# Evaluate the fitness of each individual in the population (convert binary to decimal)
def evaluate_fitness(population):
    fitness_scores = []
    for individual in population:
        x = int("".join(map(str, individual)), 2)  # binary to decimal
        fitness_scores.append(fitness_function(x))
    return fitness_scores

# Step 5: Selection
# Select individuals based on their fitness (roulette wheel selection)
def selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        return random.choice(population)  # if no fitness, return random individual
    probabilities = [fitness / total_fitness for fitness in fitness_scores]
    return random.choices(population, weights=probabilities, k=2)

# Step 6: Crossover
# Perform crossover between two selected individuals
def crossover(parent1, parent2, rate):
    if random.random() < rate:
        point = random.randint(1, len(parent1) - 1)  # Crossover point
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    else:
        return parent1, parent2  # No crossover, return parents as is

# Step 7: Mutation
# Apply mutation to individuals
def mutation(individual, rate):
    for i in range(len(individual)):
        if random.random() < rate:
            individual[i] = 1 - individual[i]  # Flip the bit
    return individual

# Step 8: Iteration
# Iterate the process for a fixed number of generations
def genetic_algorithm():
    population = create_initial_population(population_size, gene_length)
    
    for generation in range(num_generations):
        fitness_scores = evaluate_fitness(population)
        
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population, fitness_scores)
            offspring1, offspring2 = crossover(parent1, parent2, crossover_rate)
            offspring1 = mutation(offspring1, mutation_rate)
            offspring2 = mutation(offspring2, mutation_rate)
            new_population.extend([offspring1, offspring2])
        
        population = new_population
        
        best_solution = max(population, key=lambda ind: fitness_function(int("".join(map(str, ind)), 2)))
        best_fitness = fitness_function(int("".join(map(str, best_solution)), 2))
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    # Step 9: Output the Best Solution
    return best_solution

best_solution = genetic_algorithm()
best_solution_decimal = int("".join(map(str, best_solution)), 2)
print(f"Best Solution: {best_solution} (Decimal: {best_solution_decimal})")
