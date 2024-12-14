import numpy as np

# Step 1: Define the Problem
def objective_function(x, y):
    return x**2 + y**2  # A simple optimization problem to minimize

# Step 2: Initialize Parameters
grid_size = (10, 10)  # 10x10 grid of cells
num_iterations = 100
neighborhood_size = 1  # Considering immediate neighbors
mutation_rate = 0.1  # Probability of random perturbation

# Step 3: Initialize Population
np.random.seed(42)  # For reproducibility
population = np.random.uniform(-10, 10, size=(grid_size[0], grid_size[1], 2))  # Random positions (x, y)
fitness = np.full(grid_size, float('inf'))  # Initialize fitness values

# Step 4: Evaluate Fitness
def evaluate_population(pop):
    return np.array([[objective_function(ind[0], ind[1]) for ind in row] for row in pop])

fitness = evaluate_population(population)

# Step 5: Update States
def update_states(pop, fit):
    new_population = np.copy(pop)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Get neighbors' indices
            neighbors = [
                (i + dx, j + dy)
                for dx in range(-neighborhood_size, neighborhood_size + 1)
                for dy in range(-neighborhood_size, neighborhood_size + 1)
                if (0 <= i + dx < grid_size[0]) and (0 <= j + dy < grid_size[1]) and not (dx == 0 and dy == 0)
            ]
            # Choose the best neighbor
            best_neighbor = min(neighbors, key=lambda n: fit[n[0], n[1]])
            # Update the cell towards the best neighbor
            new_population[i, j] = pop[best_neighbor] + mutation_rate * np.random.uniform(-1, 1, 2)
    return new_population

# Step 6: Iterate
for iteration in range(num_iterations):
    population = update_states(population, fitness)
    fitness = evaluate_population(population)

# Step 7: Output the Best Solution
best_index = np.unravel_index(np.argmin(fitness), fitness.shape)
best_solution = population[best_index]
best_fitness = fitness[best_index]

print(f"Best Solution: {best_solution}, Fitness: {best_fitness}")
