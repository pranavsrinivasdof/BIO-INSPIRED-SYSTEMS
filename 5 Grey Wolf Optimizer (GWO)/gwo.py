import numpy as np

# Grey Wolf Optimizer
def gwo(objective_function, num_wolves, dim, lb, ub, max_iter):
    """
    Grey Wolf Optimizer Algorithm

    Parameters:
    - objective_function: Function to minimize
    - num_wolves: Number of wolves in the population
    - dim: Number of variables in the optimization problem
    - lb: Lower bound of search space
    - ub: Upper bound of search space
    - max_iter: Maximum number of iterations

    Returns:
    - Best solution found (alpha position) and its fitness value
    """
    # Initialize the positions of wolves randomly within the search space
    wolves = np.random.uniform(lb, ub, (num_wolves, dim))

    # Initialize alpha, beta, and delta (best three solutions)
    alpha_pos = np.zeros(dim)
    beta_pos = np.zeros(dim)
    delta_pos = np.zeros(dim)
    alpha_score = float("inf")  # Best fitness value (minimization)
    beta_score = float("inf")
    delta_score = float("inf")

    # Main loop for iterations
    for t in range(max_iter):
        # Evaluate the fitness of each wolf
        for i in range(num_wolves):
            fitness = objective_function(wolves[i])

            # Update alpha, beta, and delta based on fitness
            if fitness < alpha_score:
                alpha_score, alpha_pos = fitness, wolves[i].copy()
            elif fitness < beta_score:
                beta_score, beta_pos = fitness, wolves[i].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, wolves[i].copy()

        # Update the positions of all wolves
        for i in range(num_wolves):
            for j in range(dim):
                # Compute coefficients
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2 * r1 - 1, 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - wolves[i][j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2 * r1 - 1, 2 * r2
                D_beta = abs(C2 * beta_pos[j] - wolves[i][j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2 * r1 - 1, 2 * r2
                D_delta = abs(C3 * delta_pos[j] - wolves[i][j])
                X3 = delta_pos[j] - A3 * D_delta

                # Update wolf position
                wolves[i][j] = (X1 + X2 + X3) / 3

            # Ensure wolves stay within bounds
            wolves[i] = np.clip(wolves[i], lb, ub)

    # Return the best solution found
    return alpha_pos, alpha_score

# Example usage
# Define an example objective function (sphere function)
def sphere_function(x):
    return np.sum(x ** 2)

# Parameters
num_wolves = 10       # Number of wolves
dim = 5               # Dimensions of the problem
lb = -10              # Lower bound
ub = 10               # Upper bound
max_iter = 100        # Number of iterations

# Run GWO
best_solution, best_value = gwo(sphere_function, num_wolves, dim, lb, ub, max_iter)
print("Best Solution:", best_solution)
print("Best Fitness Value:", best_value)
