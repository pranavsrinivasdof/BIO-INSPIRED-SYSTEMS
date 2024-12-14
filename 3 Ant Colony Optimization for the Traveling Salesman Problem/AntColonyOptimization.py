import numpy as np
import random
import math

# Define the distance between two cities
def euclidean_distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# TSP Problem Definition
class TSP:
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    self.distance_matrix[i][j] = euclidean_distance(cities[i], cities[j])

# ACO Algorithm for TSP
class ACO:
    def __init__(self, tsp, num_ants, alpha, beta, rho, iterations):
        self.tsp = tsp
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.iterations = iterations

        # Pheromone initialization
        self.pheromone_matrix = np.ones((self.tsp.num_cities, self.tsp.num_cities))  # Initial pheromone levels
        np.fill_diagonal(self.pheromone_matrix, 0)  # No pheromone on the diagonal

    def probability(self, i, j, visited):
        """Calculate the probability of moving from city i to city j"""
        if j in visited:
            return 0
        pheromone = self.pheromone_matrix[i][j] ** self.alpha
        heuristic = (1 / self.tsp.distance_matrix[i][j]) ** self.beta
        return pheromone * heuristic

    def choose_next_city(self, current_city, visited):
        """Choose the next city based on probabilities"""
        probabilities = []
        for next_city in range(self.tsp.num_cities):
            prob = self.probability(current_city, next_city, visited)
            probabilities.append(prob)
        total_prob = sum(probabilities)
        if total_prob == 0:
            return random.choice([i for i in range(self.tsp.num_cities) if i not in visited])
        probabilities = [p / total_prob for p in probabilities]
        return np.random.choice(range(self.tsp.num_cities), p=probabilities)

    def update_pheromones(self, ants_solutions):
        """Update pheromones based on the solutions of all ants"""
        # Pheromone evaporation
        self.pheromone_matrix *= (1 - self.rho)

        # Pheromone deposition based on each ant's solution
        for solution, length in ants_solutions:
            for i in range(len(solution) - 1):
                self.pheromone_matrix[solution[i]][solution[i+1]] += 1 / length
            # Return to the starting city
            self.pheromone_matrix[solution[-1]][solution[0]] += 1 / length

    def run(self):
        best_solution = None
        best_length = float('inf')

        for _ in range(self.iterations):
            ants_solutions = []
            # Each ant constructs a solution
            for _ in range(self.num_ants):
                visited = [random.randint(0, self.tsp.num_cities - 1)]  # Random starting city
                current_city = visited[0]
                while len(visited) < self.tsp.num_cities:
                    next_city = self.choose_next_city(current_city, visited)
                    visited.append(next_city)
                    current_city = next_city

                # Complete the cycle (return to start)
                visited.append(visited[0])
                # Calculate the total length of the tour
                length = sum(self.tsp.distance_matrix[visited[i]][visited[i+1]] for i in range(len(visited) - 1))
                ants_solutions.append((visited, length))

                # Update the best solution found
                if length < best_length:
                    best_length = length
                    best_solution = visited

            # Update pheromones based on the solutions found by ants
            self.update_pheromones(ants_solutions)

        return best_solution, best_length

# Example usage of the ACO algorithm
if __name__ == "__main__":
    # Define the cities as (x, y) coordinates
    cities = [(0, 0), (1, 3), (4, 3), (6, 1), (3, 0)]

    # Create a TSP problem instance
    tsp = TSP(cities)

    # Initialize ACO parameters
    num_ants = 10
    alpha = 1.0    # Influence of pheromone
    beta = 2.0     # Influence of distance (heuristic)
    rho = 0.1      # Pheromone evaporation rate
    iterations = 100

    # Create an ACO instance
    aco = ACO(tsp, num_ants, alpha, beta, rho, iterations)

    # Run the algorithm and get the best solution
    best_solution, best_length = aco.run()

    # Output the result
    print("Best solution:", best_solution)
    print("Best solution length:", best_length)
