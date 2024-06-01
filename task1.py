import matplotlib.pyplot as plt
import numpy as np


# a)
def generate_bernoulli_matrix(N, n):
    matrix = np.random.randint(2, size=(N, n))
    return matrix


matrix = generate_bernoulli_matrix(N=20_000, n=20)

empirical_means = np.mean(matrix, axis=1)

# b)
epsilon_values = np.linspace(0, 1, 50)

empirical_probabilities = []
for epsilon in epsilon_values:
    probability = np.mean(np.abs(empirical_means - 0.5) > epsilon)
    empirical_probabilities.append(probability)

hoeffding_bounds = 2 * np.exp(-2 * 20 * epsilon_values**2)


plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, empirical_probabilities, marker='o', label='Empirical Probability')
plt.plot(epsilon_values, hoeffding_bounds, marker='x', linestyle='--', color='r', label='Hoeffding Bound')
plt.xlabel('epsilon')
plt.ylabel('Probability')
plt.title('Empirical Probability and Hoeffding Bound')
plt.legend()
plt.grid(True)
plt.show()