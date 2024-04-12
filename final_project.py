import numpy as np
import matplotlib.pyplot as plt
import random


def initial_opinions(no_opinions):
    '''Creating a list of values between 1 and 0 of a variable length'''
    return np.random.rand(no_opinions)


def compare(opinions, threshold, beta):
    '''Compares the value of the randomly selected person and a random neighbour'''
    
    for i in range(len(opinions)):
        neighbours_index = [i+1,i-1]
        
        if i == 0:
            neighbours_index = i+1
        elif i == (len(opinions) - 1): 
            neighbours_index = i-1
        else: 
            neighbours_index = random.choice(neighbours_index) # choose a random neighbour


        difference = (opinions[neighbours_index] - opinions[i])

        if abs(difference) < threshold:
            opinions[i] += beta*(opinions[neighbours_index] - opinions[i])
            opinions[neighbours_index] += beta*(opinions[i] - opinions[neighbours_index])
    return opinions

def interact(no_opinions, threshold, beta, repetitions):
    opinions_grid = initial_opinions(no_opinions)
    stored_opinions = [opinions_grid]

    for i in range(repetitions):
        opinions_grid = compare(opinions_grid, threshold, beta)
        stored_opinions.append(opinions_grid)
    return stored_opinions

def main():
    no_opinions = 10
    threshold = 0.5
    beta = 0.1
    repetitions = 100

    stored_opinions = interact(no_opinions, threshold, beta, repetitions)

    plt.figure(figsize=(10, 6))
    for i, opinions in enumerate(stored_opinions):
        plt.plot(opinions, label=f"Iteration {i}")
    plt.xlabel("Individuals")
    plt.ylabel("Opinions")
    plt.title("Opinions Evolution Over Time")
    plt.legend()
    plt.show()









































'''

import numpy as np
import matplotlib.pyplot as plt

def initialize_grid(grid_size):
    return np.random.rand(grid_size)

def interact(opinions, threshold, beta):
    for i in range(len(opinions)):
        neighbor_indices = [i-1, i+1]
        for neighbor_index in neighbor_indices:
            if neighbor_index >= 0 and neighbor_index < len(opinions):
                diff = abs(opinions[i] - opinions[neighbor_index])
                if diff < threshold:
                    mean_opinion = (opinions[i] + opinions[neighbor_index]) / 2
                    opinions[i] += beta * (mean_opinion - opinions[i])
                    opinions[neighbor_index] += beta * (mean_opinion - opinions[neighbor_index])
    return opinions

def simulate(grid_size, threshold, beta, iterations):
    grid = initialize_grid(grid_size)
    for _ in range(iterations):
        grid = interact(grid, threshold, beta)
    return grid

def plot_opinions(opinions):
    plt.plot(opinions)
    plt.xlabel('Individuals')
    plt.ylabel('Opinions')
    plt.title('Opinions Distribution')
    plt.show()

def test_defuant(threshold_values, beta_values):
    for threshold in threshold_values:
        for beta in beta_values:
            opinions = simulate(100, threshold, beta, 100)
            plot_opinions(opinions)

if __name__ == "__main__":
    test_defuant([0.2, 0.5, 0.8], [0.1, 0.5, 1.0])




#with random neighbour 

import random

# Define the parameters
threshold = 0.1
beta = 0.5

# Define the function to update opinions
def update_opinions(opinions, threshold, beta):
    for i in range(len(opinions)):
        # Get neighbor indices
        neighbor_indices = [i - 1, i + 1]
        
        # Pick a random neighbor index
        random_neighbor_index = random.choice(neighbor_indices)
        
        # Check if the random neighbor index is within the valid range
        if 0 <= random_neighbor_index < len(opinions):
            # Calculate the difference of opinions
            diff = abs(opinions[i] - opinions[random_neighbor_index])
            
            # Update opinions if difference is less than threshold
            if diff < threshold:
                opinions[i] += beta * (opinions[random_neighbor_index] - opinions[i])
                opinions[random_neighbor_index] += beta * (opinions[i] - opinions[random_neighbor_index])
    
    return opinions

# Test the function
opinions = [random.random() for _ in range(10)]  # Generate random opinions
updated_opinions = update_opinions(opinions, threshold, beta)
print("Original opinions:", opinions)
print("Updated opinions:", updated_opinions)

'''