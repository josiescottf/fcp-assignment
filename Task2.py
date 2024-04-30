import numpy as np
import matplotlib.pyplot as plt
import random
import argparse


def initial_opinions(no_opinions):
    '''
    Creating a list of values between 1 and 0 of a variable length
    Input: no_opinions
    Output: List of random values between 0 and 1 of length no_opinions
    '''
    return np.random.random(no_opinions)


def compare(opinions, threshold, beta):
    '''
    Compares the value of the randomly selected person and a random neighbour
    Input: opinions, threshold, beta
    Output: Updated list of opinions
    '''
    for i in range(len(opinions)):
        person_index = random.randint(0, len(opinions)-1)  # Gets a random index for a 'person'.
        # Fixes the edge cases
        if person_index == 0:
            neighbours_index = 1  # Only chooses neighbor to the right.
        elif person_index == len(opinions) - 1:
            neighbours_index = person_index - 1  # Only chooses neighbor to the left.
        else:
            neighbours_index = random.choice([person_index + 1, person_index - 1])  # Randomly chooses a neighbor from either side of the 'person'.

        difference = opinions[neighbours_index] - opinions[person_index] 

        if abs(difference) < threshold: # The question asks for the modulus of the different, so abs function employed.
            opinions[person_index] += beta * difference
            opinions[neighbours_index] -= beta * difference 

    return opinions


def interact(no_opinions, threshold, beta, repetitions):
    '''
    Compares all of the 'people' and 'neighbours' for a variable amount of iterations
    Input: no_opinions, threshold, beta, repetitions
    Output: Updated list of opinions and a list of all the opinions stored over time (stored_opinions_grid)
    '''
    opinions_grid = initial_opinions(no_opinions)
    stored_opinions_grid = [opinions_grid.copy()]  # Stores a copy of the initial state

    for i in range(repetitions):
        opinions_grid = compare(opinions_grid, threshold, beta)
        stored_opinions_grid.append(opinions_grid.copy())  # Append a copy after each interaction to create the list to be used in the scatter graph.

    return opinions_grid, stored_opinions_grid


def y_values(stored_opinions_grid):
    '''
    Extracts the y-values (opinions) and corresponding x-values (time steps) from a list of opinions stored over multiple time steps
    Input: The list of all the opinions stored over time (stored_opinions_grid)
    Output: Two lists, y_list and x_list for the scatter graph
    '''
    y_list = []
    x_list = []
    for index, opinions in enumerate(stored_opinions_grid):  # Enumerate gives us both the index of each opinion and the opinion itself.
        x_list.extend([index] * len(opinions))  # Creates the timesteps.
        y_list.extend(opinions)  # Adds all the opinions from the current time step to the y_list for the scatter graph.
    return y_list, x_list


def plot_scatter(ax, stored_opinions_grid):
    ''' Plots a scatter graph of stored opinions against time steps'''
    y_list, x_list = y_values(stored_opinions_grid)
    ax.scatter(x_list, y_list, color='red')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Opinions')


def plot_hist(ax, opinions_list):
    ''' Plots a histogram of the opinions from one timestep'''
    ax.hist(opinions_list)
    ax.set_xlim([0, 1])
    ax.set_xlabel('Opinions')


def argparse_function():
    ''' Function to parse the arguments from the command line'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-defuant', action='store_true')  # Set a flag for -defuant.
    parser.add_argument('-beta', type=float, default=0.2)  # Set default to 0.2 for beta and threshold.
    parser.add_argument('-threshold', type=float, default=0.2)
    return parser.parse_args()


def test_initial_opinion():
    assert len(initial_opinions(10)) == 10 # Checks if the length of the list is equal to the input.
    assert all(0 <= x <= 1 for x in initial_opinions(10)) # Checks if all the values in the list are between 0 and 1.


def test_compare():
    opinions = [0.6, 0.5, 0.6]
    threshold = 0.5
    beta = 0.2
    assert compare(opinions, threshold, beta) == [0.55, 0.5, 0.55]   # Checks if the compare function works for a difference of 0.1.


def test_interact():
    no_opinions = 10
    threshold = 0.1
    beta = 0.2
    repetitions = 10
    opinions, stored_opinions_grid = interact(no_opinions, threshold, beta, repetitions)
    assert len(stored_opinions_grid) == repetitions + 1 # Checks if the length of the stored opinions is equal to the repetions plus the original list.
    assert len(stored_opinions_grid[0]) == no_opinions # Checks if the first list in the stored opinions is equal to the number of opinions.


def test_y_values():
    stored_opinions_grid = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
    y_list, x_list = y_values(stored_opinions_grid)
    assert y_list == [0.1, 0.2, 0.3, 0.2, 0.3, 0.4]
    assert x_list == [0, 0, 0, 1, 1, 1]


def test_all_functions():
    test_initial_opinion()
    print('Initial opinion test passed')
    test_compare()
    print('Compare test passed')
    test_interact()
    print('Interact test passed')
    test_y_values()
    print('Y values test passed')


def main():
    args = argparse_function() 

    no_opinions = 100  # Defines variables.
    threshold = args.threshold # Sets the threshold and beta to the input values.
    beta = args.beta
    repetitions = 100

    opinions_grid, stored_opinions_grid = interact(no_opinions, threshold, beta, repetitions) 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6)) # Creates the subplots.

    plot_hist(ax1, opinions_grid) 
    plot_scatter(ax2, stored_opinions_grid)
    fig.suptitle(f'Coupling: {beta} and Threshold: {threshold}')

    plt.tight_layout()
    plt.show()

    test_all_functions()


if __name__ == "__main__":
    main()


