import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import random
import math

# Import node class
class Node:
    '''
    Class to represent a node in an undirected graph
    Each node has a floating point value and some connections
    '''


    def __init__(self, value, number, connections=None):
        self.index = number
        self.connections = connections
        self.value = value


    def get_neighbours(self):
        return np.where(np.array(self.connections) == 1)[0]


# Import network class
class Network:

    def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def get_mean_degree(self):
        '''
        A function that calculates the mean degree of all nodes in a given network
        Inputs:
            Self
        Outputs:
            mean_degree - The mean degree of all nodes in the network
        '''

        # Initialising list of degrees of each node
        degrees = []

        for node in self.nodes:
            # calculate the degree of each node (how many neighbours does the node have)
            neighbours = node.get_neighbours()
            degree = len(neighbours)
            # store the degrees of all nodes in a list
            degrees.append(degree)

        # calculate mean
        mean_degree = np.mean(degrees)

        return mean_degree

    def get_path_length(self):
        '''
        A function that calculates the mean path length between all nodes in a network
        Inputs:
            Self
        Outputs:
            mean_path_length - The average path length between two nodes in a network
        '''
        
        # Initialising a list to store the average path lengths for each node
        mean_paths = []

        # calculate the mean path length from a node to all other nodes
        for node in self.nodes:
            # Initialising a list to store the path length from one node to all others
            path_lengths = np.zeros(len(self.nodes))
            # Initialising path length covered
            n = 0
            # Initalising lists of neighbours to be tested and neighbours that have been tested
            new_neighbours = node.get_neighbours()
            tested_neighbours = [node.index]

            # Looping through every node found until there are none left
            while len(new_neighbours) > 0:
                n += 1

                # Adding path lengths to the list
                for i in new_neighbours:
                    path_lengths[i] = n
                
                # Adding to list of nodes that have been tested
                tested_neighbours = np.append(tested_neighbours, new_neighbours)

                # initialising list to store next degree of neighbours
                next_new_neighbours = []
                # finding next degree of neighbours 
                for i in new_neighbours:
                    next_new_neighbours = np.concatenate((next_new_neighbours, self.nodes[i].get_neighbours()))

                # removing duplicates from this list
                next_new_neighbours = np.unique(next_new_neighbours)

                # removing unnecessary nodes from this list
                for i in tested_neighbours:
                    next_new_neighbours = np.delete(next_new_neighbours, np.where(next_new_neighbours == i))
                
                # asserting the new set of neighbours (dtype=int ensures that the data can be used to iterate through lists later)
                new_neighbours = np.array(next_new_neighbours, dtype=int)

            # removing 0 values from path length (don't want to record path distance to self)
            path_lengths = np.delete(path_lengths, np.where(path_lengths == 0))
            mean_paths.append(np.mean(path_lengths))

        # calculate mean
        mean_path_length = round(np.mean(mean_paths),15)

        return mean_path_length


    def get_mean_clustering(self):
        '''
        A function that calculates the mean clustering coefficient in a network
        (clustering coefficient is the fraction of a nodes neighbours that connect to each other)
        Inputs:
            Self
        Outputs:
            mean_clustering - The mean clustering coefficient of the network
        '''
        
        clustering_coefficients = []

        for node in self.nodes:
            # create list of all neighbours of the node
            neighbours = node.get_neighbours()
            n = len(neighbours)
            
            # calculate maximum number of connections between neighbours
            max_connections = n * (n-1)/2
            
            # Initialise number of connections between neighbours
            connections = 0

            # find actual number of connections between neighbours
            for i in neighbours:
                # get neighbours of the neighbour
                possible_connections = self.nodes[i].get_neighbours()
                for j in possible_connections:
                    # only add to the number of connections if i<j so each connection is only counted once
                    if i < j and j in neighbours:
                        connections += 1
        
            # find clustering coefficient
            if max_connections != 0:
                clustering = connections/max_connections
            else:
                clustering = 0
        
            # store the clustering coefficient of all nodes in a list
            clustering_coefficients.append(clustering)
        
        # calculate mean
        mean_clustering = np.mean(clustering_coefficients)

        return mean_clustering

    def make_random_network(self, N, connection_probability):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p
        '''

        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1


    def make_ring_network(self, N, neighbour_range=1):
        '''
        A function that creates a ring network where each node is connected to its nearest neighbours

        Inputs:
            self
            N - Number of nodes in the network
            neighbour_range - Range of neighbours to connect to 
        Output:
            self - network with newly assigned nodes
        '''
        # initialise an empty list to store nodes
        self.nodes = [] 
        for node_number in range(N):
            # initialise connections for the node
            connections = [0] * N 
            for i in range(1, neighbour_range + 1):
                # calculate indices of left and right neighbour in the ring
                left_neighbor_index = (node_number - i) % N
                right_neighbor_index = (node_number + i) % N
                if left_neighbor_index != node_number:
                    # connect to the left neighbour if it's not the same as the current node
                    connections[left_neighbor_index] = 1
                if right_neighbor_index != node_number:
                    # connect to the right neighbour if it's not the same as the current node
                    connections[right_neighbor_index] = 1
            # create a new node with the calculated connections
            new_node = Node(np.random.random(), node_number, connections=connections)
            # add the node to the network
            self.nodes.append(new_node) 


    def make_small_world_network(self, N, re_wire_prob=0.1):
        '''
        A function that creates a small-world network by rewiring some connections of a ring network
        Inputs:
            self
            N - number of nodes in the network
            re_wire_prob - Probability of re-wiring a connection
        Output:
            self - network with newly assigned nodes
        '''
        self.make_ring_network(N, neighbour_range=2) #Start with a ring network of range 2
        for node in self.nodes:
            for neighbour_index, connection in enumerate(node.connections):
                # check if there is a connection and apply rewiring with a certain probability
                if connection == 1 and random.random() < re_wire_prob:
                    # choose a new neighbour index for rewiring
                    new_neighbour_index = random.choice([i for i in range(N) if i!= node.index and i != neighbour_index])
                    node.connections[neighbour_index] = 0
                    self.nodes[neighbour_index].connections[node.index] = 0
                    node.connections[new_neighbour_index] = 1
                    self.nodes[new_neighbour_index].connections[node.index] = 1

        return self


    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
        ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i + 1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

    def test_networks(self):
        # Ring network
        nodes = []
        num_nodes = 10
        for node_number in range(num_nodes):
            connections = [0 for val in range(num_nodes)]
            connections[(node_number - 1) % num_nodes] = 1
            connections[(node_number + 1) % num_nodes] = 1
            new_node = Node(0, node_number, connections=connections)
            nodes.append(new_node)
        network = Network(nodes)

        print("Testing ring network")
        assert (network.get_mean_degree() == 2), network.get_mean_degree()
        assert (network.get_mean_clustering() == 0), network.get_clustering()
        assert (network.get_path_length() == 2.777777777777778), network.get_path_length()

        nodes = []
        num_nodes = 10
        for node_number in range(num_nodes):
            connections = [0 for val in range(num_nodes)]
            connections[(node_number + 1) % num_nodes] = 1
            new_node = Node(0, node_number, connections=connections)
            nodes.append(new_node)
        network = Network(nodes)

        print("Testing one-sided network")
        assert (network.get_mean_degree() == 1), network.get_mean_degree()
        assert (network.get_mean_clustering() == 0), network.get_clustering()
        assert (network.get_path_length() == 5), network.get_path_length()

        nodes = []
        num_nodes = 10
        for node_number in range(num_nodes):
            connections = [1 for val in range(num_nodes)]
            connections[node_number] = 0
            new_node = Node(0, node_number, connections=connections)
            nodes.append(new_node)
        network = Network(nodes)

        print("Testing fully connected network")
        assert (network.get_mean_degree() == num_nodes - 1), network.get_mean_degree()
        assert (network.get_mean_clustering() == 1), network.get_clustering()
        assert (network.get_path_length() == 1), network.get_path_length()

        print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

def get_neighbour_opinions(population, i, j):
    '''
    This function returns a list of the neighbouring (right, left, above and below) peoples' opinions
    Inputs: 
        population (numpy array)
        i (int) - row of person whos neighbours' opinions are being collected
        j (int) - column of person whos neighbours' opinions are being collected
    Returns:
        neighbours - list of neighbours' opinions
    '''
    #initializing 2 variables that are equal to the amount of rows and columns there are in the population
    n,m = population.shape
    #initializing empty list of neighbours' opinions to be filled
    neighbours = []
    #appending the opinion of the left neighbour
    neighbours.append(population[i-1, j])
    #appending the opinion of the right neighbour. Using a modulus so that it will loop round if at the edge of population
    neighbours.append(population[(i+1)%n, j])
    #appending the opinion of the above neighbour
    neighbours.append(population[i, j-1])
    #appending the opinion of the below neighbour.  Using a modulus so that it will loop round if at the edge of population
    neighbours.append(population[i, (j+1)%m])
    #returning list of neighbours' opinions
    return neighbours
    

def calculate_agreement(population, row, col, external=0.0):
    '''
    This function should return the *change* in agreement that would result if the cell at (row, col) was to flip it's value
    Inputs: 
        population (numpy array)
        row (int)
        col (int)
        external (float) - optional - the magnitude of any external "pull" on opinion
    Returns:
        change_in_agreement (float)
    '''
    #initializing the agreement value
    agreement = 0
    #setting the self opinion based on the row and column
    self_opinion = population[row, col]
    #calls 'neighbour_opinions' function to collect the list of neighbouring peoples' opinions
    neighbour_opinions = get_neighbour_opinions(population, row, col)
    #calculating the agreement
    for opinion in neighbour_opinions:
        #increases agreement if opinions are the same but will decrease if they oppose
        agreement += self_opinion * opinion
    #increases agreement using the external pull value
    agreement += external * self_opinion
    return agreement

def calculate_agreement_network(network, node_number, external):
    '''
    Function to calculate the change in agreement of a node in a network
    Inputs:
        network - The network containing the node
        node_number - The index of the node to be analysed
        external - The magnitude of any external "pull" on opinions
    Outputs:
        agreement - The change in agreement of the node
    '''
    primary_node = network.nodes[node_number]

    #initializing the agreement value and opinions
    agreement = 0
    neighbour_opinions = []

    # Find the neighbours of the node
    neighbours = primary_node.get_neighbours()
    for node in neighbours:
        node = network.nodes[node]
        neighbour_opinions.append(node.value)

    #calculating the agreement
    for opinion in neighbour_opinions:
        #increases agreement if opinions are the same but will decrease if they oppose
        agreement += node.value * opinion

    #increases agreement using the external pull value
    agreement += external * node.value

    return agreement  

def ising_step(population, external=0.0, alpha=1):
    '''
    This function will perform a single update of the Ising model
    Inputs: 
        population (numpy array)
        external (float) - optional - the magnitude of any external "pull" on opinion
        alpha (float) - optional - the magnitude of the alpha value to be used in calculation
    '''
    
    #setting the values of the number of columns and rows based on the population
    n_rows, n_cols = population.shape
    #picking a random row
    row = np.random.randint(0, n_rows)
    #picking a random column
    col  = np.random.randint(0, n_cols)
    #setting the agreement to the result of the value given after running the calculation fucntion
    agreement = calculate_agreement(population, row, col, external=0.0)
    
    #negating the opinion of the person if the agreement is negative
    if agreement < 0:
        population[row, col] *= -1
    
    #if opinion wasn't negated, this elif runs a random chance that they may be negated anyway
    elif alpha:
        #producing random number between 0 and 1
        random_number = random.random()
        #calculating probability of opinion flip
        p = math.e**(-agreement/alpha)
        #checking if random number is less than p
        if random_number < p:
            #negating the opinion if the random number is less than p
            population[row, col] *= -1
    
def ising_network_step(network, external=0.0, alpha=1):
    '''
    This function will perform a single update of the Ising model for a network
    Inputs: 
        network - a network of nodes with opinions
        external (float) - optional - the magnitude of any external "pull" on opinion
        alpha (float) - optional - the magnitude of the alpha value to be used in calculation
    '''

    node_number = np.random.randint(0, len(network.nodes))
    node = network.nodes[node_number]
    
    #setting the agreement to the result of the value given after running the calculation fucntion
    agreement = calculate_agreement_network(network, node_number, external)
    
    #negating the opinion of the person if the agreement is negative
    if agreement < 0:
        node.value *= -1
    
    #if opinion wasn't negated, this elif runs a random chance that they may be negated anyway
    elif alpha:
        #producing random number between 0 and 1
        random_number = random.random()
        #calculating probability of opinion flip
        p = math.e**(-agreement/alpha)
        #checking if random number is less than p
        if random_number < p:
            #negating the opinion if the random number is less than p
            node.value *= -1

def mean_opinion(network):
    """
    Calculate the mean opinion of all the nodes at each evolution step.
    Inputs:
        network - A network of opinions
    Outputs:
        mean_opinion - The mean opinion of the network
    """
    
    opinions = [node.value for node in network.nodes]
    mean_opinion = np.mean(opinions)

    return mean_opinion
    

def plot_ising(im, population):
    '''
    This function will display a plot of the Ising model
    Inputs:
        im (AxesImage) - the image data and properties necessary for displaying the image of the population array on the plot
        population (numpy array)
    '''
    #creating new image for plot
    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    #slight pause so that each frame is shown temporarily
    plt.pause(0.01)

    
def test_ising():
    '''
    This function will test the calculate_agreement function in the Ising model
    '''
    #prints message to indicate testing of the ising model calculation is taking place
    print("Testing ising model calculations")
    #creates a small population to run tests on
    population = -np.ones((3, 3))
    #Running all the tests
    assert(calculate_agreement(population,1,1)==4), "Test 1"
    population[1, 1] = 1.
    assert(calculate_agreement(population,1,1)==-4), "Test 2"
    population[0, 1] = 1.
    assert(calculate_agreement(population,1,1)==-2), "Test 3"
    population[1, 0] = 1.
    assert(calculate_agreement(population,1,1)==0), "Test 4"
    population[2, 1] = 1.
    assert(calculate_agreement(population,1,1)==2), "Test 5"
    population[1, 2] = 1.
    assert(calculate_agreement(population,1,1)==4), "Test 6"
    #prints message to indicate testing of the ising model calculation when there is an external pull
    print("Testing external pull")
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1,1)==3), "Test 7"
    assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
    assert(calculate_agreement(population,1,1,10)==-6), "Test 9"
    assert(calculate_agreement(population,1,1,-10)==14), "Test 10"
    #prints message when all tests have passed successfully
    print("Tests passed")


def ising_network(ising_network_size, alpha=None, external=0):
    '''
    This function forms the main plot for the ising model on a network
    Inputs:
        ising_network_size - Number of nodes in the small world network to be analysed
        alpha - (optional) the magnitude of the alpha value to be used in calculation
        external - (optional) the magnitude of any external "pull" on opinion
    '''

    network = Network()
    mean_opinions = []
    steps = np.arange(1,11,1)
    small_world_network = network.make_small_world_network(ising_network_size)
    # set a random value for opinions of each node, either -1 or 1
    for node in small_world_network.nodes:
        node.value = random.uniform(-1, 1)
    
    # Iterating an update 100 times
    for frame in range(len(steps)):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            #calling step function
            ising_network_step(network, external, alpha)
        #plotting each one of the 100 frames
        mean_opinions.append(mean_opinion(network))
        network.plot()
        plt.show()

    # Plotting the mean opinions of the network over time
    plt.plot(steps, mean_opinions)
    plt.title('Mean opinions over time')
    plt.xlabel('Steps')
    plt.ylabel('Mean Opinion')
    plt.show()

#Main function for ising model - Task 1
def ising_main(population, alpha=None, external=0.0):
    '''
    This function forms the main plot of the ising model
    Inputs: 
        population (numpy array)
        alpha (float) - optional - the magnitude of the alpha value to be used in calculation
        external (float) - optional - the magnitude of any external "pull" on opinion
    '''
    #creating the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')
    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            #calling step function
            ising_step(population, external, alpha)
        #plotting each one of the 100 frames
        plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''


def initial_opinions(no_opinions):
    '''Creating a list of values between 1 and 0 of a variable length'''
    return np.random.random(no_opinions)

def compare(opinions, threshold, beta):
    '''
    A function that compares the value of the randomly selected person and a random neighbour
    Inputs:
        opinions - initial opinions on a grid 
        threshold - threshold under which opinions will change
        beta - multiplier for opinion change
    Outputs:
        opinions - updated opinions on a grid
    '''
    for i in range(len(opinions)):
        # Random index for a 'person'.
        person_index = random.randint(0, len(opinions)-1)  
        # Fixes the edge cases
        if person_index == 0:
             # Only chooses neighbor to the right.
            neighbours_index = 1 
        elif person_index == len(opinions) - 1:
            # Only chooses neighbor to the left.
            neighbours_index = person_index - 1  
        else:
            # Randomly chooses a neighbor from either side of the 'person'.
            neighbours_index = random.choice([person_index + 1, person_index - 1])  

        difference = opinions[neighbours_index] - opinions[person_index] 

        # The question asks for the modulus of the different, so abs function employed.
        if abs(difference) < threshold: 
            opinions[person_index] += beta * difference
            opinions[neighbours_index] -= beta * difference 

    return opinions

def interact(no_opinions, threshold, beta, repetitions):
    '''
    Compares all of the 'people' and 'neighbours' for a variable amount of iterations
    Inputs:
        no_opinions - number of opinions to analyse
        threshold - threshold under which opinions will change
        beta - multiplier for opinion change
        repetitions - number of repetitions to run the code for
    Outputs:
        opinions_grid - final opinions on the grid
        stored_opinions - all opinions held on the grid over time
    '''
    opinions_grid = initial_opinions(no_opinions)
    # Store a copy of the initial state
    stored_opinions = [opinions_grid.copy()]  

    for _ in range(repetitions):
        opinions_grid = compare(opinions_grid, threshold, beta) 
        # Append a copy after each interaction to create the list to be used in the scatter graph.
        stored_opinions.append(opinions_grid.copy()) 

    return opinions_grid, stored_opinions

def y_values(stored_opinions):
    '''
    Extracts the y-values (opinions) and corresponding x-values (time steps) from a list of opinions stored over multiple time steps
    Inputs:
        stored_opinions - all opinions held on a grid over time
    Outputs:
        x_list - timesteps to be plotted on the x-axis
        y_list - opinions to be plotted on the y-axis
    '''
    y_list = []
    x_list = []
    for index, opinions in enumerate(stored_opinions): # Enumerate gives us both the index of each opinion and the opinion itself.
        # Creates the timesteps.
        x_list.extend([index] * len(opinions)) 
        # Adds all the opinions from the current time step to the y_list for the scatter graph.
        y_list.extend(opinions) 
    return y_list, x_list

def plot_scatter(ax, stored_opinions):
    ''' 
    Plots a scatter graph of stored opinions against time steps
    '''
    y_list, x_list = y_values(stored_opinions)
    ax.scatter(x_list, y_list, color='red')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Opinions')

def plot_hist(ax, opinions_list):
    ''' 
    Plots a histogram of the opinions from one timestep
    '''
    ax.hist(opinions_list)
    ax.set_xlim([0, 1])
    ax.set_xlabel('Opinions')

def test_initial_opinion():
    '''
    Tests initial opinions in the defuant model
    '''
    assert len(initial_opinions(10)) == 10
    assert all(0 <= x <= 1 for x in initial_opinions(10))

def test_compare():
    '''
    Tests the compare function of the defuant model
    '''
    opinions = [0.5, 0.5, 0.5]
    threshold = 0.1
    beta = 0.2
    assert compare(opinions, threshold, beta) == [0.5, 0.5, 0.5]

def test_interact():
    '''
    Tests the interact function of the defuant model
    '''
    no_opinions = 10
    threshold = 0.1
    beta = 0.2
    repetitions = 10
    opinions, stored_opinions = interact(no_opinions, threshold, beta, repetitions)
    assert len(stored_opinions) == repetitions + 1
    assert len(stored_opinions[0]) == no_opinions
 
def test_y_values():
    '''
    Tests the y_values function of the defuant model
    '''
    stored_opinions = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
    y_list, x_list = y_values(stored_opinions)
    assert y_list == [0.1, 0.2, 0.3, 0.2, 0.3, 0.4]
    assert x_list == [0, 0, 0, 1, 1, 1]

def test_all_functions():
    '''
    Tests all functions for the defuant model
    '''
    test_initial_opinion()
    print('Initial opinion test passed')
    test_compare()
    print('Compare test passed')
    test_interact()
    print('Interact test passed')
    test_y_values()
    print('Y values test passed')

def defuant_main(no_opinions, threshold, beta, repetitions):
    '''
    Function that runs the defuant model
    Inputs:
        no_opinions - Number of opinions to analyse
        threshold - Threshold for opinions to change
        beta - Multiplier for opinion change
        repetitions - Number of times to analyse opinions
    Output:
        Creates graphs of opinions over time
    '''

    opinions_grid, stored_opinions = interact(no_opinions, threshold, beta, repetitions)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
    plot_hist(ax1, opinions_grid)
    plot_scatter(ax2, stored_opinions)
    fig.suptitle(f'Coupling: {beta} and Threshold: {threshold}')
    plt.tight_layout()
    plt.show()

def network_main(network_size, connection_probability):
    '''
    Function that runs the network model
    Inputs:
        network_size - Size of network to be created
        connection_probability - Probability of two nodes connecting
    Outputs:
        Calculates and displays the mean degree, mean path length and mean clustering coefficient of the network.
        Plots a graph of the network.
    '''
    # Creating a random network
    network = Network()
    network.make_random_network(int(network_size[0]), connection_probability)

    # Finding mean degree, mean path length and mean clustering coefficient
    print('Mean Degree:', network.get_mean_degree())
    print('Average Path Length:', network.get_path_length())
    print('Clustering co-efficient:', network.get_mean_clustering())
    network.plot()
    plt.show()

def small_world_main(small_world_size, re_wire_prob):
    '''
    Function that runs the small world network model and plots a graph of this
    '''
    small_world = Network()
    small_world.make_small_world_network(small_world_size, re_wire_prob)
    small_world.plot()
    plt.show()

def ring_main(ring_size):
    '''
    Function that runs the ring network model and plots a graph of this
    '''
    ring = Network()
    ring.make_ring_network(ring_size)
    ring.plot()
    plt.show()

def main():
    # Creating argument parser
    parser = argparse.ArgumentParser()
    
    #adding parse flags
    parser.add_argument("-ising_model", action='store_true', help='runs the ising model with default values') 
    parser.add_argument("-external", type=float, default=0, help='takes a value to use for the external pull') 
    parser.add_argument("-alpha", type=float, default=1, help='takes a value to use for alpha')
    parser.add_argument("-test_ising", action='store_true', help='checks whether to run tests for the ising model')
    parser.add_argument("-defuant", action='store_true', help='runs the defuant model')
    parser.add_argument('-beta', type=float, default=0.2, help='multiplier for opinion change in defuant model') 
    parser.add_argument("-threshold", type=float, default=0.2, help='opinion change threshold in defuant model')
    parser.add_argument("-use_network", type=int, help='flag to use network in ising model')
    parser.add_argument("-test_network", action="store_true", help=' runs network tests')
    parser.add_argument("-network", nargs=1, help='determines how many nodes there will be in a random network')
    parser.add_argument("-connection_probability", nargs=1, default=0.5, help='assigns connection probability in a random network')
    parser.add_argument("-no_opinions", nargs=1, default=100, help='number of opinions to analyse in the defuant model')
    parser.add_argument("-repetitions", nargs=1, default=100, help='number of repetitions for defuant')
    parser.add_argument("-ring_network", type=int, help='Generate a ring network of specified size')
    parser.add_argument("-small_world", type=int, help='Generate a small-world network of specified size with defualt parameters')
    parser.add_argument('-re_wire', type=float, default=0.1, help='Set the re-wiring probability for small-world network (default:0.1)')
    
    #Defining variables from arguments
    args = parser.parse_args()
    external = args.external
    alpha = args.alpha
    no_opinions = args.no_opinions
    threshold = args.threshold
    beta = args.beta
    repetitions = args.repetitions
    connection_probability = args.connection_probability
    ring_size = args.ring_network
    small_world_size = args.small_world
    re_wire_prob = args.re_wire
    ising_network_size = args.use_network


    if args.ising_model:
        # if both '-ising_model' and '-use_network' flags are present, will run the ising model on a network
        if args.use_network:
            ising_network(ising_network_size, alpha, external)
        # if only 'ising_model' flag is present, runs the ising model on a 100 by 100 grid
        else:
            pop = np.random.choice([-1,1],size=(100,100))
            ising_main(pop, alpha, external)
    
    #checks for '-test_ising' flag and runs ising test function if present
    if args.test_ising:
        test_ising()

    # checks for '-defuant' flag and runs the defuant model if present
    if args.defuant:
        defuant_main(no_opinions, threshold, beta, repetitions)

    # Checks for '-network' flag and generates a random network if present
    if args.network:
        network_main(args.network, connection_probability)

    # Tests network functions if '-test_network' flag given
    if args.test_network:
        network = Network()
        network.test_networks()

    # Check for '-ring_network' flag and creates a ring network if present
    if args.ring_network:
        ring_main(ring_size)

    # Checks for '-small_world' flag and creates a small world network if present
    if args.small_world:
        small_world_main(small_world_size, re_wire_prob)


if __name__=="__main__":
    main()