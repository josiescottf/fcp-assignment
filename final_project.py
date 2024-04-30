import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import random
import math
import statistics 

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
        mean_degree = statistics.mean(degrees)

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
            mean_paths.append(statistics.mean(path_lengths))

        # calculate mean
        mean_path_length = round(statistics.mean(mean_paths),15)

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
        mean_clustering = statistics.mean(clustering_coefficients)

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
        self.nodes = []
        for node_number in range(N):
            connections = [0] * N
            for i in range(1, neighbour_range + 1):
                left_neighbor_index = (node_number - i) % N
                right_neighbor_index = (node_number + i) % N
                if left_neighbor_index != node_number:
                    connections[left_neighbor_index] = 1
                if right_neighbor_index != node_number:
                    connections[right_neighbor_index] = 1
            new_node = Node(0, node_number, connections=connections)
            self.nodes.append(new_node)


    def make_small_world_network(self, N, re_wire_prob=0.1):
        self.make_ring_network(N, neighbour_range=2) #Start with a ring network of range 2
        for node in self.nodes:
            for neighbour_index, connection in enumerate(node.connections):
                if connection == 1 and random.random() < re_wire_prob:
                    new_neighbour_index = random.choice([i for i in range(N) if i!= node.index and i != neighbour_index])
                    node.connections[neighbour_index] = 0
                    self.nodes[neighbour_index].connections[node.index] = 0
                    node.connections[new_neighbour_index] = 1
                    self.nodes[new_neighbour_index].connections[node.index] = 1


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
    Inputs: population (numpy array)
            i (int) - row of person whos neighbours' opinions are being collected
            j (int) - column of person whos neighbours' opinions are being collected
    Returns:
            list of neighbours' opinions
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
    Inputs: population (numpy array)
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

def calculate_agreement_network(node, external):
        '''
        Function to calculate the change in agreement of a node
        '''

        #initializing the agreement value
        agreement = 0
    
        # Find the neighbours of the node
        neighbour_opinions = node.get_neighbours()

        #calculating the agreement
        for opinion in neighbour_opinions:
            #increases agreement if opinions are the same but will decrease if they oppose
            agreement += node * opinion
        #increases agreement using the external pull value
        agreement += external * node

        return agreement  

def ising_step(population, external=0.0, alpha=1):
    '''
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
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


def ising_network(network, alpha=None, external=0):
    '''
    This function forms the main plot for the ising model on a network
    Inputs:
        network 
        alpha - (optional) the magnitude of the alpha value to be used in calculation
        external - (optional) the magnitude of any external "pull" on opinion
    '''
    # creating the plot

    # iterating an update 100 times
    
#Main function for ising model - Task 1
def ising_main(population, alpha=None, external=0.0):
    '''
    This function forms the main plot
    Inputs: population (numpy array)
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
    '''Compares the value of the randomly selected person and a random neighbour'''
    for i in range(len(opinions)):
        person_index = random.randint(0, len(opinions)-1)  # Random index for a 'person'.
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
    '''Compares all of the 'people' and 'neighbours' for a variable amount of iterations'''
    opinions_grid = initial_opinions(no_opinions)
    stored_opinions = [opinions_grid.copy()]  # Stores a copy of the initial state

    for _ in range(repetitions):
        opinions_grid = compare(opinions_grid, threshold, beta) 
        stored_opinions.append(opinions_grid.copy())  # Append a copy after each interaction to create the list to be used in the scatter graph.

    return opinions_grid, stored_opinions

def y_values(stored_opinions):
    '''Extracts the y-values (opinions) and corresponding x-values (time steps) from a list of opinions stored over multiple time steps'''
    y_list = []
    x_list = []
    for index, opinions in enumerate(stored_opinions): # Enumerate gives us both the index of each opinion and the opinion itself.
        x_list.extend([index] * len(opinions)) # Creates the timesteps.
        y_list.extend(opinions)  # Adds all the opinions from the current time step to the y_list for the scatter graph.
    return y_list, x_list

def plot_scatter(ax, stored_opinions):
    ''' Plots a scatter graph of stored opinions against time steps'''
    y_list, x_list = y_values(stored_opinions)
    ax.scatter(x_list, y_list, color='red')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Opinions')

def plot_hist(ax, opinions_list):
    ''' Plots a histogram of the opinions from one timestep'''
    ax.hist(opinions_list)
    ax.set_xlim([0, 1])
    ax.set_xlabel('Opinions')

def test_initial_opinion():
    assert len(initial_opinions(10)) == 10
    assert all(0 <= x <= 1 for x in initial_opinions(10))

def test_compare():
    opinions = [0.5, 0.5, 0.5]
    threshold = 0.1
    beta = 0.2
    assert compare(opinions, threshold, beta) == [0.5, 0.5, 0.5]

def test_interact():
    no_opinions = 10
    threshold = 0.1
    beta = 0.2
    repetitions = 10
    opinions, stored_opinions = interact(no_opinions, threshold, beta, repetitions)
    assert len(stored_opinions) == repetitions + 1
    assert len(stored_opinions[0]) == no_opinions
 
def test_y_values():
    stored_opinions = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
    y_list, x_list = y_values(stored_opinions)
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

def network_main():
    make_small_world_network(self, network_size, 0.2)

def main():
    parser = argparse.ArgumentParser()
    
    #adding parse flags
    parser.add_argument("-ising_model", action='store_true') #runs the ising model with default values
    parser.add_argument("-external", type=float, default=0) #takes a value to use for the external pull
    parser.add_argument("-alpha", type=float, default=1) #takes a value to use for alpha
    parser.add_argument("-test_ising", action='store_true') #checks whether to run the tests
    parser.add_argument("-defuant", action='store_true') # runs the defuant model
    parser.add_argument('-beta', type=float, default=0.2) # multiplier for opinion change in defuant
    parser.add_argument("-threshold", type=float, default=0.2) # opinion change threshold in defuant
    parser.add_argument("-use_network", nargs=1, default=10) # flag to use network in ising model
    parser.add_argument("--test_network", action="store_true") # runs network tests
    parser.add_argument("--network", nargs=1) # determines how many nodes there will be in a random network
    parser.add_argument("--connection_probability", nargs=1, default=0.5) # assigns connection probability in a random network
    parser.add_argument("--no_opinions", nargs=1, default=100) # number of opinions to analyse in defuant
    parser.add_argument("--repetitions", nargs=1, default=100) # number of repetitions for defuant
    
    #Defining the variables
    args = parser.parse_args()
    external = args.external
    alpha = args.alpha
    no_opinions = args.no_opinions
    threshold = args.threshold
    beta = args.beta
    repetitions = args.repetitions

    #checks for '-ising_model' flag and creates a population and runs main code if present
    if args.ising_model:
        if args.use_network:
            network = Network()
            ising_network(network, alpha, external)
        else:
            pop = np.random.choice([-1,1],size=(100,100))
            ising_main(pop, alpha, external)
    
    #checks for '-test_ising' flag and runs tests fucntion if present
    if args.test_ising:
        test_ising()

    # checks for '-defuant' flag and runs the defuant model
    if args.defuant:
        defuant_main(no_opinions, threshold, beta, repetitions)


if __name__=="__main__":
    main()