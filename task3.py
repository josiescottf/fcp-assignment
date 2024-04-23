import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statistics
import argparse


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


def main():
    # creating an argument parser
    parser = argparse.ArgumentParser()

    # adding the parse arguments
    # argument to run the tests
    parser.add_argument("--test_network", action='store_true') 
    # argument to determine how many nodes in a random network
    parser.add_argument("--network", nargs=1)  
     # argument to assign connection probability in a random network
    parser.add_argument("--connection_probability", nargs=1, default=0.5) 

    args = parser.parse_args()

    # creating a network to work with
    network = Network()

    # creating a random network if network flag is given
    if args.network:
        network.make_random_network(int(args.network[0]), args.connection_probability[0])
        # finding mean degree of network
        print('Mean Degree:', network.get_mean_degree())
        # finding mean path length of the network
        print('Average Path Length:', network.get_path_length())
        # finding mean clustering coefficient of the network
        print('Clustering co-efficient:', network.get_mean_clustering())

    # testing functions if test_network flag is given
    if args.test_network:
        network.test_networks()


if __name__ == "__main__":
    main()