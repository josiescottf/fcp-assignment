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

    # calculate the mean degree of the network
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
            # store the degree of all nodes in a list
            degrees.append(degree)

        # calculate mean
        mean_degree = statistics.mean(degrees)

        return mean_degree

    # calculate the mean path length of the network
    def get_mean_path_length(self):
        '''
        A function that calculates the mean path length between all nodes in a network
        Inputs:
            Self
        Outputs:
            mean_path_length - The average path length between two nodes in a network
        '''
        # calculate the mean path length from a node to all other nodes
        # calculate the path length between 2 nodes
        # store the mean path length of all nodes in a list
        # calculate mean
        mean_path_length = 0

        return mean_path_length


    # calculate the mean clustering coefficient of the network
    def get_mean_clustering(self):
        '''
        A function that calculates the mean clustering coefficient in a network
        (clustering coefficient is the fraction of a nodes neighbours that connect to each other)
        Inputs:
            Self
        Outputs:
            mean_clustering - The mean clustering coefficient of the network
        '''
        # calculate the clustering coefficient of a node
        # create list of all neighbours of the node
        # calculate maximum number of connections between neighbours
        # find actual number of connections between neighbours
        # use these to find clustering coefficient
        # store the clustering coefficient of all nodes in a list
        # calculate mean
        mean_clustering = 0

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
        #assert (network.get_clustering() == 0), network.get_clustering()
        #assert (network.get_path_length() == 2.777777777777778), network.get_path_length()

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
        #assert (network.get_clustering() == 0), network.get_clustering()
        #assert (network.get_path_length() == 5), network.get_path_length()

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
        #assert (network.get_clustering() == 1), network.get_clustering()
        #assert (network.get_path_length() == 1), network.get_path_length()

        print("All tests passed")


def main():
    # creating an argument parser
    parser = argparse.ArgumentParser()

    # adding the parse arguments
    parser.add_argument("--test_network", action='store_true')  # argument to run the tests
    parser.add_argument("--network", nargs=1)  # argument to determine how many nodes in a random network
    parser.add_argument("--connection_probability", nargs=1,
                        default=0.5)  # argument to assign connection probability in a random network

    args = parser.parse_args()

    network = Network()

    if args.network:
        network.make_random_network(int(args.network[0]), args.connection_probability)
        print(network.get_mean_degree())

    if args.test_network:
        network.test_networks()


if __name__ == "__main__":
    main()
