import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

class Node:

	def __init__(self, value, number, connections=None):

		self.index = number
		self.connections = connections
		self.value = value
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
                node.connection[new_neighbour_index] = 1
                self.nodes[new_neighbour_index].connections[node.index] = 1
