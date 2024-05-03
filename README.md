# fcp-assignment
Repository for the 2024 further computer programming assignment.

TITLE: final_project.py

AUTHORS: Josie Scott Fleming, Esme Sturgess-Durden, Joss Stroud, Emily Shah.

DESCRIPTION:
Ising model: compares the opinions of one person with their neighbours and uses a grid of 1’s and -1’s.
Defuant model: simulates the change of peoples opinions based on their neighbours opinion. The functions compare the opinions, make the individuals interact and visualise the opinions over time.
Ring network: Creates a network where each node is connected to its nearest neighbours.
Small world network: Rewires the connections of the ring network.
The next stage would involve exploring real-world applications of our model.

FLAGS:
-ising_model  runs the Ising model with default values.
-external  takes a value to use for the external pull
-alpha  takes a value to use for alpha
-test_ising  checks whether to run the tests
-defuant  runs the defuant model
-beta  multiplier for opinion change in defuant
-threshold  opinion change threshold in defuant
-use_network  flag to use network in ising model
-test_network  runs network tests
-network  determines how many nodes there will be in a random network
-connection_probability  assigns connection probability in a random network
-no_opinions  number of opinions to analyse in defuant
-repetitions  number of repetitions for defuant 
-ring_network generates a ring network of specified size with default range 1
-small_world generates a small-world network of specified size with default parameters
-re_wire sets the re-wiring probability for small-world network

REQUIREMENTS:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import random
import math
import statistics 


RUN:
final_project.py
