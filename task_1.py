import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import sys
import random
import math




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
This section contains code for the main function
==============================================================================================================
'''
def main():
    '''
    ==============================================================================================================
    Task 1
    ==============================================================================================================
    '''
    parser = argparse.ArgumentParser()
    #You should write some code for handling flags here
    
    #adding parse flags
    parser.add_argument("-ising_model", action='store_true') #runs the ising model with default values
    parser.add_argument("-external", type=float, default=0) #takes a value to use for the external pull
    parser.add_argument("-alpha", type=float, default=1) #takes a value to use for alpha
    parser.add_argument("-test_ising", action='store_true') #checks whether to run the tests
    
    #Defining the variables
    args = parser.parse_args()
    external = args.external
    alpha = args.alpha
    
    #checks for '-ising_model' flag and creates a population and runs main code if present
    if args.ising_model:
        pop = np.random.choice([-1,1],size=(100,100))
        ising_main(pop, alpha, external)
    
    #checks for '-test_ising' flag and runs tests fucntion if present
    if args.test_ising:
        test_ising()
    


if __name__=="__main__":
    main()


