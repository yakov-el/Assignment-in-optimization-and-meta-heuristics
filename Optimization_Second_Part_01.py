import math
import numpy as np
import random
import matplotlib.pyplot as plt


def genetic_algorithm():
    # initial population - in format of (x,y)
    seeding = [(1.8, 15.2), (2.9, 14.1), (4.5, 12.5), (4.0, 13.0), (5, 12),
               (6.8, 10.2), (7.3, 9.7), (8, 9), (0.1, 16.9), (0, 17),
               (7, 10), (2.9, 14.1), (4.2, 12.8), (0.7, 16.3), (7.9, 9.1)]

    def encoding(number):  # taking the values of x and y and converting them into a joint binary code
        gene = ""  # this string will contain the gene
        for i in range(2):  # iterating over x and y respectively
            g = number[i] * 10  # multiplying by 10 to encode another decimal
            g = math.floor(g)  # we split hours only by 10th hours and not more than that (rounding down)
            if i == 0:  # for x
                binary_g = format(g, '07b') # formatting each var into a binary code of 7 digits
            else:  # for y
                binary_g = format(g, '08b')  # formatting each var into a binary code of 8 digits
            gene += binary_g  # concatenation of the variables into the gene
        return gene

    parents = []  # list of parents
    for seed in seeding:
        parents.append(encoding(seed))  # we take the coordinates and encode them into genes

    def decoding(gene):  # converting a binary gene to x and y in the original form
        x = int(gene[:7], 2) / 10  # the first 8 digits represent x and we count x by hours hence the /10
        y = int(gene[7:], 2) / 10  # the last 8 digits represent y and we count y by hours hence the /10
        return x, y

    def fitness(gene):
        x, y = decoding(gene)  # revert the existing gene into x and y using the decoding function
        if x + y > 17 or x > y:  # if any of the constraints is broken
            return -1  # we chose a negative value in a maximization problem to show incompetence - (min(f(x,y)) = 0)

        else:  # none of the constraints is broken
            return x**0.5 + y**0.5 + (x*y)**0.5

    def one_point_crossover(parent1, parent2):  # creates one point crossover between the parents to create 2 offsprings
        point = random.randint(1, len(parent1) - 1)  # Choose a random crossover point
        child1 = parent1[:point] + parent2[point:]  # crossover of the bits at the crossover point between the 2 parents
        child2 = parent2[:point] + parent1[point:]
        return [child1, child2]

    def mutation(gene):  # the function will create mutations by a given probability
        mutated_string = ""  # this will be the string that will become the new gene whether it was changed or not
        for bit in gene:  # iterating over each bit in the string
            if random.random() <= 1/15:  # Check if mutation occurs based on probability
                # Flip the bit (change 0 to 1 or vice versa)
                if bit == '0':
                   mutated_string += '1'  # if the current value is 0 it will be mutated to be 1
                else:
                   mutated_string += '0'  # if the current value is 1 it will be mutated to be 0
            else:
                mutated_string += bit  # Keep the bit unchanged
        return mutated_string

    parents = []  # a list that contains all of the parents in each generation
    for seed in seeding:
        parents.append(encoding(seed))  # taking the seeds and encode them into our binary code chromosomes
    size = 15  # population size in each generation
    iter = 0  # iteration number
    best_array = np.zeros(1000)   # array that will contain the value of the best solution of each generation
    avg_array = np.zeros(1000)    # array the will contain the value of the average solution of each generation
    variance_array = np.zeros(1000) # array the will contain the value of the STD solution of each generation
    while iter < 1000:  # as long as we didn't cross 1000 generations
        offsprings = []  # list of the offsprings of each generation
        sum_f = 0  # sum values of functions of all of the parents
        parents_scores = np.zeros(15)  # scores of all of the parents
        p = []  # p will contain parents and their calculated value
        for i in range(len(parents)):  # calculating the scores of all parents and their sum
            score = fitness(parents[i])  # calculating the fitness of each parent in the list
            parents_scores[i] = score  # saving the fitness score of each parent within an array
            sum_f += score  # counting the total sum of fitness of all parents
        for j in range(len(parents)):  # calculating the probability of each parent to be selected
            p.append((parents[j], parents_scores[j] / sum_f))  # add parents and their probability to be picked
        p = sorted(p, key=lambda x: x[1], reverse=True)  # now p will be sorted according to probabilities
        best_array[iter] = max(parents_scores)  # fetching the best fitness value of the generation
        avg_array[iter] = sum_f/size  # average value of the population each generation
        variance_array[iter] = np.std(parents_scores)  # calculating STD of all parents
        for index in range(2):
            offsprings.append(p[index][0])  # elitism - we pick the best two solutions from the sorted list
        # breeding
        n = 2  # the population already consists of the best 2 solutions from the last generation (elitism)
        while n < size:  # while we don't have enough children
            # pick first parent by cape function
            roll = np.random.uniform(0, 1)  # rolling the dice
            prob = 0
            first, second = 0, 0  # python asked us to define global values to the parents- temporary definition
            for parent in p:
                prob += parent[1]  # Increment probability
                if prob >= roll:  # if the local parent is picked based on the incremented probability
                    first = parent[0]  # fetching the parent itself
                    partners = p.copy()   # creating a copy of the parents list for the suitable partners options
                    partners.remove(parent)  # remove the first parent from the mating candidates
                    second = random.choice(partners)[0]  # pick second parent by uniform selection
                    break  # Exit loop once a parent is selected
            children = one_point_crossover(first, second)  # breeding
            for child in children:  # running additional test on the newly born children
                child = mutation(child)  # sending the child to add mutations
                # checking if the child is valid (according to our constraints)
                val = fitness(child)
                # checking if the child is valid (according to our constraints) if not val will be -1
                if val >= 0 and len(offsprings) < 15:
                    offsprings.append(child)  # if valid then the child is added
                    n += 1  # the number fo children is updated

        parents = offsprings  # elitism, picking two best solutions (current parents) ans pass them as offsprings
        iter += 1
    print("the optimal solution is: ", decoding(p[0][0]), "\n", "and its value is: ", round(max(parents_scores), 3))
    # Graphing
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))

    ax[0].plot(best_array, label='Best Solution', color='crimson')
    ax[0].set_title('Best Solution Each Generation')
    ax[0].set_xlabel('Generation')
    ax[0].set_ylabel('Best Solution')
    ax[0].set_ylim(bottom=min(best_array)-0.5, top=max(best_array)+0.5)

    ax[1].plot(avg_array, label='Average Result', color='cornflowerblue')
    ax[1].plot(label='Trendline', color='aqua')  # Plotting the trend line
    ax[1].set_title('Average Result Each Generation')
    ax[1].set_xlabel('Generation')
    ax[1].set_ylabel('Average Result')

    # Adding trend line to the STD graph
    x_values = np.arange(len(variance_array))  # Generating x values (generation numbers)
    coefficients = np.polyfit(x_values, variance_array, 2)  # Fitting a 2nd-degree polynomial curve
    trendline = np.polyval(coefficients, x_values)  # Calculating y values for the trend line

    ax[2].plot(variance_array, label='Standard Deviation', color='indianred')
    ax[2].plot(trendline, label='Trendline', linestyle='--', color='maroon')  # Plotting the trend line
    ax[2].set_title('Standard Deviation Each Generation')
    ax[2].set_xlabel('Generation')
    ax[2].set_ylabel('Standard Deviation')
    plt.tight_layout()
    plt.show()


genetic_algorithm()


