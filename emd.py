import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
import math

from numpy.core.numeric import Inf

from scipy.optimize import linprog
from util import extractData, extractLabels

cluster_size = (7, 7)


if (len(sys.argv) != 13):
    print("Usage : ./search -d <input file original space> -q <query file original space> --l1 <labels of input dataset> --l2 <labels of query dataset> -o <output file> - EMD")
    sys.exit(-1)

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "d:q:o:", ["l1=", "l2=", "EMD="])
except getopt.GetoptError:
    print("Usage : ./search -d <input file original space> -q <query file original space> --l1 <labels of input dataset> --l2 <labels of query dataset> -o <output file> - EMD")
    sys.exit(-1)

for option, argument in opts:
    if option == "-d":
        input_file_original_space = argument
    elif option == "-q":
        query_file_original_space = argument
    elif option in ("--l1"):
        labels_of_input_dataset = argument
    elif option in ("--l2"):
        labels_of_query_dataset = argument
    elif option ==  "-o":
        output_file = argument

train_data, x, y = extractData(input_file_original_space)
test_data, x, y = extractData(query_file_original_space)
train_labels = extractLabels(labels_of_input_dataset)
test_labels = extractLabels(labels_of_query_dataset)

# 8elei na tsekarw oti oles oi eikones exoun to idio sinoliko baros 
# profanws den 8a paradwsw me auto alla 8a to kanw na dw kai poso einai kserwgw antrika

def breakToBlocks(arr, rows, columns):

    total = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            total = total + arr[i][j]

    # print(total)

    arr = arr / (total+1) # giati xreiazetai + 1?

    # total = 0
    # for i in range(arr.shape[0]):
    #     for j in range(arr.shape[1]):
    #         total = total + arr[i][j]

    # print(total)

    h, w = arr.shape
    assert h % rows == 0, "{} rows cannot be divised by that number {}".format(h, rows)
    assert w % columns == 0, "{} columns cannot be divised by that number {}".format(w, columns)
    # print (rows, columns)
    newArr = arr.reshape(h//rows, rows, -1, columns).swapaxes(1,2).reshape(-1, rows, columns)
    # print (len(newArr))

    new_col = []

    # take the first pixel of the cluster
    for i in range(int(28/cluster_size[0])):
        for j in range(int(28/cluster_size[1])):
            new_col.append((i*cluster_size[0], j*cluster_size[1]))

    return newArr, new_col


def euclideanDistance(coordinate1, coordinate2):
    return math.sqrt(pow(coordinate1[0] - coordinate2[0] , 2) + pow(coordinate1[1] - coordinate2[1] , 2))


def weightOfCluster(arr):
    total = 0
    # print(arr.shape[0], arr.shape[1])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            total = total + arr[i][j]

    return total



if __name__=="__main__":

    train_data = train_data[:40]
    train_labels = train_labels[:40]

    test_data = test_data[:80]
    test_labels = test_labels[:80]

    for train_number in range(len(train_data)):

        minimum = [float(Inf), float(Inf)]
        for test_number in range(len(test_data)):

            broken_train, coordinates = breakToBlocks(train_data[train_number], cluster_size[0], cluster_size[1])
            broken_query, coordinates = breakToBlocks(test_data[test_number], cluster_size[0], cluster_size[1])

            # dij array
            d = [[0 for x in range(len(coordinates))] for y in range(len(coordinates))]

            for i in range(len(coordinates)):
                for j in range(len(coordinates)):
                    d[i][j] = euclideanDistance(coordinates[i], coordinates[j])


            obj = []

            # left constraints len(array) = 2 * clusters and each len(subarray) = clusters*clusters
            left_constraints_ineq = [[0 for x in range( pow(len(coordinates), 2) )] for y in range( 2 * len(coordinates) )] # sum fij <= wpi

            # right constraints len(array) = 2 * clusters
            right_constraints_ineq = [0 for x in range( 2 * len(coordinates) )] # sum fij <= wqi


            for i in range( len(coordinates) ):
                for j in range( len(coordinates) ):  # len(coordinates) == 16 edw px apla to pli8os twn cluster einai
                    left_constraints_ineq[i][i*16 + j] = 1

            for i in range( len(coordinates) ):
                for j in range( len(coordinates) ):  # len(coordinates) == 16 edw px apla to pli8os twn cluster einai
                    left_constraints_ineq[len(coordinates) + i][i*16 + j] = 1


            for i in range(len(coordinates)):
                for j in range(len(coordinates)):
                    obj.append(d[i][j])


            for i in range( len(coordinates) ):
                right_constraints_ineq[i] = weightOfCluster(broken_train[i]) # sum fij <= wpi

            for i in range( len(coordinates) ):
                right_constraints_ineq[len(coordinates) + i] = weightOfCluster(broken_query[i]) # sum fij <= wqi


            left_constraints_eq = [[1 for x in range( pow(len(coordinates), 2) )]]

            right_constraints_eq = min( weightOfCluster(train_data[train_number]) , weightOfCluster(test_data[test_number]) )

            opt = linprog(c=obj, A_ub=left_constraints_ineq, b_ub=right_constraints_ineq,
                        A_eq=left_constraints_eq, b_eq=right_constraints_eq, method="simplex"
            )

            if (opt.fun < minimum[0]):
                minimum[0] = opt.fun
                minimum[1] = test_number

        
        print (train_labels[train_number], test_labels[minimum[1]])