import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt

from util import extractData, extractLabels

if (len(sys.argv) != 13):
    print("Usage : ./search -d <input file original space> -q <query file original space> --l1 <labels of input dataset> --l2 <labels of query dataset> -o <output file> - EMD")
    sys.exit(-1)

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "d:q:", ["l1=", "l2=", "EMD="])
except getopt.GetoptError:
    print("WTF")
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

print (input_file_original_space, query_file_original_space, labels_of_input_dataset, labels_of_query_dataset, output_file)