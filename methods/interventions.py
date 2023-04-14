# Generic methods that deal with the compilation of intervention data.
# Note: may be renamed later on if the methods are generalisable to other problems.

import numpy as np

# Turns a 0 and 1 array into a numpy Boolean array
def makeBoolean(array):
    return (np.array(array) == 1)


# Splits a np.where array in as many sequences of consecutive indices as are present in it.
# This works up to a step, which size defaults to 3, i.e. if the index difference is less than 3
# it will count it as one sequence and fill in the missing indices
def splitIdxArray(array, step=2):
    final_list = [[]]
    flistIdx = 0
    step = 2
    
    if len(array) == 1:
        return [[array[0]]]
    elif len(array) == 0:
        return []

    for i in np.arange(1, len(array)):

        diff = abs(array[i] - array[i-1])

        if i == 1:
            final_list[flistIdx].append(array[i-1])

        if diff > step:
            final_list.append([array[i]])
            flistIdx += 1
        elif diff > 1:
            for j in np.arange(1, diff + 1):
                final_list[flistIdx].append(array[i-1]+j) 
        else:
            final_list[flistIdx].append(array[i])

    return final_list
