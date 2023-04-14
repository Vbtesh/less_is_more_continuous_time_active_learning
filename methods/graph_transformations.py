import numpy as np

def causality_vector(causality_matrix, dim2=None):
    s = causality_matrix.shape[0]**2 - causality_matrix.shape[0]
    if dim2:
        causal_vec = np.zeros((s, dim2))
    else:
        causal_vec = np.zeros(s)
    idx = 0
    for i in range(causality_matrix.shape[0]):
        for j in range(causality_matrix.shape[1]):
            if i != j:
                if dim2:
                    causal_vec[idx, :] = causality_matrix[i, j]
                else:
                    causal_vec[idx] = causality_matrix[i, j]
                idx += 1
    return causal_vec


def causality_matrix(causality_vector, fill_diag=1):
    num_var = int((1 + np.sqrt(1 + 4*len(causality_vector))) / 2)
    causal_mat = fill_diag * np.ones((num_var, num_var))

    idx = 0
    for i in range(num_var):
        for j in range(num_var):
            if i != j:
                causal_mat[i, j] = causality_vector[idx] 
                idx += 1
            
    return causal_mat


## Roots analysis
def find_roots(graph):
    power = np.abs(graph).sum(axis=1) - 1
    determination = np.abs(graph).sum(axis=0) - 1

    power = (np.abs(graph) > 0).sum(axis=1) - 1
    determination = (np.abs(graph) > 0).sum(axis=0) - 1
 
    roots_power = np.where(power == power.max())[0]
    roots_determination = np.where(determination == determination.min())[0]

    types = ['leaf' for i in range(graph.shape[0])]

    for i, r in enumerate(roots_power):
        if r in roots_determination:
            types[r] = 'root'
        #else:
        #    types[r] = 'root_weak'

    return types


## Indirect effects
def indirect_effects_vector(graph):
    out = np.array([graph[1]*graph[5], 
                    graph[0]*graph[3], 
                    graph[3]*graph[4], 
                    graph[2]*graph[1], 
                    graph[5]*graph[2], 
                    graph[4]*graph[0]])
    return out


def find_indirect_errors(gt, p):
    ie_gt = indirect_effects_vector(gt)
    gt_out = ie_gt + gt
    gt_out[np.abs(gt_out) > 1] = gt_out[np.abs(gt_out) > 1] / gt_out[np.abs(gt_out) > 1]

    return p[gt_out != gt] != gt[gt_out != gt], gt_out != gt, gt_out