# Stores methods that deal with extracting the raw data from the json file imported
# to generate the basic experiment, participant and trial objects.
import numpy as np

#def normalising_constant(base_model):
#    max_dist = [-1, -1, 1, 1, 1]
#    possibilities = [1, 0.5, 0, -0.5, -1]
#
#    base_model = base_model.flatten()
#
#    max_dist_model = np.zeros(len(base_model))
#    for i in range(len(max_dist_model)):
#        max_dist_model[i] = max_dist[possibilities.index(base_model[i])]
#
#    return euclidean_distance(max_dist_model, base_model)[0]


def signChangeCode(prior, post):
    value = None
     # Value logic
    if prior == post:
        # No change
        value = 0
    elif prior == 1 and post == -1:
        # Pos to neg
        value = 1
    elif prior == -1 and post == 1:
        # Neg to pos
        value = 2
    elif post == 0:
        # Link to no link
        value = 3
    elif prior == 0:
        # No link to link
        value = 4

    return value


def corGivenCond(truthCond, postTruthCond):
    if np.sum(truthCond) == 0:
        return np.nan
    else:
        return np.sum(postTruthCond) / np.sum(truthCond)


def getModelOrder(name):

    contractions = name.split('_')[1]
    
    first = contractions[0: 3]
    second = contractions[3: 6]
    third = contractions[6: 9]
    
    if "Cri" in contractions:
        names = ['Police action', 'Population happiness', 'Crime rate']
        name = 'crime'
    elif "Hou" in contractions:
        names = ['House Prices', 'Desirability', 'Population Density']
        name = 'estate'
    elif "Vir" in contractions:
        names = ['Virus Cases', 'Stock Prices', 'Lockdown Measures']
        name = 'finance'
    order = []
    idx = 0
    while True:
        if len(order) == 0 and first == names[idx][0: 3]:
            order.append(names[idx])
        elif len(order) == 1 and second == names[idx][0: 3]:
            order.append(names[idx])
        elif len(order) == 2 and third == names[idx][0: 3]:
            order.append(names[idx])
            break

        idx += 1
        if idx > 2:
            idx = 0

    return order, name


def extractTruthModel(name, presets):
    scenarios = ['crime', 'finance', 'estate']

    if len(name.split('_')) == 2 and name.split('_')[0] in scenarios:
        m = presets[name[:-2]][int(name[-1])]
    elif name.split('_')[0] in scenarios:
        modelName = name.split('_')[0] + '_' + name.split('_')[-1]
        m = presets[modelName[:-2]][int(modelName[-1])]
    else:
        m = presets[name]

    gt_model = [m[3], m[6],
                m[1], m[7],
                m[2], m[5]]

    order = m[9:12]

    #print(m)
    return gt_model, order


def vecToDict(order, report):
    varLabels = ['X', 'Y', 'Z']
    variables = {}
    model = {}
    for i in range(3):
        # Assign variable name to letter
        variables[varLabels[i]] = order[i]

        # Generate model report as dictionary
        model[order[i]] = {}
        for j in range(3):
            if j != i:
                if not model[order[i]]:
                    model[order[i]][order[j]] = report[i*2]
                else:
                    model[order[i]][order[j]] = report[i*2+1]

    return model, variables


def dictToVec(modelDict):
    #print(modelDict.keys())
    if 'Police action' in modelDict.keys():
        order = ['Police action', 'Crime rate', 'Population happiness']
    elif 'House Prices' in modelDict.keys():
        order = ['House Prices', 'Desirability', 'Population Density']
    elif 'Virus Cases' in modelDict.keys():
        order = ['Virus Cases', 'Stock Prices', 'Lockdown Measures']
    elif 'one' in modelDict.keys():
        order = ['one', 'two', 'three']
    elif 'effect' in modelDict.keys():
        order = ['cause1', 'cause2', 'effect']
    elif 'cause' in modelDict.keys():
        order = ['cause', 'effect1', 'effect2']
    else:
        order = ['Blue', 'Red', 'Green']

    #print(order)  
    report = [
        modelDict[order[0]][order[1]],
        modelDict[order[0]][order[2]],
        modelDict[order[1]][order[0]],
        modelDict[order[1]][order[2]],
        modelDict[order[2]][order[0]],
        modelDict[order[2]][order[1]]]

    return report, order


def remapDict(stdDict, dictModel):
    keys = [k for k in stdDict.keys()]

    outDict = {
            keys[0]: {
                keys[1] : dictModel[stdDict[keys[0]]][stdDict[keys[1]]],
                keys[2] : dictModel[stdDict[keys[0]]][stdDict[keys[2]]]
            },
            keys[1]: {
                keys[0] : dictModel[stdDict[keys[1]]][stdDict[keys[0]]],
                keys[2] : dictModel[stdDict[keys[1]]][stdDict[keys[2]]]
            },
            keys[2]: {
                keys[0] : dictModel[stdDict[keys[2]]][stdDict[keys[0]]],
                keys[1] : dictModel[stdDict[keys[2]]][stdDict[keys[1]]]
            }
    }
    return outDict


def standardiseModel(name, dictModel):
    # Different algorithm for each model
    modelType = name[:-2]
    vecModel = dictToVec(dictModel)
    if modelType == 'pos_chain' or modelType == 'confound' or modelType == 'dampened':
        stdDict = {
            'one' : None,
            'two' : None,
            'three' : None
        }
        newModelDict = {
            'one' : None,
            'two' : None,
            'three' : None
        }
        
        for k in dictModel.keys():
            q = np.array([i for i in dictModel[k].values()]).sum()
            if q == 0:
                stdDict['three'] = k
                newModelDict['three'] = {
                    'one' : 0,
                    'two' : 0
                }
            
        for k in dictModel.keys():
            if k in stdDict.values():
                continue
            if abs(dictModel[k][stdDict['three']]) == 1:
                stdDict['two'] = k
                newModelDict['two'] = {
                    'one' : 0,
                    'three' : dictModel[k][stdDict['three']]
                }
                

        for k in dictModel.keys():
            if k in stdDict.values():
                continue
            if abs(dictModel[k][stdDict['two']]) == 1:
                stdDict['one'] = k
                newModelDict['one'] = {
                    'two' : dictModel[k][stdDict['two']],
                    'three' : dictModel[k][stdDict['three']]
                }
    
    elif modelType == 'collider':
        stdDict = {
            'cause1': None,
            'cause2': None,
            'effect': None
        }
        newModelDict = {
            'cause1': None,
            'cause2': None,
            'effect': None
        }

        for k in dictModel.keys():
            q = np.array([i for i in dictModel[k].values()]).sum()
            if q == 0:
                stdDict['effect'] = k
                newModelDict['effect'] = {
                    'cause1' : 0,
                    'cause2' : 0
                }
            
        for k in dictModel.keys():
            if k in stdDict.values():
                continue
            if dictModel[k][stdDict['effect']] != 0:
                stdDict['cause1'] = k
                newModelDict['cause1'] = {
                    'cause2' : 0,
                    'effect' : dictModel[k][stdDict['effect']]
                }
                
        # Look for last unassigned key value
        for k in dictModel.keys():
            if k not in stdDict.values():
                stdDict['cause2'] = k
                newModelDict['cause2'] = {
                    'cause1' : 0,
                    'effect' : dictModel[k][stdDict['effect']]
                }

    elif modelType == 'ccause':
        stdDict = {
            'cause': None,
            'effect1': None,
            'effect2': None
        }
        newModelDict = {
            'cause': None,
            'effect1': None,
            'effect2': None
        }

        for k in dictModel.keys():
            q = np.array([i for i in dictModel[k].values()]).sum()
            if q == 0:
                stdDict['effect1'] = k
                newModelDict['effect1'] = {
                    'cause' : 0,
                    'effect2' : 0                    
                }
            
        for k in dictModel.keys():
            if k in stdDict.values():
                continue
            q = np.array([i for i in dictModel[k].values()]).sum()
            if q == 0 and k != stdDict['effect1']:
                stdDict['effect2'] = k
                newModelDict['effect2'] = {
                    'cause' : 0,
                    'effect1' : 0
                }
                
        # Look for last unassigned key value
        for k in dictModel.keys():
            if k not in stdDict.values():
                stdDict['cause'] = k
                newModelDict['cause'] = {
                    'effect1' : dictModel[k][stdDict['effect1']],
                    'effect2' : dictModel[k][stdDict['effect2']]
                }
        
        #print(newModelDict, '\n')

    return stdDict, newModelDict


def parse_links_str(link_string):
    if len(link_string) < 3:
        return link_string
    else:
        first_cut = link_string.split(';">')[-1].split('</f')[0]
        if len(first_cut) < 3:
            return first_cut
        else:
            return first_cut.split('">')[-1]
    

def links_list_to_ndarray(links_list, posterior):
    out_array = np.zeros((len(links_list[0]), len(links_list)))

    current_estimate = np.empty(posterior.shape)
    current_estimate[:] = np.nan

    current = parse_links_str(links_list[0][0]).strip()
    for j in range(out_array.shape[1]):
        for i in range(out_array.shape[0]):
            current = parse_links_str(links_list[j][i]).strip()
            far_idx = i+4 if i + 4 < out_array.shape[0]-1 else out_array.shape[0]-1
            next_value =  parse_links_str(links_list[j][far_idx]).strip()
            
            if current == '?':
                out_array[i, j] = np.nan
            elif int(current) != int(next_value):
                out_array[i, j] = np.nan
            else:
                if int(current) != current_estimate[j]:
                    out_array[i, j] = int(current) / 2
                    current_estimate[j] = int(current)
                else:
                    out_array[i, j] = np.nan
          
    #out_array[out_array.shape[0]-1, :] = posterior
    #print(out_array[out_array.shape[0]-1, :], posterior)
    return out_array



