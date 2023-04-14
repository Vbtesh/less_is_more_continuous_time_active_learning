from methods.extraction import dictToVec, standardiseModel, remapDict, corGivenCond, links_list_to_ndarray
from methods.interventions import splitIdxArray, makeBoolean
from methods.modelling_methods import logical_to_index_actions
from methods.graph_transformations import causality_matrix, causality_vector, find_roots, find_indirect_errors
import numpy as np
from scipy.spatial.distance import pdist
from classes.intervention import Intervention



class Trial():
    def __init__(self, name, utid, type_trial, report, order_in_trial, contTime=True, model=None, xyz=None, valueLists=None, qual=None, cond_distance=False):
        self.name = name
        self.type_trial = type_trial
        self.utid = utid
        #print(name)
        self.report = report # standardised for labelled and unlabelled complex models

        
        if qual:
            self.sense = qual[0]
            self.rationale = qual[1]
        else:
            self.sense = None
            self.rationale = None

        if not contTime:
            self.type = 'genModel'
            self.model, self.trueModelOrder = dictToVec(self.report)
            self.order_in_trial = order_in_trial

        else:
            self.type = 'contTime'
            self.model = model # formal model with coefficients
            #print('model in trial:', model)
            self.xyz = xyz # Dictionary mapping letters to variable names
            self.order_in_trial = order_in_trial

            # Set up prior properties
            self.prior_dict = None
            self.prior_vec = None
            self.prior_matrix = None

            # Generate formal reports
            gtModel, gtOrder = dictToVec(self.model)
            self.trueOrigin = np.array(gtModel)
            self.trueOriginOrder = gtOrder

            rModel, rOrder = dictToVec(self.report)
            self.postModel = np.array(rModel)
            self.postModelOrder = rOrder

            if self.name.split('_')[0] not in ['crime', 'estate', 'finance']:
                stdDict, newModelDict = standardiseModel(self.name, self.model)                
                self.modelStd = newModelDict
                self.postModelStd = remapDict(stdDict, self.report)
                
            else:
                self.modelStd = self.model
                self.postModelStd = self.report

            # Generate formal reports
            # Ground truth
            gtModel, gtOrder = dictToVec(self.modelStd)
            self.trueModel = np.array(gtModel)
            report = self.trueModel
            self.trueSignModel = [
                1 if report[0] > 0 else -1 if report[0] < 0 else 0,
                1 if report[1] > 0 else -1 if report[1] < 0 else 0,
                1 if report[2] > 0 else -1 if report[2] < 0 else 0,
                1 if report[3] > 0 else -1 if report[3] < 0 else 0,
                1 if report[4] > 0 else -1 if report[4] < 0 else 0,
                1 if report[5] > 0 else -1 if report[5] < 0 else 0,
            ]
            self.trueModelOrder = gtOrder
            # Ground truth matrix
            self.ground_truth_causality_matrix = causality_matrix(self.trueModel)

            if self.name[:-2] == 'pos_chain':
                if self.trueModel.sum() == -2:
                    self.name = 'neg_chain_' + self.name[-1]
            elif self.name[:-2] == 'confound':
                if self.trueModel.sum() < 0:
                    self.name = 'neg_confound_' + self.name[-1]
                else:
                    self.name = 'pos_confound_' + self.name[-1]

            # Posterior judgement, standardised variables
            rModel, rOrder = dictToVec(self.postModelStd)
            self.postModel = np.array(rModel)
            self.postModelOrder = rOrder
            report = self.postModel
            self.postSignModel = [
                1 if report[0] > 0 else -1 if report[0] < 0 else 0,
                1 if report[1] > 0 else -1 if report[1] < 0 else 0,
                1 if report[2] > 0 else -1 if report[2] < 0 else 0,
                1 if report[3] > 0 else -1 if report[3] < 0 else 0,
                1 if report[4] > 0 else -1 if report[4] < 0 else 0,
                1 if report[5] > 0 else -1 if report[5] < 0 else 0,
            ]
            # Posterior judgement, as matrix
            self.posterior_causal_matrix = causality_matrix(self.postModel)

            # Number of indirect links errors
            errors, indirect_links_loc, out = find_indirect_errors(np.array(self.trueSignModel), np.array(self.postSignModel))
            if len(errors) > 0:
                self.indirect_links_errors = errors.sum()
            else:
                self.indirect_links_errors = 0

            if indirect_links_loc.sum() > 0:
                self.num_indirect_links = indirect_links_loc.sum()
            else:
                self.num_indirect_links = 0
            
            # Posterior judgement, unstandardised variables
            oModel, oOrder = dictToVec(self.report)
            self.postOrigin = np.array(oModel)
            self.postOriginOrder = oOrder
            #print(self.postOrigin)
            
            if cond_distance:
                self.cond_dist = cond_distance
            else:
                self.cond_dist = None

            # Call the calc metrics method
            self.calcMetrics()

        if valueLists:
            self.times = valueLists['times']
            # True variables values (as in experiment)
            # Values
            self.xValExp = valueLists['xVals']
            self.yValExp = valueLists['yVals']
            self.zValExp = valueLists['zVals']
            # Interventions
            self.xIntExp = valueLists['xInt']
            self.yIntExp = valueLists['yInt']
            self.zIntExp = valueLists['zInt']

            # Model history, has to be adjusted complex model
            self.XonY = valueLists['XonY']
            self.XonZ = valueLists['XonZ']
            self.YonX = valueLists['YonX']
            self.YonZ = valueLists['YonZ']
            self.ZonX = valueLists['ZonX']
            self.ZonY = valueLists['ZonY']

            # Model params
            self.trial_params = {
                'N': len(self.times),
                'K': 3,
                'theta': 0.5,
                'dt': 0.2,
                'sigma': 1
            }

            # Standardised variables
            if self.name.split('_')[0] not in ['crime', 'estate', 'finance']:
                inv_stdDict = {v: k for k, v in stdDict.items()}
                X = inv_stdDict[self.xyz['X']]
                Y = inv_stdDict[self.xyz['Y']]
                Z = inv_stdDict[self.xyz['Z']]

                self.mapping = {key:inv_stdDict[self.xyz[key]] for key in self.xyz.keys()}
            else:
                X = self.xyz['X']
                Y = self.xyz['Y']
                Z = self.xyz['Z']

                self.mapping = self.xyz
                
            self.values = {
                X: valueLists['xVals'],
                Y: valueLists['yVals'],
                Z: valueLists['zVals']
            }
            self.int = {
                X: valueLists['xInt'],
                Y: valueLists['yInt'],
                Z: valueLists['zInt']
            }
            self.links_hist_dict = {
                X: {
                    Y: self.XonY,
                    Z: self.XonZ
                },
                Y: {
                    X: self.YonX,
                    Z: self.YonZ
                },
                Z: { 
                    X: self.ZonX,
                    Y: self.ZonY
                }
            }
            links_hist, _ = dictToVec(self.links_hist_dict)
            self.links_hist_std = links_list_to_ndarray(links_hist, self.postModel)





    def calcMetrics(self):
        # Euclidian distance
        self.normEuc = 1-pdist(np.stack((self.trueModel, self.postModel)))[0] / np.linalg.norm(abs(np.array(self.trueModel)) + 2*np.ones((1, 6)))
        # Hamming distance
        self.hammingFull = 1-pdist(np.stack((self.trueModel, self.postModel)), 'hamming')[0]
        
        # Just link sign
        self.hamming = 1-pdist(np.stack((self.trueSignModel, self.postSignModel)), 'hamming')[0]
        
        
        
        # Binary correctness of the generative model
        if np.array_equal(self.trueModel, self.postModel):
            self.correct = 1
        else:
            self.correct = 0

        # Correctness for negative and positive links
        self.negCorr = corGivenCond(self.trueModel < 0, self.postModel[self.trueModel < 0] < 0)
        self.negNum = np.sum(self.trueModel < 0)

        self.posCorr = corGivenCond(self.trueModel > 0, self.postModel[self.trueModel > 0] > 0)
        self.posNum = np.sum(self.trueModel > 0)

        # Correctness for weak and strong links given the sign
        self.weakPosCorr = corGivenCond(self.trueModel == 0.5, self.postModel[self.trueModel == 0.5] == 0.5)
        self.weakPosNum = np.sum(self.trueModel == 0.5)
        self.stgPosCorr = corGivenCond(self.trueModel == 1, self.postModel[self.trueModel == 1] == 1)
        self.stgPosNum = np.sum(self.trueModel == 1)

        self.weakNegCorr = corGivenCond(self.trueModel == -0.5, self.postModel[self.trueModel == -0.5] == -0.5)
        self.weakNegNum = np.sum(self.trueModel == -0.5)
        self.stgNegCorr = corGivenCond(self.trueModel == -1, self.postModel[self.trueModel == -1] == -1)
        self.stgNegNum = np.sum(self.trueModel == -1)

        # Correctness for weak and strong links
        condition_true = np.logical_or(self.trueModel == 0.5, self.trueModel == -0.5)
        self.weakCorr = corGivenCond(condition_true, np.logical_or(self.postModel[condition_true] == 0.5, self.postModel[condition_true] == -0.5))
        
        if self.weakPosNum + self.weakNegNum == 0:
            self.weakCorr = np.nan
        else:
            weakCorr_den = self.weakPosNum + self.weakNegNum
            if self.weakPosNum == 0:
                weakCorr_num = self.weakNegCorr * self.weakNegNum
            elif self.weakNegNum == 0:
                weakCorr_num = self.weakPosCorr * self.weakPosNum
            else:
                weakCorr_num = self.weakNegCorr * self.weakNegNum + self.weakPosCorr * self.weakPosNum
                
            self.weakCorr = weakCorr_num / weakCorr_den

        self.weakNum = np.sum(abs(self.trueModel) == 0.5)

        condition_true = np.logical_or(self.trueModel == 1, self.trueModel == -1)
        self.stgCorr = corGivenCond(condition_true, np.logical_or(self.postModel[condition_true] == 1, self.postModel[condition_true] == -1))
        
        if self.stgPosNum + self.stgNegNum == 0:
            self.stgCorr = np.nan
        else:
            stgCorr_den = self.stgPosNum + self.stgNegNum
            if self.stgPosNum == 0:
                stgCorr_num = self.stgNegCorr * self.stgNegNum
            elif self.weakNegNum == 0:
                stgCorr_num = self.stgPosCorr * self.stgPosNum
            else:
                stgCorr_num = self.stgNegCorr * self.stgNegNum + self.stgPosCorr * self.stgPosNum
                
            self.stgCorr = stgCorr_num / stgCorr_den
            
        self.stgNum = np.sum(abs(self.trueModel) == 1)

        # Correctness for no links
        self.nullCorr = corGivenCond(self.trueModel == 0, self.postModel[self.trueModel == 0] == 0)
        self.nullNum = np.sum(self.trueModel == 0)
        
        self.nonNullCorr = corGivenCond(self.trueModel != 0, self.postModel[self.trueModel != 0] != 0)
        self.nullNum = np.sum(self.trueModel != 0)


    def calcRoots(self):
        order = self.postModelOrder
        self.prior_roots = None
        if type(self.prior_matrix) == np.ndarray:
            self.prior_roots = find_roots(self.prior_matrix)
            self.prior_root_map = {k: v for k, v in zip(order, self.prior_roots)}

        self.posterior_roots = find_roots(self.posterior_causal_matrix)
        self.posterior_root_map = {k: v for k, v in zip(order, self.posterior_roots)}
        self.ground_truth_roots = find_roots(self.ground_truth_causality_matrix)
        self.gt_root_map = {k: v for k, v in zip(order, self.ground_truth_roots)}

        cond = self.type_trial
        if cond in ['congruent', 'incongruent']:
            
            root_prior = self.prior_roots
            root_gt = self.ground_truth_roots
        # Update interventions
        for inter in self.interventions:
            # Add the label corresponding the inter name (inter.var)
            if self.prior_roots:
                inter.add_var_types(self.posterior_root_map[inter.var],
                                    self.gt_root_map[inter.var],
                                    self.prior_root_map[inter.var])
            else:
                inter.add_var_types(self.posterior_root_map[inter.var],
                                    self.gt_root_map[inter.var])


    def calcInterventions(self):
        trial_length = len(self.xIntExp)
        # To Do
        ## Number of interventions : sum of lengths of splitIdxArray output for each variable
        
        int_xyz = [splitIdxArray(np.where(makeBoolean(self.xIntExp))[0]), 
                   splitIdxArray(np.where(makeBoolean(self.yIntExp))[0]), 
                   splitIdxArray(np.where(makeBoolean(self.zIntExp))[0])]

        self.num_int = len(int_xyz[0]) + len(int_xyz[1]) + len(int_xyz[2])

        ## Order of interventions: compare first and last indices for each variable
        ## Intervened time vs observed time
        # Generate intervention list

        int_np = np.zeros((len(int_xyz), len(self.int[list(self.int.keys())[0]])))
        for i, v in enumerate(self.int.values()):
            int_np[i, :] = v

        int_time = 0
        self.interventions = []
        inter_starts = []

        for k, v in self.int.items():

            inter_list = splitIdxArray(np.where(makeBoolean(v))[0])
            values = np.array(self.values[k])

            

            for i, inter in enumerate(inter_list):
                if not inter:
                    continue
                # Compile intervention time
                int_time += len(inter)

                if (len(inter) == 1 or len(inter) == 2) and inter[-1] < trial_length-1:
                    a_inter = inter + [inter[-1]+1]
                    inter_values = values[a_inter]
                else:
                    a_inter = inter
                    inter_values = values[a_inter]

                # Find var_pos in x y z to var name mapping
                if k == self.mapping['X']:
                    var_pos = 0
                elif k == self.mapping['Y']:
                    var_pos = 1
                else:
                    var_pos = 2

                graph = self.modelStd.copy()
    
                effects_estimates = self.postModelStd[k]

                network_states = {k:v[a_inter[0]:a_inter[-1]+1] for k, v in self.values.items()}

                int_np_0 = np.where(int_np > 0)
                where_new_int = [i for i in np.arange(a_inter[-1] + 1, a_inter[-1] + 15) if i in int_np_0[1]]
                if where_new_int:
                    end = where_new_int[0]
                elif a_inter[-1]+15 > len(self.times) - 1:
                    end = len(self.times) - 1
                else:
                    end = a_inter[-1]+15
                network_states_wide = {k:v[a_inter[0]-1:end] for k, v in self.values.items()}
                self.interventions.append(Intervention(var=k, 
                                                       var_inter_idx=i+1,
                                                       indices=a_inter, 
                                                       values=inter_values, 
                                                       var_pos = var_pos,
                                                       trial_length=trial_length,
                                                       utid=self.utid,
                                                       graph=graph,
                                                       gt_order=self.trueModelOrder,
                                                       gt_vec=self.trueModel,
                                                       effects_estimates=effects_estimates,
                                                       network_states=network_states,
                                                       network_states_wide=network_states_wide,
                                                       graph_prior=self.prior_dict))

                inter_starts.append(inter[0])
            
        inter_starts = np.array(inter_starts)
        inter_starts_decay = np.array([i for i in inter_starts])
        inter_order = np.zeros(len(inter_starts))
        
        order = 1
        while len(inter_starts_decay) > 0:
            min_start = np.min(inter_starts_decay)
            min_idx = np.where(inter_starts == min_start)[0]
            inter_order[min_idx] = order
            order += 1
            inter_starts_decay = np.delete(inter_starts_decay, np.argmin(inter_starts_decay))

        # Add inter order as num
        for i, inter in enumerate(self.interventions):
            inter.add_num(inter_order[i])
        

        # Compute the ratio between intervened time and total time of trial
        self.intervened_time = int_time / trial_length


        # Compile value and int data into a numpy array for further display
        self.data = np.zeros((len(self.times), 3))
        self.inters_logical = np.zeros((len(self.times), 3))
        for i in range(3):
            label = self.trueModelOrder[i]
            self.data[:, i] = self.values[label]
            self.inters_logical[:, i] = self.int[label]  

        # Finally create a fittable intervention array
        self.inters_fit_logical = np.zeros(self.inters_logical.shape)
        for inter in self.interventions:
            col_num = self.trueModelOrder.index(inter.var)
            idx = inter.indices
            self.inters_fit_logical[idx, col_num] = 1     

        # Finally convert both to indices
        self.inters = logical_to_index_actions(self.inters_logical)
        self.inters_fit = logical_to_index_actions(self.inters_fit_logical) 

        


