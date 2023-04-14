import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from methods.extraction import dictToVec
from scipy.spatial.distance import pdist
from methods.graph_transformations import causality_matrix


class Participant():
    def __init__(self, pid, experiment, condition, demographics, pilot, diff_order=None, label_order=None, reports=None, trials=None):
        self.pid = pid
        self.cond = condition
        self.experiment = experiment

        self.diff_order = diff_order
        self.label_order = label_order

        # Demographics: has to be a dictionary
        self.age = int(demographics['age'])
        self.gender = demographics['gender']
        self.causalFam = demographics['causal_fam']
        self.activity = demographics['activity']
        self.graphSent = demographics['graph_fb']
        self.modelSent = demographics['loopy_fb']
        self.pilot = pilot

        self.codeDemo()

        self.reports = []
        self.trials = []
        if reports:
            self.reports = reports
        if trials:
            self.trials = trials

    def codeDemo(self):
        # Gender
        if self.gender == 'female':
            self.gender = 0
        else: 
            self.gender = 1

        # Conditon
        if self.cond == 'label':
            self.cond = 0
        else:
            self.cond = 1

        # Causal Familiarity
        if self.causalFam == 'never':
            self.causalFam = 0
        elif self.causalFam == 'once':
            self.causalFam = 1
        elif self.causalFam == 'somewhat':
            self.causalFam = 2
        elif self.causalFam == 'familiar':
            self.causalFam = 3
        elif self.causalFam == 'extreme':
            self.causalFam = 4

        # Graph Sentiment
        if self.modelSent == 'loopy-hate':
            self.modelSent = 0
        elif self.modelSent == 'loopy-annoy':
            self.modelSent = 1
        elif self.modelSent == 'loopy-indif':
            self.modelSent = 2
        elif self.modelSent == 'loopy-like':
            self.modelSent = 3
        elif self.modelSent == 'loopy-love':
            self.modelSent = 4

        # Dynamic model sentiment
        if self.graphSent == 'graph-hate':
            self.graphSent = 0
        elif self.graphSent == 'graph-annoy':
            self.graphSent = 1
        elif self.graphSent == 'graph-indif':
            self.graphSent = 2
        elif self.graphSent == 'graph-like':
            self.graphSent = 3
        elif self.graphSent == 'graph-love':
            self.graphSent = 4

    def addTrials(self, trialIn):
        if type(trialIn) == list:
            self.trials += trialIn
        else:
            self.trials.append(trialIn)
        # Rerun any trials data processing methods (to be defined)
        self.compileResults()

    def addReports(self, reportIn):
        if type(reportIn) == list:
            self.reports += reportIn
        else:
            self.reports.append(reportIn)

    def compileResults(self):
        # Aggregate data
        absCorr = 0
        correcLinks = np.zeros((11, len(self.trials)))

        # DataFrames
        columns_d = ['experiment', 'pid', 'cond', 'pilot', 'absCorrect',
                   'gMean_acc', 'mean_acc',
                   'crime_acc', 'crime_priorPosterior', 'crime_priorTruth', 'crime_cond_label', 'crime_cond', 'crime_cond_order',
                   'estate_acc', 'estate_priorPosterior', 'estate_priorTruth', 'estate_cond_label', 'estate_cond', 'estate_cond_order',
                   'finance_acc', 'finance_priorPosterior', 'finance_priorTruth', 'finance_cond_label', 'finance_cond', 'finance_cond_order',
                   'chain_n', 'chain_sign', 'chain_acc', 
                   'confound_n', 'confound_sign', 'confound_acc',
                   'dampened_n', 'dampened_sign', 'dampened_acc',
                   'ccause_n', 'ccause_acc', 
                   'collider_n', 'collider_acc', 
                   'crime_exp1_n', 'crime_exp1_acc', 
                   'crime_Corr', 'estate_Corr', 'finance_Corr', 'chain_Corr', 'confound_Corr', 'dampened_Corr', 'ccause_Corr', 'collider_Corr', 'crime_exp1_Corr',
                   'genCorr', 'posCorr', 'negCorr', 'weakCorr', 'stgCorr', 'weakPosCorr', 'stgPosCorr', 'weakNegCorr', 'stgNegCorr', 'nullCorr', 'nonNullCorr',
                   'inters', 'inters_real',
                   'crime_sense', 'crime_expect', 'estate_sense', 'estate_expect', 'finance_sense', 'finance_expect',
                   'dataValid']

        # Distance DataFrame
        self.dDistance = pd.DataFrame(columns=columns_d)
        self.editDist = pd.DataFrame(columns=columns_d)

        self.dDistance.loc[self.pid, 'pid'] = self.pid
        self.dDistance.loc[self.pid, 'experiment'] = self.experiment
        self.dDistance.loc[self.pid, 'cond'] = self.cond
        self.dDistance.loc[self.pid, 'pilot'] = self.pilot
        self.dDistance.loc[self.pid, 'dataValid'] = 1

        self.editDist.loc[self.pid, 'pid'] = self.pid
        self.editDist.loc[self.pid, 'experiment'] = self.experiment
        self.editDist.loc[self.pid, 'cond'] = self.cond
        self.editDist.loc[self.pid, 'pilot'] = self.pilot
        self.editDist.loc[self.pid, 'dataValid'] = 1
        

        cols_df_trials = [
            'experiment', 'pid', 'utid', 
            'trial_type', 'trial_name', 'trial_spec',
            'accuracy', 'hamming', 'num_indirect_errors', 'num_indirect_links',
            'priorTruth', 'priorISpost', 'cond_label', 'cond', 'cond_order',
            'genCorr', 'posCorr', 'negCorr', 'weakCorr', 'stgCorr', 'weakPosCorr', 'stgPosCorr', 'weakNegCorr', 'stgNegCorr', 'nullCorr', 'nonNullCorr',
            'inters', 'inters_real',
            'sense', 'expect',
            'variables'
        ]

        self.df_trials = pd.DataFrame(columns=cols_df_trials)
        # Interventions
        # 2DF
        ## 1: All interventions
        ## 
        cols_node_loc = [
            'experiment', 'pid', 'utid', 
            'trial_type', 'trial_name', 'trial_spec'
        ]
        self.df_node_location = pd.DataFrame(columns=cols_node_loc + ['var_pos_1', 'var_pos_2', 'var_pos_3'])
        
        columns_inter = [
            'experiment',
            'pid',
            'utid',
            'uiid',
            'trial_type',
            'trial',
            'trial_idx',
            'int_idx',
            'var_inter_idx',
            'variable',
            'prior_type',
            'gt_type',
            'var_position',
            'int_num',
            'pos_in_trial',
            'length',
            'length_sec',
            'relative_length',
            'swipes',
            'swipes10',
            'abs_mean_idle',
            'idle_freq',
            'distance_travelled',
            'min',
            'max',
            'avg',
            'sd',
            'mean_abs',
            'std_abs',
            'mode_abs',
            'mode_abs_len',
            'mode_abs_len_rel',
            'range',
            'relevant',
            'num_inters',
            'effect_1_label',
            'effect_2_label',
            'effect_1',
            'effect_2',
            'effect_1_prior',
            'effect_2_prior',
            'effect_1_estimate',
            'effect_2_estimate',
            'effect_1_error',
            'effect_2_error',
            'dataValid'
        ]
        
        #self.intervention_report = pd.DataFrame(columns=columns_inter)
        int_data_list = []
        num_int = 0
        num_int_real = 0

        self.intervention_db = {}

        for i, t in enumerate(self.trials):
            # Compile aggregate data over trials

            columns = self.df_trials.columns.to_list()
            index = self.df_trials.index.to_list()
            
            absCorr += t.correct
            
            correcLinks[0, i] = t.posCorr
            correcLinks[1, i] = t.negCorr
            correcLinks[2, i] = t.weakCorr
            correcLinks[3, i] = t.stgCorr
            correcLinks[4, i] = t.weakPosCorr
            correcLinks[5, i] = t.stgPosCorr
            correcLinks[6, i] = t.weakNegCorr
            correcLinks[7, i] = t.stgNegCorr
            correcLinks[8, i] = t.nullCorr
            correcLinks[9, i] = t.nonNullCorr
            correcLinks[10, i] = t.hammingFull

            self.df_trials.loc[t.utid, 'experiment'] = self.experiment
            self.df_trials.loc[t.utid,'pid'] = self.pid
            self.df_trials.loc[t.utid,'utid'] = t.utid
            self.df_trials.loc[t.utid,'cond'] = self.cond            
            self.df_trials.loc[t.utid, 'trial_type'] = t.type_trial    
            self.df_trials.loc[t.utid, 'trial_name'] = t.name[:-2]
            self.df_trials.loc[t.utid, 'trial_spec'] = None
            self.df_trials.loc[t.utid, 'accuracy'] = t.normEuc
            self.df_trials.loc[t.utid, 'hamming'] = t.hamming
            self.df_trials.loc[t.utid, 'num_indirect_errors'] = t.indirect_links_errors 
            self.df_trials.loc[t.utid, 'num_indirect_links'] = t.num_indirect_links
            self.df_trials.loc[t.utid,'genCorr'] = t.hammingFull
            self.df_trials.loc[t.utid,'posCorr'] = t.posCorr
            self.df_trials.loc[t.utid,'negCorr'] = t.negCorr
            self.df_trials.loc[t.utid,'weakCorr'] = t.weakCorr
            self.df_trials.loc[t.utid,'stgCorr'] = t.stgCorr
            self.df_trials.loc[t.utid,'weakPosCorr'] = t.weakPosCorr
            self.df_trials.loc[t.utid,'stgPosCorr'] = t.stgPosCorr
            self.df_trials.loc[t.utid,'weakNegCorr'] = t.weakNegCorr
            self.df_trials.loc[t.utid,'stgNegCorr'] = t.stgNegCorr
            self.df_trials.loc[t.utid,'nullCorr'] = t.nullCorr
            self.df_trials.loc[t.utid,'nonNullCorr'] = t.nonNullCorr
            self.df_trials.loc[t.utid,'std_order'] = "_".join(t.postModelOrder)
            self.df_trials.loc[t.utid, 'postOriginOrder'] = "_".join(t.postOriginOrder)
            self.df_trials.loc[t.utid, 'num_inter'] = t.num_int
            self.df_trials.loc[t.utid, 'intervened_time'] = t.intervened_time

            columns = self.df_trials.columns.to_list()
            index = self.df_trials.index.to_list()

            
    
            
            scenarios = ['crime', 'estate', 'finance']
            # Fill dataframes
            # Special case of experiment 1 control condition
            if t.name.split('_')[0] in scenarios and self.experiment == 'experiment_1' and self.cond == 1:
                self.dDistance.loc[self.pid, 'crime_exp1_n'] = int(t.name[-1])
                self.dDistance.loc[self.pid, 'crime_exp1_acc'] = t.normEuc
                self.dDistance.loc[self.pid, 'crime_exp1_Corr'] = t.correct
                self.editDist.loc[self.pid, 'crime_exp1_n'] = int(t.name[-1])
                self.editDist.loc[self.pid, 'crime_exp1_acc'] = t.hamming
                self.editDist.loc[self.pid, 'crime_exp1_Corr'] = t.correct
            

            elif t.name.split('_')[0] in scenarios:

                # Do node location
                self.df_node_location.loc[t.utid, 'experiment'] = self.experiment
                self.df_node_location.loc[t.utid,'pid'] = self.pid
                self.df_node_location.loc[t.utid,'utid'] = t.utid        
                self.df_node_location.loc[t.utid, 'trial_type'] = t.type_trial    
                self.df_node_location.loc[t.utid, 'trial_name'] = t.name[:-2]
                self.df_trials.loc[t.utid, 'trial_spec'] = None
            
                for i, var in enumerate(t.order_in_trial):
                    self.df_node_location.loc[t.utid, f'var_pos_{i+1}'] = t.prior_root_map[var]
                    self.df_node_location.loc[t.utid, f'var_name_pos_{i+1}'] = var
                for i, var in enumerate(t.order_in_trial):
                    self.df_node_location.loc[t.utid, var] = i+1

                

                self.dDistance.loc[self.pid, t.name.split('_')[0] + '_acc'] = t.normEuc
                self.editDist.loc[self.pid, t.name.split('_')[0] + '_acc'] = t.hamming
                self.dDistance.loc[self.pid, t.name.split('_')[0] + '_Corr'] = t.correct
                self.editDist.loc[self.pid, t.name.split('_')[0] + '_Corr'] = t.correct

                # Generate report vector
                ## Get correct prior report 
                prior_vec = None
                for r in self.reports:
                    if r.name == t.name.split('_')[0]:
                        prior_dict = r.report
                        prior_vec, _ = dictToVec(prior_dict)
                        prior_vec = np.array(prior_vec)
                        t.prior_vec = prior_vec
                        t.prior_matrix = causality_matrix(prior_vec)
                
    
                # Calc euclidian distance with ground truth model and prior
                distPriorTruth = np.linalg.norm(t.trueModel - prior_vec) / np.linalg.norm(abs(prior_vec) + np.ones((1, 6)))

                self.df_trials.loc[t.utid, 'priorTruth'] = distPriorTruth
                self.df_trials.loc[t.utid, 'priorISpost'] = np.array_equal(prior_vec, t.postModel)

                # Add them to df
                #self.dDistance.loc[self.pid,  t.name.split('_')[0] + '_priorPosterior'] = 1-distGenReport
                self.dDistance.loc[self.pid,  t.name.split('_')[0] + '_priorTruth'] = 1-distPriorTruth
                #self.dDistance.loc[self.pid,  t.name.split('_')[0] + '_cond_dist'] = t.cond_dist
                if self.experiment == 'experiment_2':
                    cond_as_label = 'congruent' if t.cond_dist > 0.8 else 'implausible' if t.cond_dist < 0.3 else 'incongruent'
                    cond = 1 if t.cond_dist > 0.8 else 3 if t.cond_dist < 0.3 else 2
                    
                    self.dDistance.loc[self.pid,  t.name.split('_')[0] + '_cond_label'] = cond_as_label
                    self.dDistance.loc[self.pid,  t.name.split('_')[0] + '_cond'] = cond
                    self.dDistance.loc[self.pid,  t.name.split('_')[0] + '_cond_order'] = self.diff_order.index(cond_as_label) + 1

                    self.df_trials.loc[t.utid, 'cond'] = cond
                    self.df_trials.loc[t.utid, 'cond_order'] = self.diff_order.index(cond_as_label) + 1
                elif self.experiment == 'experiment_3' or self.experiment == 'experiment_4':
                    cond_as_label = t.type_trial
                    cond = 1 if t.type_trial == 'congruent' else 2

                    self.dDistance.loc[self.pid,  t.name.split('_')[0] + '_cond_label'] = cond_as_label
                    self.dDistance.loc[self.pid,  t.name.split('_')[0] + '_cond'] = cond
                    self.dDistance.loc[self.pid,  t.name.split('_')[0] + '_cond_order'] = self.diff_order.index(cond_as_label) + 1
                    
                    self.df_trials.loc[t.utid, 'cond'] = cond
                    self.df_trials.loc[t.utid, 'cond_order'] = self.diff_order.index(cond_as_label) + 1

                self.dDistance.loc[self.pid,  t.name.split('_')[0] + '_sense'] = t.sense
                self.dDistance.loc[self.pid,  t.name.split('_')[0] + '_expect'] = t.rationale

                self.df_trials.loc[t.utid, 'sense'] = t.sense
                self.df_trials.loc[t.utid, 'expect'] = t.rationale
            

                # Calc Minimum Edit Distance
                editPriorPost = pdist(np.stack((t.postModel, prior_vec)), 'hamming')[0]
                editPriorTruth = pdist(np.stack((t.trueModel, prior_vec)), 'hamming')[0]
                #self.editDist.loc[self.pid, t.name.split('_')[0] + '_priorPosterior'] = editPriorPost
                self.editDist.loc[self.pid, t.name.split('_')[0] + '_priorTruth'] = editPriorTruth
                #self.editDist.loc[self.pid,  t.name.split('_')[0] + '_cond_dist'] = t.cond_dist
                if self.experiment == 'experiment_2':
                    cond_as_label = 'congruent' if t.cond_dist > 0.8 else 'implausible' if t.cond_dist < 0.3 else 'incongruent'
                    cond = 1 if t.cond_dist > 0.8 else 3 if t.cond_dist < 0.3 else 2

                    self.editDist.loc[self.pid,  t.name.split('_')[0] + '_cond_label'] = cond_as_label
                    self.editDist.loc[self.pid,  t.name.split('_')[0] + '_cond'] = 1 if t.cond_dist > 0.8 else 3 if t.cond_dist < 0.3 else 2
                    self.editDist.loc[self.pid,  t.name.split('_')[0] + '_cond_order'] = self.diff_order.index(cond_as_label) + 1
                elif self.experiment == 'experiment_3' or self.experiment == 'experiment_4':
                    cond_as_label = t.type_trial
                    cond = 1 if t.type_trial == 'congruent' else 2

                    self.editDist.loc[self.pid,  t.name.split('_')[0] + '_cond_label'] = cond_as_label
                    self.editDist.loc[self.pid,  t.name.split('_')[0] + '_cond'] = t.type_trial
                    self.editDist.loc[self.pid,  t.name.split('_')[0] + '_cond_order'] = self.diff_order.index(cond_as_label) + 1
                self.editDist.loc[self.pid,  t.name.split('_')[0] + '_sense'] = t.sense
                self.editDist.loc[self.pid,  t.name.split('_')[0] + '_expect'] = t.rationale
                
            else:
                name = t.name[:-2]
                if len(name.split('_')) > 1:
                    sign = name.split('_')[0]
                    name = name.split('_')[1]

                    self.dDistance.loc[self.pid, f'{name}_sign'] = sign
                    self.editDist.loc[self.pid, f'{name}_sign'] = sign
                    self.df_trials.loc[t.utid, 'trial_spec'] = sign

                self.df_trials.loc[t.utid, 'trial_name'] = name
                
                
                self.dDistance.loc[self.pid, f'{name}_n'] = int(t.name[-1])
                self.dDistance.loc[self.pid, f'{name}_acc'] = t.normEuc
                self.dDistance.loc[self.pid, f'{name}_Corr'] = t.correct

                self.editDist.loc[self.pid, f'{name}_n'] = int(t.name[-1])
                self.editDist.loc[self.pid, f'{name}_acc'] = t.hamming
                self.editDist.loc[self.pid, f'{name}_Corr'] = t.correct

            # Interventions
            #if t.type_trial == 'nolabel_control':
            #    print('hello')

            if t.interventions:
                for j, inter in enumerate(t.interventions):

                    num_int += 1
                    if inter.relevant:
                        num_int_real += 1
                    
                    int_data = {}
                    int_data['experiment'] = self.experiment
                    int_data['pid'] = self.pid
                    int_data['trial_type'] = t.type_trial
                    int_data['utid'] = t.utid
                    int_data['uiid'] = inter.uiid
                    int_data['trial'] = t.name
                    int_data['trial_idx'] = i
                    int_data['int_idx'] = j
                    int_data['var_inter_idx'] = inter.var_inter_idx
                    int_data['variable'] = inter.var
                    int_data['prior_type'] = inter.prior_type
                    int_data['gt_type'] = inter.gt_type
                    int_data['var_position'] = inter.var_pos
                    int_data['int_num'] = int(inter.num)
                    int_data['pos_in_trial'] = inter.pos_in_trial
                    int_data['length'] = inter.length
                    int_data['length_sec'] = inter.length / 5
                    int_data['relative_length'] = inter.rel_length
                    int_data['swipes'] = inter.swipes
                    int_data['abs_mean_idle'] = inter.abs_mean_idle
                    int_data['idle_freq'] = inter.idle_freq
                    int_data['swipes10'] = inter.swipes10
                    int_data['distance_travelled'] = inter.distance_travelled
                    int_data['min'] = inter.bounds[0]
                    int_data['max'] = inter.bounds[1]
                    int_data['avg'] = inter.avg
                    int_data['sd'] = inter.sd
                    int_data['mean_abs'] = inter.mean_abs
                    int_data['std_abs'] = inter.std_abs
                    int_data['mode_abs'] = inter.mode_abs
                    int_data['mode_abs_len'] = inter.mode_abs_len
                    int_data['mode_abs_len_rel'] = inter.mode_abs_len_rel
                    int_data['range'] = inter.range
                    int_data['relevant'] = inter.relevant
                    int_data['num_inters'] = t.num_int
                    int_data['effect_1_label'] = inter.effects_labels[0]
                    int_data['effect_2_label'] = inter.effects_labels[1]
                    int_data['effect_1'] = inter.effects[0]
                    int_data['effect_2'] = inter.effects[1]
                    int_data['effect_1_prior'] = inter.effects_prior[0]
                    int_data['effect_2_prior'] = inter.effects_prior[1]
                    int_data['effect_1_estimate'] = inter.effects_estimates[0]
                    int_data['effect_2_estimate'] = inter.effects_estimates[1]
                    int_data['effect_1_error'] = inter.effects_error[0]
                    int_data['effect_2_error'] = inter.effects_error[1]
                    int_data['dataValid'] = 1

                    # Add to participant db
                    self.intervention_db[inter.uiid] = int_data
                    self.intervention_db[inter.uiid]['trial_name'] = t.name
                    self.intervention_db[inter.uiid]['data'] = inter.data
                    self.intervention_db[inter.uiid]['inters'] = inter.inters
                    self.intervention_db[inter.uiid]['values'] = inter.network_states
                    self.intervention_db[inter.uiid]['values_wide'] = inter.network_states_wide
                    self.intervention_db[inter.uiid]['ground_truth_dict'] = inter.ground_truth_dict
                    self.intervention_db[inter.uiid]['ground_truth_order'] = inter.ground_truth_order
                    self.intervention_db[inter.uiid]['ground_truth'] = inter.ground_truth_vec


                    int_data_list.append(int_data)
                    
        # Fill Intervention report
        self.intervention_report = pd.DataFrame(data=int_data_list, columns=columns_inter)

        self.intervention_report['inters'] = num_int / len(self.trials)
        self.intervention_report['inters_real'] = num_int_real / len(self.trials)
        
        # Add number of interventions interventions
        self.dDistance.loc[self.pid, 'inters'] = num_int / len(self.trials)
        self.dDistance.loc[self.pid, 'inters_real'] = num_int_real / len(self.trials)

        self.editDist.loc[self.pid, 'inters'] = num_int / len(self.trials)
        self.editDist.loc[self.pid, 'inters_real'] = num_int_real / len(self.trials)

        # Add more intervention related variables
        self.dDistance.loc[self.pid, 'int_avg_range'] = self.intervention_report[self.intervention_report.relevant == 1].range.mean()
        self.dDistance.loc[self.pid, 'int_avg_length_sec'] = self.intervention_report[self.intervention_report.relevant == 1].length_sec.mean()
        self.dDistance['int_avg_range'].fillna(0, inplace=True)
        self.dDistance['int_avg_length_sec'].fillna(0, inplace=True)
        self.dDistance['inter_time'] = np.mean(np.array([t.intervened_time for t in self.trials]))

        self.editDist.loc[self.pid, 'int_avg_range'] = self.intervention_report[self.intervention_report.relevant == 1].range.mean()
        self.editDist.loc[self.pid, 'int_avg_length_sec'] = self.intervention_report[self.intervention_report.relevant == 1].length_sec.mean()
        self.editDist['int_avg_range'].fillna(0, inplace=True)
        self.editDist['int_avg_length_sec'].fillna(0, inplace=True)
        self.editDist['inter_time'] = np.mean(np.array([t.intervened_time for t in self.trials]))

        # Fill aggregates
        self.absCorrect = absCorr / len(self.trials)
        self.dDistance.loc[self.pid, 'absCorrect'] = self.absCorrect
        self.editDist.loc[self.pid, 'absCorrect'] = self.absCorrect

        cols = ['posCorr', 'negCorr', 'weakCorr', 'stgCorr', 'weakPosCorr', 'stgPosCorr', 'weakNegCorr', 'stgNegCorr', 'nullCorr', 'nonNullCorr', 'genCorr']
        #test = np.nanmean(correcLinks, axis=1, keepdims=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.dDistance.loc[self.pid, cols] = np.nanmean(correcLinks, axis=1, keepdims=True).flatten()
            self.editDist.loc[self.pid, cols] = np.nanmean(correcLinks, axis=1, keepdims=True).flatten()
        
        # Calculate mean accuracy
        accVar = [col for col in self.dDistance.columns if col[-3:] == 'acc' and col != 'mean_acc' and col != 'gMean_acc']
        self.dDistance.loc[self.pid, 'gMean_acc'] = self.dDistance.loc[self.pid, accVar].mean()
        self.editDist.loc[self.pid, 'gMean_acc'] = self.editDist.loc[self.pid, accVar].mean()
        
        labelAcc = [acc for acc in accVar if acc != 'generic_acc']
        self.dDistance.loc[self.pid, 'mean_acc'] = self.dDistance.loc[self.pid, labelAcc].mean()
        self.editDist.loc[self.pid, 'mean_acc'] = self.editDist.loc[self.pid, labelAcc].mean()

