import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from methods.extraction import vecToDict, dictToVec, signChangeCode

class Behavioural_experiment():
    def __init__(self, name, date, experiments, details=None, participants=None, exceptions=[]):
        self.name = name
        self.date = date
        self.experiments = experiments
        self.details = details
        self.participants = []
        if participants:
            self.participants = participants # Has to be a list of participant objects, see participants.py
        
        self.partIds = []
        self.prolific_dfs = {}
        self.demographics = None
        self.reports = None # All reports that are directly inputed by participants that are not demographics
        self.exceptions = exceptions

    def addParticipants(self, partIn):
        if type(partIn) == list:
            self.participants += partIn
        else:
            self.participants.append(partIn)
        
        # Rerun partIds, demographics and reports init (to be defined)
        self.genIdList()
        self.genDataTables()

    def genIdList(self):
        for part in self.participants:
            if not part in self.partIds:
                self.partIds.append(part.pid)

    def genDataTables(self):
        # General intervention database
        self.inters_db = {}

        # Demographics
        self.demographics = pd.DataFrame()

        # Sign focused df for crime model
        self.signChange = pd.DataFrame()
        self.signSummary = pd.DataFrame()

        # Models dataframes
        ## 5 link values world
        self.priors = pd.DataFrame()
        self.posteriors = pd.DataFrame()
        ## 3 link values world
        self.priors_3 = pd.DataFrame()       
        self.posteriors_3 = pd.DataFrame()
        ## Standard models
        self.models = pd.DataFrame()
        self.modelsOrigin = pd.DataFrame()

        # Performance metrics
        self.euclDist = pd.DataFrame(columns=self.participants[0].dDistance.columns)
        self.editDist = pd.DataFrame(columns=self.participants[0].editDist.columns)

        # Trials dataframe
        self.df_trials = pd.DataFrame()

        # Location dataframe
        self.df_node_location = pd.DataFrame()

        # Root dataframe
        self.df_root_location = pd.DataFrame()

        # Performance long form
        ## EXISTS ONLY FOR EXP2 AND EXP3
        self.accuracy_lf_exp2 = pd.DataFrame(columns=['accuracy', 'difficulty', 'scenario', 'participant'])
        self.accuracy_lf_exp3 = pd.DataFrame(columns=['accuracy', 'difficulty', 'scenario', 'participant'])

        self.editdist_lf_exp2 = pd.DataFrame(columns=['accuracy', 'difficulty', 'scenario', 'participant'])
        self.editdist_lf_exp3 = pd.DataFrame(columns=['accuracy', 'difficulty', 'scenario', 'participant'])

        # Interventions
        self.inter_report = pd.DataFrame()
        
        for part in self.participants:
            # Collect interventions
            for k, v in part.intervention_db.items():
                self.inters_db[k] = v

            if self.df_trials.empty:
                self.df_trials = part.df_trials
            else:   
                self.df_trials = pd.concat([self.df_trials, part.df_trials], axis=0)

            if self.df_node_location.empty:
                self.df_node_location = part.df_node_location
            else:   
                self.df_node_location = pd.concat([self.df_node_location, part.df_node_location], axis=0)

            self.demographics.loc[part.pid, 'experiment'] = part.experiment
            self.demographics.loc[part.pid, 'age'] = part.age
            self.demographics.loc[part.pid, 'gender'] = part.gender
            self.demographics.loc[part.pid, 'cond'] = part.cond
            self.demographics.loc[part.pid, 'activity'] = part.activity
            self.demographics.loc[part.pid, 'causalFam'] = part.causalFam
            self.demographics.loc[part.pid, 'graphSent'] = part.graphSent
            self.demographics.loc[part.pid, 'modelSent'] = part.modelSent

            
            for trial in part.trials:

                self.posteriors.loc[trial.utid, 'pid'] = part.pid
                self.posteriors.loc[trial.utid, 'experiment'] = part.experiment
                self.posteriors.loc[trial.utid, 'trial_type'] = trial.type_trial

                self.posteriors_3.loc[trial.utid, 'pid'] = part.pid
                self.posteriors_3.loc[trial.utid, 'experiment'] = part.experiment
                self.posteriors_3.loc[trial.utid, 'trial_type'] = trial.type_trial

                self.signChange.loc[trial.utid, 'pid'] = part.pid
                self.signChange.loc[trial.utid, 'experiment'] = part.experiment
                self.signChange.loc[trial.utid, 'trial_type'] = trial.type_trial

                self.models.loc[trial.utid, 'pid'] = part.pid
                self.models.loc[trial.utid, 'experiment'] = part.experiment
                self.models.loc[trial.utid, 'trial_type'] = trial.type_trial

                # Prior dataframes
                self.priors.loc[trial.utid, 'pid'] = part.pid
                self.priors.loc[trial.utid, 'experiment'] = part.experiment
                self.priors.loc[trial.utid, 'trial_type'] = trial.type_trial
                
                self.priors_3.loc[trial.utid, 'pid'] = part.pid
                self.priors_3.loc[trial.utid, 'experiment'] = part.experiment
                self.priors_3.loc[trial.utid, 'trial_type'] = trial.type_trial

                
                

                if trial.name.split('_')[0] in ['crime', 'estate', 'finance'] and not (part.experiment=='experiment_1' and part.cond==1):
                    report = trial.prior_vec
                    order = trial.postModelOrder
                    self.priors.loc[trial.utid, f'{order[0]}>>{order[1]}'] = report[0]
                    self.priors.loc[trial.utid, f'{order[0]}>>{order[2]}'] = report[1]
                    self.priors.loc[trial.utid, f'{order[1]}>>{order[0]}'] = report[2]
                    self.priors.loc[trial.utid, f'{order[1]}>>{order[2]}'] = report[3]
                    self.priors.loc[trial.utid, f'{order[2]}>>{order[0]}'] = report[4]
                    self.priors.loc[trial.utid, f'{order[2]}>>{order[1]}'] = report[5]

                    # Sign change
                    report_3 = [
                            1 if report[0] > 0 else -1 if report[0] < 0 else 0,
                            1 if report[1] > 0 else -1 if report[1] < 0 else 0,
                            1 if report[2] > 0 else -1 if report[2] < 0 else 0,
                            1 if report[3] > 0 else -1 if report[3] < 0 else 0,
                            1 if report[4] > 0 else -1 if report[4] < 0 else 0,
                            1 if report[5] > 0 else -1 if report[5] < 0 else 0,
                        ]

                    self.priors_3.loc[trial.utid, f'{order[0]}>>{order[1]}'] = report_3[0]
                    self.priors_3.loc[trial.utid, f'{order[0]}>>{order[2]}'] = report_3[1]
                    self.priors_3.loc[trial.utid, f'{order[1]}>>{order[0]}'] = report_3[2]
                    self.priors_3.loc[trial.utid, f'{order[1]}>>{order[2]}'] = report_3[3]
                    self.priors_3.loc[trial.utid, f'{order[2]}>>{order[0]}'] = report_3[4]
                    self.priors_3.loc[trial.utid, f'{order[2]}>>{order[1]}'] = report_3[5]
                    
                    # Posterior dataframes

                    self.posteriors.loc[trial.utid, 'pid'] = part.pid
                    self.posteriors.loc[trial.utid, 'experiment'] = part.experiment
                    self.posteriors.loc[trial.utid, 'trial_type'] = trial.type_trial

                    self.posteriors_3.loc[trial.utid, 'pid'] = part.pid
                    self.posteriors_3.loc[trial.utid, 'experiment'] = part.experiment
                    self.posteriors_3.loc[trial.utid, 'trial_type'] = trial.type_trial

                    self.signChange.loc[trial.utid, 'pid'] = part.pid
                    self.signChange.loc[trial.utid, 'experiment'] = part.experiment
                    self.signChange.loc[trial.utid, 'trial_type'] = trial.type_trial

                    self.models.loc[trial.utid, 'pid'] = part.pid
                    self.models.loc[trial.utid, 'experiment'] = part.experiment
                    self.models.loc[trial.utid, 'trial_type'] = trial.type_trial

   
                    report, order = dictToVec(trial.postModelStd)

                    self.posteriors.loc[trial.utid, f'{order[0]}>>{order[1]}'] = report[0]
                    self.posteriors.loc[trial.utid, f'{order[0]}>>{order[2]}'] = report[1]
                    self.posteriors.loc[trial.utid, f'{order[1]}>>{order[0]}'] = report[2]
                    self.posteriors.loc[trial.utid, f'{order[1]}>>{order[2]}'] = report[3]
                    self.posteriors.loc[trial.utid, f'{order[2]}>>{order[0]}'] = report[4]
                    self.posteriors.loc[trial.utid, f'{order[2]}>>{order[1]}'] = report[5]

                    # Sign Change
                    report_3 = [
                        1 if report[0] > 0 else -1 if report[0] < 0 else 0,
                        1 if report[1] > 0 else -1 if report[1] < 0 else 0,
                        1 if report[2] > 0 else -1 if report[2] < 0 else 0,
                        1 if report[3] > 0 else -1 if report[3] < 0 else 0,
                        1 if report[4] > 0 else -1 if report[4] < 0 else 0,
                        1 if report[5] > 0 else -1 if report[5] < 0 else 0,
                    ]

                    self.posteriors_3.loc[trial.utid, f'{order[0]}>>{order[1]}'] = report_3[0]
                    self.posteriors_3.loc[trial.utid, f'{order[0]}>>{order[2]}'] = report_3[1]
                    self.posteriors_3.loc[trial.utid, f'{order[1]}>>{order[0]}'] = report_3[2]
                    self.posteriors_3.loc[trial.utid, f'{order[1]}>>{order[2]}'] = report_3[3]
                    self.posteriors_3.loc[trial.utid, f'{order[2]}>>{order[0]}'] = report_3[4]
                    self.posteriors_3.loc[trial.utid, f'{order[2]}>>{order[1]}'] = report_3[5]

                    # Populate signchange dataframe

                    #    print('Problem')
                    self.signChange.loc[trial.utid, f'{order[0]}>>{order[1]}'] = signChangeCode(int(self.priors_3.loc[trial.utid, f'{order[0]}>>{order[1]}']), report_3[0])
                    self.signChange.loc[trial.utid, f'{order[0]}>>{order[2]}'] = signChangeCode(int(self.priors_3.loc[trial.utid, f'{order[0]}>>{order[2]}']), report_3[1])
                    self.signChange.loc[trial.utid, f'{order[1]}>>{order[0]}'] = signChangeCode(int(self.priors_3.loc[trial.utid, f'{order[1]}>>{order[0]}']), report_3[2])
                    self.signChange.loc[trial.utid, f'{order[1]}>>{order[2]}'] = signChangeCode(int(self.priors_3.loc[trial.utid, f'{order[1]}>>{order[2]}']), report_3[3])
                    self.signChange.loc[trial.utid, f'{order[2]}>>{order[0]}'] = signChangeCode(int(self.priors_3.loc[trial.utid, f'{order[2]}>>{order[0]}']), report_3[4])
                    self.signChange.loc[trial.utid, f'{order[2]}>>{order[1]}'] = signChangeCode(int(self.priors_3.loc[trial.utid, f'{order[2]}>>{order[1]}']), report_3[5])

                else:
                    # Do models standardised
                    report, order = dictToVec(trial.postModelStd)
  
                    # Add model type
                    if len(trial.name[:-2].split('_')) > 1:
                        sign, name = trial.name[:-2].split('_')
                        if sign == 'crime':
                            name = 'crime_exp1'
                    else:
                        sign = np.nan
                        name = trial.name[:-2]

                    self.models.loc[trial.utid, 'sign'] = sign
                    self.models.loc[trial.utid, 'type'] = name

                    # Add model number
                    self.models.loc[trial.utid, 'num'] = part.dDistance.loc[part.pid, f'{name}_n']

                    self.models.loc[trial.utid, f'{order[0]}>>{order[1]}'] = report[0]
                    self.models.loc[trial.utid, f'{order[0]}>>{order[2]}'] = report[1]
                    self.models.loc[trial.utid, f'{order[1]}>>{order[0]}'] = report[2]
                    self.models.loc[trial.utid, f'{order[1]}>>{order[2]}'] = report[3]
                    self.models.loc[trial.utid, f'{order[2]}>>{order[0]}'] = report[4]
                    self.models.loc[trial.utid, f'{order[2]}>>{order[1]}'] = report[5]

                    ## Do models origin
                    self.modelsOrigin.loc[trial.utid, 'experiment'] = part.experiment
                    self.modelsOrigin.loc[trial.utid, 'trial_type'] = trial.type_trial

                    self.modelsOrigin.loc[trial.utid, 'sign'] = sign
                    self.modelsOrigin.loc[trial.utid, 'type'] = name

                    report = trial.postOrigin
                    # Add model number
                    self.modelsOrigin.loc[trial.utid, 'num'] = part.dDistance.loc[part.pid, f'{name}_n']
                
                    self.modelsOrigin.loc[trial.utid, f'{order[0]}>>{order[1]}'] = report[0]
                    self.modelsOrigin.loc[trial.utid, f'{order[0]}>>{order[2]}'] = report[1]
                    self.modelsOrigin.loc[trial.utid, f'{order[1]}>>{order[0]}'] = report[2]
                    self.modelsOrigin.loc[trial.utid, f'{order[1]}>>{order[2]}'] = report[3]
                    self.modelsOrigin.loc[trial.utid, f'{order[2]}>>{order[0]}'] = report[4]
                    self.modelsOrigin.loc[trial.utid, f'{order[2]}>>{order[1]}'] = report[5]

           
            # Accuracy measures
            self.euclDist.loc[part.pid] = part.dDistance.loc[part.pid]
            self.editDist.loc[part.pid] = part.editDist.loc[part.pid]
            if part.pid in self.exceptions:
                self.euclDist.loc[part.pid, 'dataValid'] = 0
                self.editDist.loc[part.pid, 'dataValid'] = 0
            # Intervention compilations
            if self.inter_report.empty:
                self.inter_report = part.intervention_report
            else:
                self.inter_report = self.inter_report.append(part.intervention_report, ignore_index=True)


        dfSign_arr = self.signChange.to_numpy()

        # Compute change type frequencies across all conditions 
        self.signSummary['no_change'] = np.sum(dfSign_arr == 0, axis=1) / dfSign_arr.shape[1]
        self.signSummary['sign_change'] = (np.sum(dfSign_arr == 1, axis=1) + np.sum(dfSign_arr == 2, axis=1)) / dfSign_arr.shape[1]
        self.signSummary['link_noLink'] = np.sum(dfSign_arr == 3, axis=1) / dfSign_arr.shape[1]
        self.signSummary['noLink_link'] = np.sum(dfSign_arr == 4, axis=1) / dfSign_arr.shape[1]

        # Add part id as index
        self.signSummary.index = self.signChange.index

        # # Add time taken to df from prolific export        
        for e in self.experiments:
            if e in self.prolific_dfs.keys():
                for prolific_part in self.prolific_dfs[e].participant_id.to_list():
                    if prolific_part in self.euclDist.index:
                        self.euclDist.at[prolific_part, 'time_taken'] = self.prolific_dfs[e].loc[prolific_part, 'time_taken'] / 60 # in minutes
                        self.editDist.at[prolific_part, 'time_taken'] = self.prolific_dfs[e].loc[prolific_part, 'time_taken'] / 60 # in minutes

        ## Add exceptions to time taken:
        special_cases = {
            '5badf0aeae1d020001df2cc7': [datetime.fromtimestamp(1623165556), datetime.fromtimestamp(1623171197)],
            'test_625034503': [datetime.fromtimestamp(1592829237), datetime.fromtimestamp(1592830645)]
        }
        
        for special_id, times in special_cases.items():
            if special_id in self.euclDist.index:
                begin = times[0]
                end = times[1]
                self.euclDist.at[special_id, 'time_taken'] = (end-begin).total_seconds() / 60 # As minutes
                self.editDist.at[special_id, 'time_taken'] = (end-begin).total_seconds() / 60 # As minutes

        # Long form dataframe for experiment 2 and 3
        # Generate long form dataframe for linear mixed effect model
        ## NEEDS UPDATING TO ACCOMODATE BOTH EXPERIMENTS
        
        mapper_2 = {
          'crime_acc':'crime',
          'estate_acc':'estate',
          'finance_acc':'finance'
        }
        mapper_3 = {
          'crime_acc':'crime',
          'finance_acc':'finance'
        }

        exp_2_euc = self.euclDist[self.euclDist.experiment == 'experiment_2']
        exp_3_euc = self.euclDist[self.euclDist.experiment == 'experiment_3']
        exp_4_euc = self.euclDist[self.euclDist.experiment == 'experiment_4']
        #pids = pd.Index([pid for pid in self.euclDist.index if pid not in self.exceptions])
        df_2_input = exp_2_euc[exp_2_euc.dataValid == 1].loc[:,['crime_acc', 'crime_cond', 'estate_acc', 'estate_cond', 'finance_acc', 'finance_cond', 'gMean_acc', 'time_taken', 'inters_real', 'int_avg_range', 'int_avg_length_sec']].rename(mapper=mapper_2, axis=1)
        df_3_input = exp_3_euc[exp_3_euc.dataValid == 1].loc[:,['crime_acc', 'crime_cond', 'finance_acc', 'finance_cond', 'gMean_acc', 'time_taken', 'inters_real', 'int_avg_range', 'int_avg_length_sec']].rename(mapper=mapper_3, axis=1)
        df_4_input = exp_4_euc[exp_4_euc.dataValid == 1].loc[:,['crime_acc', 'crime_cond', 'finance_acc', 'finance_cond', 'gMean_acc', 'inters_real', 'int_avg_range', 'int_avg_length_sec']].rename(mapper=mapper_3, axis=1)

        exp_2_ee = self.editDist[self.editDist.experiment == 'experiment_2']
        exp_3_ee = self.editDist[self.editDist.experiment == 'experiment_3']
        exp_4_ee = self.editDist[self.editDist.experiment == 'experiment_4']

        df_2_ee_input = exp_2_ee[exp_2_ee.dataValid == 1].loc[:,['crime_acc', 'crime_cond', 'estate_acc', 'estate_cond', 'finance_acc', 'finance_cond', 'gMean_acc', 'time_taken', 'inters_real', 'int_avg_range', 'int_avg_length_sec']].rename(mapper=mapper_2, axis=1)
        df_3_ee_input = exp_3_ee[exp_3_ee.dataValid == 1].loc[:,['crime_acc', 'crime_cond', 'finance_acc', 'finance_cond', 'gMean_acc', 'time_taken', 'inters_real', 'int_avg_range', 'int_avg_length_sec']].rename(mapper=mapper_3, axis=1)
        df_4_ee_input = exp_4_ee[exp_4_ee.dataValid == 1].loc[:,['crime_acc', 'crime_cond', 'finance_acc', 'finance_cond', 'gMean_acc', 'inters_real', 'int_avg_range', 'int_avg_length_sec']].rename(mapper=mapper_3, axis=1)


        if len(df_2_input) > 0:
            # Accuracy
            df_2_input['participant'] = df_2_input.index
            df_2_input['part_idx'] = df_2_input.reset_index().index
            # Edit distance
            df_2_ee_input['participant'] = df_2_ee_input.index
            df_2_ee_input['part_idx'] = df_2_ee_input.reset_index().index

            # Long form dataframe
            ## Accuracy
            df_2_s = pd.melt(df_2_input, ['crime_cond', 'estate_cond', 'finance_cond', 'part_idx', 'participant', 'gMean_acc', 'time_taken', 'inters_real', 'int_avg_range', 'int_avg_length_sec'])
            df_2_s2 = df_2_s.rename({'variable':'scenario', 'value':'accuracy', 'gMean_acc':'mean_accuracy', 'int_avg_range':'int_range', 'int_avg_length_sec':'int_length'}, axis=1)

            ## Edit distance
            df_2_ee_s = pd.melt(df_2_ee_input, ['crime_cond', 'estate_cond', 'finance_cond', 'part_idx', 'participant', 'gMean_acc', 'time_taken', 'inters_real', 'int_avg_range', 'int_avg_length_sec'])
            df_2_ee_s2 = df_2_ee_s.rename({'variable':'scenario', 'value':'accuracy', 'gMean_acc':'mean_accuracy', 'int_avg_range':'int_range', 'int_avg_length_sec':'int_length'}, axis=1)


            def filtering_2(crime, estate, finance, scenario):
                if scenario == 'crime':
                    return crime
                elif scenario == 'estate':
                    return estate
                else:
                    return finance

            # Accuracy
            df_2_s2['difficulty'] = df_2_s2.apply(lambda x: filtering_2(x.crime_cond, x.estate_cond, x.finance_cond, x.scenario), axis=1)
            self.accuracy_lf_exp2 = df_2_s2.drop(['crime_cond', 'estate_cond', 'finance_cond'], axis=1)
            # Edit distances
            df_2_ee_s2['difficulty'] = df_2_s2.apply(lambda x: filtering_2(x.crime_cond, x.estate_cond, x.finance_cond, x.scenario), axis=1)
            self.editdist_lf_exp2 = df_2_ee_s2.drop(['crime_cond', 'estate_cond', 'finance_cond'], axis=1)
            
        
        if len(df_3_input) > 0:
            # Accuracy
            df_3_input['participant'] = df_3_input.index
            df_3_input['part_idx'] = df_3_input.reset_index().index
            # Edit distance
            df_3_ee_input['participant'] = df_3_ee_input.index
            df_3_ee_input['part_idx'] = df_3_ee_input.reset_index().index

            # Long form dataframes
            # Accuracy
            df_3_s = pd.melt(df_3_input, ['crime_cond', 'finance_cond', 'part_idx', 'participant', 'gMean_acc', 'time_taken', 'inters_real', 'int_avg_range', 'int_avg_length_sec'])
            df_3_s2 = df_3_s.rename({'variable':'scenario', 'value':'accuracy', 'gMean_acc':'mean_accuracy', 'int_avg_range':'int_range', 'int_avg_length_sec':'int_length'}, axis=1)
            # Edit distance
            df_3_ee_s = pd.melt(df_3_ee_input, ['crime_cond', 'finance_cond', 'part_idx', 'participant', 'gMean_acc', 'time_taken', 'inters_real', 'int_avg_range', 'int_avg_length_sec'])
            df_3_ee_s2 = df_3_ee_s.rename({'variable':'scenario', 'value':'accuracy', 'gMean_acc':'mean_accuracy', 'int_avg_range':'int_range', 'int_avg_length_sec':'int_length'}, axis=1)

            def filtering_3(crime, finance, scenario):
                if scenario == 'crime':
                    return crime
                else:
                    return finance

            # Accuracy
            df_3_s2['difficulty'] = df_3_s2.apply(lambda x: filtering_3(x.crime_cond, x.finance_cond, x.scenario), axis=1)
            self.accuracy_lf_exp3 = df_3_s2.drop(['crime_cond', 'finance_cond'], axis=1)
            # Edit distance
            df_3_ee_s2['difficulty'] = df_3_ee_s2.apply(lambda x: filtering_3(x.crime_cond, x.finance_cond, x.scenario), axis=1)
            self.editdist_lf_exp3 = df_3_ee_s2.drop(['crime_cond', 'finance_cond'], axis=1)

        if len(df_4_input) > 0:
            # Accuracy
            df_4_input['participant'] = df_4_input.index
            df_4_input['part_idx'] = df_4_input.reset_index().index
            # Edit distance
            df_4_ee_input['participant'] = df_4_ee_input.index
            df_4_ee_input['part_idx'] = df_4_ee_input.reset_index().index

            # Long form dataframes
            # Accuracy
            df_4_s = pd.melt(df_4_input, ['crime_cond', 'finance_cond', 'part_idx', 'participant', 'gMean_acc', 'inters_real', 'int_avg_range', 'int_avg_length_sec'])
            df_4_s2 = df_4_s.rename({'variable':'scenario', 'value':'accuracy', 'gMean_acc':'mean_accuracy', 'int_avg_range':'int_range', 'int_avg_length_sec':'int_length'}, axis=1)
            # Edit distance
            df_4_ee_s = pd.melt(df_4_ee_input, ['crime_cond', 'finance_cond', 'part_idx', 'participant', 'gMean_acc', 'inters_real', 'int_avg_range', 'int_avg_length_sec'])
            df_4_ee_s2 = df_4_ee_s.rename({'variable':'scenario', 'value':'accuracy', 'gMean_acc':'mean_accuracy', 'int_avg_range':'int_range', 'int_avg_length_sec':'int_length'}, axis=1)

            def filtering_3(crime, finance, scenario):
                if scenario == 'crime':
                    return crime
                else:
                    return finance

            # Accuracy
            df_4_s2['difficulty'] = df_4_s2.apply(lambda x: filtering_3(x.crime_cond, x.finance_cond, x.scenario), axis=1)
            self.accuracy_lf_exp4 = df_4_s2.drop(['crime_cond', 'finance_cond'], axis=1)
            # Edit distance
            df_4_ee_s2['difficulty'] = df_4_ee_s2.apply(lambda x: filtering_3(x.crime_cond, x.finance_cond, x.scenario), axis=1)
            self.editdist_lf_exp4 = df_4_ee_s2.drop(['crime_cond', 'finance_cond'], axis=1)

    # Extract trial data, inters and fittable inters
    def extract_modelling_data(self):
        data_dict = {}

        num_trials = 0
        for p in self.participants:
            if p in self.exceptions:
                continue
            data_dict[p.pid] = {}
            order_of_trials = p.diff_order
            data_dict[p.pid]['experiment'] = p.experiment
            data_dict[p.pid]['trial_order'] = order_of_trials
            data_dict[p.pid]['trials'] = {}
            trial_idx = 0
            for t in p.trials:
                
                if t.name[:-2] not in ['crime', 'estate', 'finance']:
        
                    model_order = t.trueOriginOrder 
                    model_order_trial = t.order_in_trial
                    
                    data_dict[p.pid]['trials'][t.type_trial] = {
                        'utid': t.utid,
                        'name': t.name,
                        'trial_index': trial_idx,
                        'data' : t.data,
                        'inters': t.inters,
                        'inters_fit': t.inters_fit,
                        'standard_order': model_order,
                        'in_trial_order': model_order_trial,
                        'posterior': t.postModel,
                        'ground_truth': t.trueModel,
                        'links_hist': t.links_hist_std
                    }

                    trial_idx += 1
                    num_trials += 1
                    
                    continue

                else:                        
                    
                    trial_idx = order_of_trials.index(t.type_trial) + 1 if order_of_trials else 4

                
                model_order = t.trueModelOrder
                model_order_trial = t.order_in_trial
                
                for r in p.reports:
                    if r.trueModelOrder == model_order:
                        prior = np.array(r.model)

                data_dict[p.pid]['trials'][t.type_trial] = {
                        'utid': t.utid,
                        'name': t.name,
                        'trial_index': trial_idx,
                        'data' : t.data,
                        'inters': t.inters,
                        'inters_fit': t.inters_fit,
                        'standard_order': model_order,
                        'in_trial_order': model_order_trial,
                        'prior': prior,
                        'posterior': t.postModel,
                        'ground_truth': t.trueModel,
                        'links_hist': t.links_hist_std
                    }
                
                num_trials += 1

        print(num_trials)

        in_dict = 0
        for p in data_dict.keys():
            in_dict += len(data_dict[p]['trials'].keys())

        print(in_dict)

        return data_dict

    # recover participant data 
    def recover_participant_trial_data(self, idx=None, pid=None, trial_idx=0):
        # Plot all four trials
        if type(idx) == int:
            part = self.participants[idx]
        elif type(pid) == str:
            for p in self.participants:
                if p.pid == pid:
                    part = p
        else:
            raise TypeError('Error in participant identifier')
        # Need to show condition (difficulty) and true model      
        
        trial = part.trials[trial_idx]
        params = trial.trial_params

        N, K, theta, dt, sigma = params['N'], params['K'], params['theta'], params['dt'], params['sigma']
        X = trial.data
        labels = trial.trueModelOrder
        inters = trial.inters
        trial_name = trial.type_trial.capitalize()

        return X, inters, labels, trial_name 
    
    # Plot participant data 
    def plot_participant_data(self, idx=None, pid=None, summary=False, inters_fit=False):
        # Plot all four trials
        if type(idx) == int:
            part = self.participants[idx]
        elif type(pid) == str:
            for p in self.participants:
                if p.pid == pid:
                    part = p
        else:
            raise TypeError('Error in participant identifier')
        # Need to show condition (difficulty) and true model
        
        if type(idx) == int:
            print('Participant: ', idx, 'PID:', part.pid)
        else:
            print('Participant: ', pid)

        fig = plt.figure(figsize=(12, 12))
        
        plot_idx = 1

        for i in range(len(part.trials)):
            

            trial = part.trials[i]
            params = trial.trial_params

            N, K, theta, dt, sigma = params['N'], params['K'], params['theta'], params['dt'], params['sigma']
            X = trial.data
            
            if inters_fit:
                I = [trial.inters, trial.inters_fit]
            else:
                I = [trial.inters]
            labels = trial.trueModelOrder

            for j in range(len(I)):
                palette = sns.color_palette() # Set palette
                sns.set_palette(palette)

                plt.subplot(len(part.trials), len(I), plot_idx)
                plot_idx += 1

                inters = I[j]

                for k in range(K):
                    # PLot data 
                    ax = sns.lineplot(data=X[:,k], lw=1.5, label=labels[k]) # Plot data

                    # Plot interventions where relevant
                    
                    ints = inters == k
                    if np.sum(ints) == 0:
                        continue
                    
                    x = np.arange(len(ints))
                    y1 = -100 * ints
                    y2 = 100 * ints
                    ax.fill_between(x, y1, y2, color=palette[k], alpha=0.15)

                plt.legend()

                if trial.cond_dist:
                    cond = "{:.2f}".format(trial.cond_dist)
                else:
                    cond = trial.name

                accuracy = "{:.2f}".format(trial.normEuc)

                if trial.type_trial in ['label', 'congruent', 'incongruent', 'implausible']:
                    name = trial.type_trial.capitalize()
                else:
                    if 'chain' in trial.name:
                        name = 'Chain'
                    elif 'confound' in trial.name:
                        name = 'Confound'
                    else:
                        name = trial.name[:-2].capitalize()
                plt.title(f'Trial: {name}, Model: {trial.trueModel}, Acc.: {accuracy}')
                plt.ylim(-100, 100)
                plt.xlim(0, len(ints))
                plt.legend(labelspacing=2, loc=6, bbox_to_anchor=(1, 0.5))

            plt.tight_layout()
                

            if summary:
                print('Condition: ', cond)
                print('True model: ', trial.trueModel)
                print('Posterior mode: ', trial.postModel)
                print('Accuracy: ', accuracy, '\n')








   



            
        