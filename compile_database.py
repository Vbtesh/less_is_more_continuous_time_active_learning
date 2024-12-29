import json
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import warnings
import csv
import os
from classes.behavioural_experiment import Behavioural_experiment
from classes.participant import Participant
from classes.trial import Trial
from classes.models import presets
from methods.extraction import getModelOrder, vecToDict, dictToVec, extractTruthModel
from methods.graph_transformations import causality_matrix

# Define experiments to add to the database
experiments = [
    'experiment_1', 
    'experiment_2', 
    'experiment_3',
    'experiment_4'
]

exceptions = [
    '566feba6b937e400052d33b2', 
    '5f108dea719866356702d26f', 
    #'5fbfe145e52a44000a9c2966',
    
]
failed = [
    '6156bf16808a46d20d0fafcb'
]
# Initialise experiment
database = Behavioural_experiment('Global database for all experiments', 
                                  datetime.now().date(),
                                  experiments,
                                  "Exp1: pilot, Exp2: prior NED based manipulation, Exp3: prior link based manipulation",
                                  exceptions=exceptions)


parts = []
failedList = []
for exp_idx, experiment in enumerate(experiments):
    # Load raw experiment data files
    with open(f'./data/raw/{experiment}_raw_data.json', 'r') as db:
        db_raw = json.load(db)

    exceptions = ['test_625034503']
    PIDs = [uid for uid in db_raw['data'].keys() if (uid[0:5] != 'pilot' and uid[0:4] != 'test' and uid[0:4] != 'beta' and uid[0:4] != 'vict') or uid in exceptions]

    data = db_raw['data']

    # Deal with prolific metadata
    ## Add time taken to df from prolific export
    if os.path.isfile(f'./data/raw/prolific_export_{experiment}.csv'):
        prolific_export = pd.read_csv(f'./data/raw/prolific_export_{experiment}.csv')
        prolific_df = prolific_export[prolific_export.status == 'APPROVED']
        # Use Pid as index
        prolific_df.index = prolific_df.participant_id

        # Correct data from timed out participants    
        database.prolific_dfs[experiment] = prolific_df


    # Do main data
    uid_idx = 0
    for uid in PIDs:    
        print(uid_idx, uid, '\n')
        uid_idx += 1
        if uid not in db_raw['demo'].keys() or uid in failed:
            failedList.append(uid)
            continue
        
        if uid not in PIDs:
            continue

        # ALL but experiment 1
        if experiment != 'experiment_1':
            cond_diff = db_raw['states'][uid]['condDifficulty']
            cond_label = db_raw['states'][uid]['condLabel']
        else:
            cond_diff = None
            cond_label = None

        part = Participant(uid, 
                           experiment,
                           db_raw['states'][uid]['condition'],
                           db_raw['demo'][uid],
                           diff_order=cond_diff,
                           label_order=cond_label,
                           pilot=False)

        # Create participants
        trials = []
        link_trials = []
        gen_idx = 0
        cond_idx = 0
        for i, t in enumerate(data[uid].keys()):

            # Skip if trial is part of training
            if t in ['links1', 'links2', 'links3']:
                continue
            
            # Recover report from database
            report = list(np.array(data[uid][t]['report']) / 2)

            # If trial is links, recover report and generate dict
            if t[:5] == 'links':
                order, name = getModelOrder(t)
                r_model, variables = vecToDict(order, report)

                utid = f'{experiment[-1]}-{uid}-{t}-links'

                # Add report to current participants
                trial = Trial(name, utid, 'links', r_model, order, contTime=False)

                link_trials.append(trial)

                continue

            # Generate ground truth dictionary model

            if 'preset_dict' not in data[uid][t].keys():
                # Get ground truth model from presets
                gt_report, order = extractTruthModel(t, presets) 

                # Generate ground truth dictionary model
                gt_model, gt_var = vecToDict(order, gt_report)

                # Generate report dictionary model
                model, variables = vecToDict(order, report)
                #print(t, gt_report, order)

            else:
                # Labelled models
                gt_model = data[uid][t]['preset_dict']
                order = data[uid][t]['order']

                # Generate report dictionary model
                model, variables = vecToDict(order, report)


            if 'expect' in data[uid][t].keys() or 'sense' in data[uid][t].keys():
                if experiment != 'experiment_1':
                    qual = [
                        int(data[uid][t]['sense'][-1]),
                        int(data[uid][t]['expect'][-1]) - 4
                    ]
                else:
                    qual = [
                        int(data[uid][t]['sense'][-1]),
                        data[uid][t]['reason']
                    ]
                

                if experiment == 'experiment_1':
                    type_trial = 'label'
                    cond_distance = None
                elif experiment == 'experiment_2':
                    cond_distance = data[uid][t]['ned_prior'] 
                    if cond_distance > 0.8:
                        type_trial = 'congruent'
                    elif cond_distance < 0.3:
                        type_trial = 'implausible'
                    else:
                        type_trial = 'incongruent'
                elif experiment == 'experiment_3' or experiment == 'experiment_4':
                    cond_distance = data[uid][t]['ned_prior']
                    if cond_distance < 1:
                        type_trial = 'incongruent'
                    else:
                        type_trial = 'congruent'
                    
                    #type_trial = cond_diff[cond_idx]
                    #cond_idx += 1

            else:
                qual = None
                cond_distance = False

                if t[:-2] == 'crime_control':
                    type_trial = 'nolabel_control'
                    print(type_trial)
                else:
                    type_trial = f'generic_{gen_idx}'

                gen_idx += 1

            utid = f'{experiment[-1]}-{uid}-{t}-{type_trial}'

            # Add trial to participant
            trial = Trial(t, utid, type_trial, model, order, model=gt_model, xyz=variables, valueLists=data[uid][t], qual=qual, cond_distance=cond_distance)
            trials.append(trial)

        # Add priors if appropriate
        for t in trials:
            for r in link_trials:
                if r.name == t.name.split('_')[0]:
                    ptorder = t.postModelOrder
                    prior_dict = r.report
                    prior_vec, prior_order = dictToVec(prior_dict)
                    prior_vec = np.array(prior_vec)
                    t.prior_dict = prior_dict
                    t.prior_vec = prior_vec
                    t.prior_matrix = causality_matrix(prior_vec)

            t.calcInterventions()
            t.calcRoots()
            

        part.addReports(link_trials)
        part.addTrials(trials)
        parts.append(part)

# Add participants at the very end
database.addParticipants(parts)

# Export specific dataframes
database.demographics.to_csv('./data/df_demographics.csv', index=False)
database.euclDist.to_csv('./data/df_participants.csv', index=False)
database.df_trials.to_csv('./data/df_trials.csv', index=False)
database.inter_report.to_csv('./data/df_interventions.csv', index=False)


# Export long form data for R analysis to csv
database.accuracy_lf_exp2.to_csv('./data/accuracy_lf_exp2.csv')
database.accuracy_lf_exp3.to_csv('./data/accuracy_lf_exp3.csv')
database.accuracy_lf_exp4.to_csv('./data/accuracy_lf_exp4.csv')

database.editdist_lf_exp2.to_csv('./data/editdist_lf_exp2.csv')
database.editdist_lf_exp3.to_csv('./data/editdist_lf_exp3.csv')
database.editdist_lf_exp4.to_csv('./data/editdist_lf_exp4.csv')

# Save dataframes to individual files


with open('./data/intervention_db.obj', 'wb') as outFile:
    pickle.dump(database.inters_db, outFile)

with open('./data/database.obj', 'wb') as outFile:
    pickle.dump(database, outFile)

with open('./data/failedList.csv', 'w') as failedFile:
    failedListStr = ','.join(failedList)
    failedFile.write(failedListStr)

# Export trial data and intervention
data_modelling = database.extract_modelling_data()

with open('./data/global_modelling_data.obj', 'wb') as model_data:
    pickle.dump(data_modelling, model_data)

pass

