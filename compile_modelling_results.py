import pandas as pd
import numpy as np
import pickle

experiments = [2, 3, 4]

cases = {
    2 : [
        '.\\data\\simulated_data\\ces_no_strength_&_guess_2_all',
        '.\\data\\simulated_data\\ces_strength_&_str_guess_2_all',
        '.\\data\\simulated_data\\change_obs_fk_&_att_cha_2_all',
        '.\\data\\simulated_data\\change_obs_fk_&_att_cha_prior_2_all',
        '.\\data\\simulated_data\\LC_discrete_&_1_2_all',
        '.\\data\\simulated_data\\LC_discrete_&_prior_2_all',
        '.\\data\\simulated_data\\LC_discrete_att_&_att_2_all',
        '.\\data\\simulated_data\\LC_discrete_att_&_att_prior_2_all',
        '.\\data\\simulated_data\\normative_&_1_2_all',
        '.\\data\\simulated_data\\normative_&_prior_2_all'
    ],
    3 : [
        '.\\data\\simulated_data\\ces_no_strength_&_guess_3_all',
        '.\\data\\simulated_data\\ces_strength_&_str_guess_3_all',
        '.\\data\\simulated_data\\change_obs_fk_&_att_cha_3_all',
        '.\\data\\simulated_data\\change_obs_fk_&_att_cha_prior_3_all',
        '.\\data\\simulated_data\\LC_discrete_&_1_3_all',
        '.\\data\\simulated_data\\LC_discrete_&_prior_3_all',
        '.\\data\\simulated_data\\LC_discrete_att_&_att_3_all',
        '.\\data\\simulated_data\\LC_discrete_att_&_att_prior_3_all',
        '.\\data\\simulated_data\\normative_&_1_3_all',
        '.\\data\\simulated_data\\normative_&_prior_3_all',
    ],
    4 : [
        '.\\data\\simulated_data\\ces_no_strength_&_guess_4_all',
        '.\\data\\simulated_data\\ces_strength_&_str_guess_4_all',
        '.\\data\\simulated_data\\change_obs_fk_&_att_cha_4_all',
        '.\\data\\simulated_data\\change_obs_fk_&_att_cha_prior_4_all',
        '.\\data\\simulated_data\\LC_discrete_&_1_4_all',
        '.\\data\\simulated_data\\LC_discrete_&_prior_4_all',
        '.\\data\\simulated_data\\LC_discrete_att_&_att_4_all',
        '.\\data\\simulated_data\\LC_discrete_att_&_att_prior_4_all',
        '.\\data\\simulated_data\\normative_&_1_4_all',
        '.\\data\\simulated_data\\normative_&_prior_4_all',
    ]
}



print('\n Adding modelling results to core dataframes...')
final_columns = ['participant', 'scenario', 'difficulty', 'accuracy', 'actions']

df_prior_bic = pd.read_csv('./data/prior_fitting_data.csv')
df_prior_aic = pd.read_csv('./data/prior_fitting_data_aic.csv')
df_prior_aic = df_prior_aic.rename({'final':'final_aic', 'best_prior':'best_prior_aic'}, axis=1)
prior_columns = df_prior_aic.columns
df_prior = df_prior_bic.merge(df_prior_aic)

df = pd.read_csv('./data/df_participants.csv')
df_prior.index = df.index.to_list()
df_participants = df.merge(df_prior) 

df_trials = pd.read_csv('./data/df_trials.csv')
df_trials['lc_prior_bf'] = 0

for experiment in experiments:
    print(f'Doing Experiment {experiment}...')

    #df_final.loc[0, 'difficulty'] = 2
    df_norm_CA = pd.read_csv(f'.\\data\\simulated_data\\normative_&_1_{experiment}_OA.csv')
    df_lc_CA = pd.read_csv(f'.\\data\\simulated_data\\LC_discrete_&_1_{experiment}_OA.csv')

    #df_3_norm_CA = df_3_norm_CA[~df_3_norm_CA.participant.isin(['566feba6b937e400052d33b2', '5f108dea719866356702d26f', '5fbfe145e52a44000a9c2966'])]


    #for i, case in enumerate(cases[experiment]):
    #    df_import = pd.read_csv(case + '.csv')
    #    df = df_import[final_columns]
    #    df['normative_error'] = np.nan
    #    df['lc_error'] = np.nan

    #    parts = df.participant.unique()

    ## Experiment 3: participants
    #df_long_3.difficulty = df_long_3.difficulty.replace([1, 2], ['Congruent', 'Incongruent'])
    df_long = pd.read_csv(f'./data/accuracy_lf_exp{experiment}.csv')
    parts = df_long.participant.to_list()

    final_columns = ['participant', 'scenario', 'difficulty', 'accuracy']

    df_final = df_long[final_columns]

    df_final['normative_error'] = np.nan
    df_final['normative_score'] = np.nan
    df_final['normative_entropy'] = np.nan
    df_final['lc_error'] = np.nan
    df_final['lc_score'] = np.nan
    df_final['lc_entropy'] = np.nan

    for i, part in enumerate(parts):
        # Prior bf
        part_prior_bf = df_prior[df_prior.pid == part].best_prior.to_list()[0]
        df_trials.loc[df_trials.pid == part, 'lc_prior_bf'] = 1 if part_prior_bf == 1 else 0

        part_prior_bf_aic = df_prior_aic[df_prior_aic.pid == part].best_prior_aic.to_list()[0]
        df_trials.loc[df_trials.pid == part, 'lc_prior_bf_aic'] = 1 if part_prior_bf_aic == 1 else 0

        part_data = df_final.loc[df_final.participant == part]
        norm_data = df_norm_CA.loc[df_norm_CA.participant == part]
        lc_data = df_lc_CA[df_lc_CA.participant == part]

        if part_data[part_data.difficulty == 2].empty:
            continue

        

        if experiment == 3 or experiment == 4:

            norm_score = norm_data.accuracy.to_list()
            if len(norm_score) > 2:
                norm_score = norm_score[:2]
            s = df_final.loc[df_final.participant == part]
            df_final.loc[df_final.participant == part, 'normative_score'] = norm_score

            df_final.loc[df_final.participant == part, 'normative_entropy'] = norm_data.posterior_entropy.to_list()
            shape = df_final.loc[df_final.participant == part, 'lc_score'].shape
            lc_list = lc_data.accuracy.to_list()
            df_final.loc[df_final.participant == part, 'lc_score'] = lc_data.accuracy.to_list()
            df_final.loc[df_final.participant == part, 'lc_entropy'] = lc_data.posterior_entropy.to_list()

            diff_norm = norm_data[norm_data.difficulty == 1].accuracy.values - norm_data[norm_data.difficulty == 2].accuracy.values
            diff_norm_array = [diff_norm for _ in range(part_data.shape[0])]
            df_final.loc[df_final.participant == part, 'normative_error'] = diff_norm_array

            diff_lc = lc_data[lc_data.difficulty == 1].accuracy.values - lc_data[lc_data.difficulty == 2].accuracy.values
            diff_lc_array = [diff_lc for _ in range(part_data.shape[0])]
            df_final.loc[df_final.participant == part, 'lc_error'] = diff_lc_array

        elif experiment == 2:
            continue
            #diff_norm_12 = norm_data[norm_data.difficulty == 1].accuracy.values - norm_data[norm_data.difficulty == 2].accuracy.values
            #diff_norm_23 = norm_data[norm_data.difficulty == 2].accuracy.values - norm_data[norm_data.difficulty == 3].accuracy.values
            #diff_norm_array = [diff_norm_12 + diff_norm_23 for _ in range(part_data.shape[0])]
            #df_final.loc[df_final.participant == part, 'normative_error'] = diff_norm_array

            #diff_lc_12 = lc_data[lc_data.difficulty == 1].accuracy.values - lc_data[lc_data.difficulty == 2].accuracy.values
            #diff_lc_23 = lc_data[lc_data.difficulty == 2].accuracy.values - lc_data[lc_data.difficulty == 3].accuracy.values
            #diff_lc_array = [diff_lc_12 + diff_lc_23 for _ in range(part_data.shape[0])]
            #df_final.loc[df_final.participant == part, 'lc_error'] = diff_lc_array


    # Use best fit by prior
    df_exp = df_participants[df_participants.experiment == f'experiment_{experiment}']

    df_final['lc_prior_bf'] = 0
    df_final.loc[df_long.participant.isin(df_exp[df_exp.best_prior == 1].pid), 'lc_prior_bf'] = 1
    df_final['lc_prior_bf_aic'] = 0
    df_final.loc[df_long.participant.isin(df_exp[df_exp.best_prior_aic == 1].pid), 'lc_prior_bf_aic'] = 1


    df_final.to_csv(f'./data/accuracy_lf_exp{experiment}_wprior.csv', index=False)

    # Do edit distance
    df_editdist = pd.read_csv(f'./data/editdist_lf_exp{experiment}.csv')
    df_final['accuracy'] = df_editdist.accuracy.to_list()
    df_final.to_csv(f'./data/editdist_lf_exp{experiment}_wprior.csv', index=False)

df_participants.to_csv('./data/df_participants_wprior.csv', index=False)
df_trials.to_csv('./data/df_trials_wprior.csv', index=False)
print('Done.')