library(reshape2)
library(sdamr)
library(lme4)
library(afex)
library(tidyverse)

# Experiment 1
## Linear model for experiment 1
data = read.csv('./data/experiment_1_rdata.csv')

mod = lm(crime_acc ~ crime_priorTruth + mean_acc, data=data)
summary(mod)

# Mixed effect model looking at contribution of structure and polarity in accuracy

# Experiment 2
## Analysis using accuracy from euclidean norm
data_2 = read.csv('./data/accuracy_lf_exp2.csv')
data_2$participant = as.factor(data_2$participant)
data_2$scenario = as.factor(data_2$scenario)
data_2$difficulty = as.factor(data_2$difficulty)

mod = afex::mixed(accuracy ~ difficulty*scenario + (1| participant), data=data_2, method='S')
mod
em1 = emmeans::emmeans(mod, 'difficulty')
em1
pairs(em1)
afex_plot(mod, "difficulty")

# Experiment 3
## Analysis using accuracy from euclidean norm
data_3 = read.csv('./data/accuracy_lf_exp3_wprior.csv')

data_3$participant = as.factor(data_3$participant)
data_3$scenario = as.factor(data_3$scenario)
data_3$difficulty = as.factor(data_3$difficulty)
data_3$lc_prior_bf = as.factor(data_3$lc_prior_bf)

# initial analysis: Model without prior shape
mod = afex::mixed(accuracy ~ difficulty*scenario + (1| participant), data=data_3, method='S')
mod
em1 = emmeans::emmeans(mod, 'difficulty')
em1
pairs(em1)
afex_plot(mod, "difficulty")

# Secondary analysis model with prior shape
# BIC
mod = afex::mixed(accuracy ~ difficulty*scenario*lc_prior_bf + (1| participant), data=data_3, method='S')
mod
em1 = emmeans::emmeans(mod, 'difficulty', 'lc_prior_bf')
em1
pairs(em1)
em1 = emmeans::emmeans(mod, 'scenario', 'lc_prior_bf')
em1
pairs(em1)
afex_plot(mod, 'scenario', 'lc_prior_bf')

# AIC

mod = afex::mixed(accuracy ~ difficulty*scenario*lc_prior_bf_aic + (1| participant), data=data_3, method='S')
mod
em1 = emmeans::emmeans(mod, 'difficulty', 'lc_prior_bf_aic')
em1
pairs(em1)
em1 = emmeans::emmeans(mod, 'scenario', 'lc_prior_bf_aic')
em1
pairs(em1)
afex_plot(mod, 'difficulty', 'lc_prior_bf_aic')



# Experiment 4
## Analysis using accuracy from euclidean norm
data_4 = read.csv('./data/accuracy_lf_exp4_wprior.csv')
data_4$participant = as.factor(data_4$participant)
data_4$scenario = as.factor(data_4$scenario)
data_4$difficulty = as.factor(data_4$difficulty)
data_4$lc_prior_bf = as.factor(data_4$lc_prior_bf)

mod = afex::mixed(accuracy ~ difficulty*scenario + lc_score + (1| participant), data=data_4, method='S')
mod
em1 = emmeans::emmeans(mod, 'difficulty')
em1
pairs(em1)
afex_plot(mod, "difficulty")

# Secondary analysis model with prior shape
# BIC

mod = afex::mixed(accuracy ~ difficulty*scenario*lc_prior_bf + lc_score + (1| participant), data=data_4, method='S')
mod
em1 = emmeans::emmeans(mod, 'difficulty', 'lc_prior_bf')
em1
pairs(em1)
afex_plot(mod, "difficulty", 'lc_prior_bf')

# AIC
mod = afex::mixed(accuracy ~ difficulty*scenario*lc_prior_bf_aic + lc_score + (1| participant), data=data_4, method='S')
mod
em1 = emmeans::emmeans(mod, 'difficulty', 'lc_prior_bf_aic')
em1
pairs(em1)
afex_plot(mod, "difficulty", 'lc_prior_bf_aic')


# Analysis of indirect link errors in labeled trials for experiment 3 and 4
df_trials = read.csv('./data/df_trials_wprior.csv')
df_trials$lc_prior_bf = as.factor(df_trials$lc_prior_bf)
df_trials$trial_type = as.factor(df_trials$trial_type)

df_trials_labelled = df_trials %>% filter(trial_name=='crime' | trial_name=='finance')

trials_3 = df_trials_labelled[df_trials_labelled$experiment == 'experiment_3', ]
trials_4 = df_trials_labelled[df_trials_labelled$experiment == 'experiment_4', ]

# Experiment 3 indirect links
mod = afex::mixed(num_indirect_errors ~ lc_prior_bf*trial_type + num_indirect_links + (1| pid), data=trials_3, method='S')
mod
em1 = emmeans::emmeans(mod, 'trial_type', 'lc_prior_bf')
em1
pairs(em1)
afex_plot(mod, 'trial_type', 'lc_prior_bf')

# Experiment 4 indirect links
mod = afex::mixed(num_indirect_errors ~ lc_prior_bf*trial_type + num_indirect_links + (1| pid), data=trials_4, method='S')
mod
em1 = emmeans::emmeans(mod, 'trial_type', 'lc_prior_bf')
em1
pairs(em1)
afex_plot(mod, 'trial_type', 'lc_prior_bf')

