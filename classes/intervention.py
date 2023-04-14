from cv2 import threshold
import numpy as np
from scipy import stats
from methods.interventions import splitIdxArray


class Intervention():
    def __init__(self, var, 
                       var_inter_idx,
                       indices, 
                       values, 
                       var_pos, 
                       trial_length, 
                       utid, 
                       graph, 
                       gt_order, 
                       gt_vec, 
                       effects_estimates, 
                       network_states,
                       network_states_wide,
                       graph_prior=None):

        self.utid = utid
        self.pid = utid.split('-')[1]
        self.experiment = 'experiment_' + utid[0]

        self.var_inter_idx = var_inter_idx

        self.var = var
        self.var_pos = var_pos

        self.network_states = network_states
        self.network_states_wide = network_states_wide

        self.previous_value = self.network_states_wide[self.var][0]
        self.obs_time = len(self.network_states_wide[self.var]) - len(self.network_states[self.var]) - 1

        self.num = None

        self.ground_truth_dict = graph
        self.prior_dict = graph_prior
  
        self.ground_truth_order = gt_order
        self.ground_truth_vec = gt_vec

        self.effects_labels = list(graph[self.var].keys())
        self.effects = np.array(list(graph[self.var].values()))
        if self.prior_dict:
            self.effects_prior = np.array(list(graph_prior[self.var].values()))
        else:
            self.effects_prior = np.array([np.nan, np.nan])
        self.effects_estimates = np.array(list(effects_estimates.values()))
        self.effects_error = np.abs(self.effects - self.effects_estimates)

        
        
        # Function that removes noise from intervention (travel to aimed value)
        self.indices = indices
        self.values = values

        self.clean_indices, self.clean_values = self._remove_behavioural_noise(indices, values)
        self.values_abs = np.abs(self.values)

        self.length = len(self.indices)
        self.rel_length = self.length / trial_length

        if self.length > 1:
            self.value_changes = self.values[1:] - self.values[:-1]
            self.swipes, self.idle = self._find_swipes(self.value_changes)
            #abs_idle = np.abs(self.values[self.idle])
            self.idle_freq = self.idle.sum() / self.values.size
            if self.idle_freq == 0:
                self.abs_mean_idle = np.abs(self.values[-1])
            else:
                self.abs_mean_idle = np.abs(self.values[self.idle]).mean()

            self.swipes10, self.idle10 = self._find_swipes(self.value_changes, threshold=10)
            self.idle10_freq = self.idle10.sum() / self.values.size
            if self.idle10_freq == 0:
                self.abs_mean_idle10 = np.abs(self.values[-1])
            else:
                self.abs_mean_idle10 = np.abs(self.values[self.idle]).mean()

            self.distance_travelled = np.sum(np.abs(self.value_changes))
        else:
            self.value_changes = None
            self.swipes = 0
            self.abs_mean_idle = np.abs(self.values.mean())
            self.idle_freq = 1
            self.swipes10 = 0 
            self.distance_travelled = 0

        self.idx = self.indices[0]
        self.pos_in_trial = self.idx / trial_length
        self.end = self.idx + self.length

        self.bounds = np.array([np.min(self.values), np.max(self.values)])
        self.range = np.abs(self.bounds[1] - self.bounds[0])
        self.avg = np.mean(self.values)
        self.sd = np.std(self.values)

        self.mean_abs = np.mean(self.values_abs)
        self.std_abs = np.std(self.values_abs)
        mode_abs_out = stats.mode(self.values_abs)

        self.mode_abs = mode_abs_out[0][0]
        self.mode_abs_len = mode_abs_out[1][0]
        self.mode_abs_len_rel = self.mode_abs_len / self.length
        
        self.sign = self.values / self.values_abs

        self.values_mode = self.mode_abs * self.sign

        # Build inters and values
        self.inters = np.empty(len(self.network_states_wide[self.ground_truth_order[0]]))
        self.inters[:] = np.nan
        self.inters[1:len(self.indices)+1] = self.ground_truth_order.index(self.var)

        self.data = np.zeros((len(self.network_states_wide[self.ground_truth_order[0]]), len(self.ground_truth_order)))
        for i, var in enumerate(self.ground_truth_order):
            self.data[:, i] = self.network_states_wide[var]


        # Set if intervention is relevant
        ## This removes clicks, NOT GOOD
        if self.range < 5 and self.length < 5:
            self.relevant = 0
        else:
            self.relevant = 1
        

    def add_num(self, num):
        self.num = num
        self.uiid = self.utid + '-' + str(int(self.num))
        if self.uiid == '2-60ba6740a76fadb32842870e-finance_2-implausible-4':
            a = 1

    def add_var_types(self, posterior_type, gt_type, prior_type=None):
        self.posterior_type = posterior_type
        self.gt_type = gt_type
        self.prior_type = prior_type


    def _find_swipes(self, change_data, threshold=10):
        changes = 0
        prev_direction = 0
        idle = np.zeros(change_data.size + 1)
        for i, c in enumerate(change_data):
            if np.abs(c) < threshold:
                idle[i + 1] = 1
                continue
            # Check if still going in the same direction
            direction = 1 if c > 0 else -1

            if direction != prev_direction:
                changes += 1

            prev_direction = direction

        swipes = changes - 1
        return swipes + 1, idle.astype(bool)

    
    def _remove_behavioural_noise(self, indices, values):
        if len(indices) < 3:
            clean_indices = np.array([indices[-1]])
            clean_values = np.array([values[-1]])
            self._type = 'click'
            return clean_indices, clean_values
        elif len(indices) == 3:
            dist_from_init = np.abs(np.array(values) - values[0])
            clean_indices = np.array([indices[np.argmax(dist_from_init)]])
            clean_values = np.array([values[np.argmax(dist_from_init)]])
            self._type = 'click'
            return clean_indices, clean_values

        ## If frame by frame change is significant, treat it as movement, else treat is a unintentional
        threshold = 20
        last_direction = -1 if values[0] > values[1] else 1 if values[0] < values[1] else 0
        direction = None
        clean_indices = []
        clean_values = []
        for i in range(1, len(indices)-1):
            direction = -1 if values[i] > values[i+1] else 1 if values[i] < values[i+1] else 0

            if np.abs(values[i-1] - values[i]) > threshold:
                if direction == 0 or direction != last_direction:
                    clean_indices.append(indices[i])
                    clean_values.append(values[i])
            else:
                clean_indices.append(indices[i])
                clean_values.append(values[i])

            last_direction = direction

        clean_indices.append(indices[-1])
        clean_values.append(values[-1])

        return np.array(clean_indices), np.array(clean_values)





        




