import numpy as np

# Converts actions as logical array (as they come in the)
def logical_to_index_actions(logical_actions):
    no_actions = 1 - logical_actions.sum(axis=1)
    actions_idle = np.zeros((logical_actions.shape[0], 1+logical_actions.shape[1]))
    actions_idle[:, 0] = no_actions
    actions_idle[:, 1:] = logical_actions

    actions_args = np.argmax(actions_idle, axis=1)
    actions_args = actions_args - 1

    actions_index = np.empty(actions_args.shape)
    actions_index[:] = np.nan

    mask = actions_args > -1
    actions_index[mask] = actions_args[mask].astype(int)

    return actions_index