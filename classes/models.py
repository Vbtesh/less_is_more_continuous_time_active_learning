# Here insert true models
presets = {
    'sin': [1, 0, 1, 
            1, 1, 0, 
            0, -1, 1, 
            'X', 'Y', 'Z'],
    'no_link': [1, 0, 0, 
            0, 1, 0, 
            0, 0, 1, 
            'X', 'Y', 'Z'],
    'easy_1': [1, 0, 0, 
               1, 1, 0, 
               0, 0, 1, 
               'Blue', 'Red', 'Green'],
    'easy_2': [1, 0, 0, 
               0, 1, 0, 
               0, -1, 1, 
               'Blue', 'Red', 'Green'],
    'easy_3': [1, 0, 1, 
               0, 1, 0, 
               0, 0, 1, 
               'Blue', 'Red', 'Green'],
    'pos_chain_1': [1, 0, 0, 
                    1, 1, 0, 
                    0, 1, 1, 
                    'Blue', 'Red', 'Green'],
    'pos_chain_2': [1, -1, 0, 
                    0, 1, 0, 
                    -1, 0, 1, 
                    'Blue', 'Red', 'Green'],
    'pos_chain_3': [1, 1, 0, 
                    0, 1, 1, 
                    0, 0, 1, 
                    'Blue', 'Red', 'Green'],
    'dampened_1': [1, 0, 0,
                   1, 1, 0,
                   -0.5, 1, 1,
                   'Blue', 'Red', 'Green'],
    'dampened_2': [1, -1, 0,
                   0, 1, 0,
                   -1, -0.5, 1,
                   'Blue', 'Red', 'Green'],
    'dampened_3': [1, 1, 0.5,
                   0, 1, -1,
                   0, 0, 1,
                   'Blue', 'Red', 'Green'],
    'collider_1': [1, 0, 0,
                   0, 1, 0,
                   -1, -1, 1,
                   'Blue', 'Red', 'Green'],
    'collider_2': [1, -1, -1,
                   0, 1, 0,
                   0, 0, 1,
                   'Blue', 'Red', 'Green'],
    'collider_3': [1, 0, 0,
                   1, 1, 1,
                   0, 0, 1,
                   'Blue', 'Red', 'Green'],
    'ccause_1': [1, 0, 0,
                 1, 1, 0,
                 1, 0, 1,
                 'Blue', 'Red', 'Green'],
    'ccause_2': [1, 1, 0,
                 0, 1, 0,
                 0, 1, 1,
                 'Blue', 'Red', 'Green'],
    'ccause_3': [1, 0, 1,
                 0, 1, 1,
                 0, 0, 1,
                 'Blue', 'Red', 'Green'],
    'confound_1': [1, 0, 0,
                   0.5, 1, -1,
                   -1, 0, 1,
                   'Blue', 'Red', 'Green'],
    'confound_2': [1, 1, 0,
                   0, 1, 0,
                   1, 0.5, 1,
                   'Blue', 'Red', 'Green'],
    'confound_3': [1, 0, 1,
                   1, 1, 0.5,
                   0, 0, 1,
                  'Blue', 'Red', 'Green'],
    'avg_z' : [1, 0, 0,
                1, 1, 0,
                1, -1, 1,
                'X', 'Y', 'Z'],
    'convergence' : [1, -1, 1,
               1, 1, -1,
               -1, 1, 1,
               'X', 'Y', 'Z'],
    'flanck' : [1, 0, 0,
                1, 1, -1,
                1, -1, 1,
                'X', 'Y', 'Z'],
    'virus': [1, 0.8, 0.2,
              0, 1, 0, 
              0.2, -0.8, 1, 
              'Positive tests', 'Infected cases', 'Negative tests'],
    'crime': [[1, -1, 0,
               1, 1, 0, 
              -0.5, -0.5, 1, 
              'Crime rate', 'Police action', 'Population happiness',
              'Crime<br>rate', 'Police<br>action', 'Population happiness'],
              [1, 0, -1,
               -0.5, 1, -0.5, 
               1, 0, 1, 
               'Crime rate', 'Population happiness', 'Police action', 
               'Crime<br>rate', 'Population happiness', 'Police<br>action'],
               [1, 1, 0,
                -1, 1, 0, 
               -0.5, -0.5, 1, 
               'Police action', 'Crime rate', 'Population happiness',
               'Police<br>action', 'Crime<br>rate', 'Population happiness'],
               [1, 0, 1,
                -0.5, 1, -0.5, 
               -1, 0, 1, 
               'Police action', 'Population happiness', 'Crime rate',
               'Police<br>action', 'Population happiness','Crime<br>rate'],
               [1, -0.5, -0.5,
                0, 1, -1, 
                0, 1, 1, 
               'Population happiness', 'Crime rate', 'Police action', 
               'Population happiness', 'Crime<br>rate', 'Police<br>action'],
               [1, -0.5, -0.5,
                0, 1, 1, 
                0, -1, 1, 
               'Population happiness', 'Police action', 'Crime rate',
               'Population happiness', 'Police<br>action', 'Crime<br>rate']],
    'crime_control': [[1, -1, 0,
                       1, 1, 0, 
                      -0.5, -0.5, 1, 
                      'Blue', 'Red', 'Green'],
                      [1, 0, -1,
                       -0.5, 1, -0.5, 
                       1, 0, 1, 
                       'Blue', 'Red', 'Green'],
                       [1, 1, 0,
                        -1, 1, 0, 
                       -0.5, -0.5, 1, 
                       'Blue', 'Red', 'Green'],
                       [1, 0, 1,
                        -0.5, 1, -0.5, 
                       -1, 0, 1, 
                       'Blue', 'Red', 'Green'],
                       [1, -0.5, -0.5,
                        0, 1, -1, 
                        0, 1, 1, 
                        'Blue', 'Red', 'Green'],
                       [1, -0.5, -0.5,
                        0, 1, 1, 
                        0, -1, 1, 
                        'Blue', 'Red', 'Green']],
    'finance': [1, -0.5, -1,
                0, 1, -1,
                -0.5, 1, 1, 
                'Stock Prices', 'Virus cases', 'Confinement measures'],
    'finance_control': [1, -0.5, -1,
                        0, 1, -1,
                        -0.5, 1, 1, 
                        'Blue', 'Red', 'Green'],
    'estate': [1, 1, 1,
               -0.5, 1, 0.5,
               -1, -1, 1, 
               'House Prices', 'Population Density', 'Desirability',
               'House Prices', 'Population Density', 'Desira-<br>bility'],
    'estate_control': [1, 1, 1,
                       -0.5, 1, 1,
                       -1, -1, 1, 
                       'Blue', 'Red', 'Green'],
    'virus-2': [1, -0.5, -1,
                0, 1, -1,
                0.5, 1, 1, 
                'Stock Prices', 'Covid-19 cases', 'Confinement measures']
}