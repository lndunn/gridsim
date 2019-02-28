import os
import pandas as pd
import numpy as np


def get_loads():
    directory_path = os.path.dirname(os.path.abspath(__file__))
    loads = pd.DataFrame(columns=['bus', 'vn_kv', 'vn_min_pu', 'vn_max_pu', 'kW', 'kVar', 'phases', 'zone'])

    for system_type in os.listdir(directory_path):
        if system_type[0] == '.':
            continue

        df = pd.read_csv(os.path.join(directory_path, system_type, 'Loads.dss'), delimiter=' ',
                         header=None, usecols=[3, 4, 5, 6, 8, 9, 10],
                         names=['bus', 'vn_kv', 'vn_min_pu', 'vn_max_pu', 'kW', 'kVar', 'phases', ], )
        for col in list(df.keys()):
            df[col] = pd.DataFrame(np.array(list(zip(*df[col].str.split('=')))).T)[1]
            if col in ['vn_kv', 'vn_min_pu', 'vn_max_pu', 'kW', 'kVar']:
                df[col] = df[col].astype(float)

        df['zone'] = system_type
        df['bus'] = pd.DataFrame(np.array(list(zip(*df['bus'].str.split('.')))).T)[0]
        loads = loads.append(df)
    return loads
