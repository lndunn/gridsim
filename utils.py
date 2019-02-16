import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta, date
from scipy import stats

import pandapower as pp


def get_loads():
    path_to_gridfiles = '../../../../media/lndunn/datadrive/projects/gridsim/GSO_Base_Network/GSO_Base_Network/'
    loads = pd.DataFrame(columns=['bus', 'vn_kv', 'vn_min_pu', 'vn_max_pu', 'kW', 'kVar', 'phases','zone'])

    for system_type in os.listdir(path_to_gridfiles):
        if system_type[0]=='.':
            continue

        df = pd.read_csv(os.path.join(path_to_gridfiles, system_type, 'Loads.dss'), delimiter=' ',
                         header=None, usecols=[3,4,5,6,8,9,10],
                         names=['bus', 'vn_kv', 'vn_min_pu', 'vn_max_pu', 'kW', 'kVar', 'phases',],)
        for col in df.keys():
            df[col] = pd.DataFrame(np.array(zip(*df[col].str.split('='))).T)[1]
            if col in ['vn_kv', 'vn_min_pu', 'vn_max_pu', 'kW', 'kVar']:
                df[col]=df[col].astype(float)

        df['zone'] = system_type
        df['bus'] = pd.DataFrame(np.array(zip(*df['bus'].str.split('.'))).T)[0]
        loads = loads.append(df)
    return loads