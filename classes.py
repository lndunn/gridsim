import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta, date
from scipy import stats

import pandapower as pp
import utils


class GridNetwork(object):
    def __init__(self, path_to_network):
        print 'ingesting network info'
        self.network = pp.from_json(path_to_network)
        self.bus_index = self.bus_names_to_numbers()
        self.loads = self.map_loads_to_busses()
        
#         calculating cost functions (by bus)
        groups = self.loads.groupby('bus')
        self.bus_outages = pd.DataFrame(columns=['customers','load','cost'], index=groups.groups.keys())
        self.bus_outages['customers'] = groups['zone'].count()
        self.bus_outages['load'] = groups['kW'].sum()
        
#         read in cost functions (by line)
        self.affected_busses = self.get_affected_busses()
        self.line_outages = pd.read_csv(os.path.join('inputs','line_outage_consequences.csv'), index_col='line')
        
        
    def bus_names_to_numbers(self):
        bus_index = self.network['bus'][['name','zone']].reset_index()
        bus_index = bus_index.rename(columns={'name':'bus_name', 'index':'bus'})
        bus_index = bus_index.set_index(['zone','bus_name'])
        return bus_index
    
    def map_loads_to_busses(self):
        loads = utils.get_loads()
        loads = loads.rename(columns={'bus': 'bus_name'})
        loads = loads.join(self.bus_index, on=['zone','bus_name'])
        return loads
    
    def get_affected_busses(self):
        affected_busses = pd.read_csv(os.path.join('inputs','affected_busses.csv'), header=None)
        affected_busses = affected_busses.drop_duplicates().set_index(0)
        return affected_busses[1]
    
    def calc_line_consequences(self):
        if 'line_outage_consequences.csv' in os.listdir('inputs'):
            line_outages = pd.read_csv(os.path.join('inputs','line_outage_consequences.csv'), index_col='line')
        else:
            line_outages = pd.DataFrame(columns=['customers','load','cost'], index=self.network['line'].index)

        it = 0
        idx = pd.isnull(self.line_outages['customers'])
        lines_to_process = self.line_outages[idx].index
        for l, bus_list in self.affected_busses.loc[lines_to_process].iteritems():
            if pd.isnull(bus_list):
                pass
            else:
                bus_list = np.array(bus_list.split('-')).astype(int)
                bus_list = set(bus_list) & set(self.bus_outages.index)
                
                line_outages.loc[l] = self.bus_outages.loc[bus_list].sum()
            it += 1
            if it % 1000 == 0:
                print it, 'of', len(self.affected_busses)
                line_outages.to_csv(os.path.join('inputs','line_outage_consequences.csv'), index_label='line')
        line_outages.to_csv(os.path.join('inputs','line_outage_consequences.csv'), index_label='line')
        
