import os
import pandas as pd
import numpy as np
import pandapower as pp
import datetime
from scipy import stats
import logging
import gridsim.utils as u


class GridNetwork:
    def __init__(self, path_to_network):
        self.network = pp.from_json(path_to_network)
        self.loads = self._get_and_join_loads_to_network(self.network)
        self.bus_outages_cost = self._compute_bus_outages_cost(self.loads)
        self.affected_buses = self._get_affected_buses()
        self.line_outages = self._initialize_line_outage(self.network)

    def reset_network(self):
        self.network['line']['in_service'] = True

    def compute_and_save_line_consequences(self):
        it = 0
        idx = pd.isnull(self.line_outages['customers'])
        lines_to_process = self.line_outages[idx].index
        for l, bus_list in self.affected_buses.loc[lines_to_process].items():
            if not pd.isnull(bus_list):
                bus_list = np.array(bus_list.split('-')).astype(int)
                bus_list = set(bus_list) & set(self.bus_outages_cost.index)

                self.line_outages.loc[l] = self.bus_outages_cost.loc[bus_list].sum()
            it += 1
            if it % 1000 == 0:
                logging.info('%s of %s', (it, len(self.affected_buses)))
                self.line_outages.to_csv(u.LINE_OUTAGE_PATH, index_label='line')
        # TODO(Mathilde): Check if that line is necessary
        self.line_outages.to_csv(u.LINE_OUTAGE_PATH, index_label='line')

    def _get_and_join_loads_to_network(self, network):
        bus_index = network['bus'][['name', 'zone']].reset_index()
        bus_index = bus_index.rename(columns={'name': 'bus_name', 'index': 'bus'})
        bus_index = bus_index.set_index(['zone', 'bus_name'])
        loads = self._get_loads()
        loads = loads.rename(columns={'bus': 'bus_name'})
        loads = loads.join(bus_index, on=['zone', 'bus_name'])
        return loads

    @staticmethod
    def _initialize_line_outage(network):
        if u.LINE_OUTAGE_FILE in os.listdir(u.INPUT_DIRECTORY):
            line_outages = pd.read_csv(u.LINE_OUTAGE_PATH, index_col='line')
        else:
            pd.DataFrame(columns=['customers', 'load', 'cost'], index=network['line'].index)
        return line_outages

    @staticmethod
    def _get_affected_buses():
        affected_buses = pd.read_csv(u.LINE_OUTAGE_PATH, header=None)
        affected_buses = affected_buses.drop_duplicates().set_index(0)
        return affected_buses[1]

    @staticmethod
    def _get_loads():
        loads = pd.DataFrame(columns=['bus', 'vn_kv', 'vn_min_pu', 'vn_max_pu', 'kW', 'kVar', 'phases', 'zone'])

        for system_type in os.listdir(u.GSO_DIRECTORY):
            if system_type[0] == '.':
                continue

            df = pd.read_csv(os.path.join(u.GSO_DIRECTORY, system_type, 'Loads.dss'), delimiter=' ',
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

    @staticmethod
    def _compute_bus_outages_cost(loads):
        groups = loads.groupby('bus')
        bus_outages_cost = pd.DataFrame(columns=['customers', 'load', 'cost'], index=list(groups.groups.keys()))
        bus_outages_cost['customers'] = groups['zone'].count()
        bus_outages_cost['load'] = groups['kW'].sum()
        return bus_outages_cost


class Simulation:
    def __init__(self, grid_network, line_types, outage_rates, repair_times, path_network, event_duration=30,
                 multiplier=(1, 1), k=10,
                 dt=15. / 60,
                 results_dir=u.RESULTS_DIRECTORY):

        # this is possible if outage_tables exist
        # self.number = sum(['simulation' in f for f in os.listdir(os.path.join(results_dir, 'outage_tables'))])
        self.number = 1

        self.grid_network = grid_network
        self.grid_network.reset_network()

        self.k = k
        self.dt = dt
        self.event_duration = event_duration
        self.multiplier = multiplier
        self.results_dir = results_dir
        if not hasattr(outage_rates, '__len__'):
            unique_line_types = line_types.unique()
            self.rates = pd.Series(outage_rates, index=unique_line_types)
            self.repair_times = pd.Series(repair_times, index=unique_line_types)
        else:
            self.rates = outage_rates
            self.repair_times = repair_times

        self.line_types = line_types
        self.idx_l_type = dict(zip(self.rates.index, [line_types == l_type for l_type in outage_rates.index]))

        self.line_failure_rates = self.calc_failure_rates()
        self.outage_table = self.generate_outage_realization()

    def save_results(self, outage_table, total_costs):
        keys = [('number', self.number),
                ('timestamp', str(datetime.datetime.now()).split('.')[0]),
                ('event_duration', self.event_duration),
                ('k', self.k),
                ('percent_underground',
                 (self.grid_network.network['line']['length_km'][self.line_types == 'underground']).sum() /
                 self.grid_network.network['line'][
                     'length_km'].sum()),
                ('lambda_overhead', self.multiplier[0]),
                ('lambda_underground', self.multiplier[1])]
        metadata = pd.DataFrame(pd.Series(dict(keys), name=self.number)).T

        if 'metadata.csv' not in os.listdir(self.results_dir):
            metadata.to_csv(os.path.join(self.results_dir, 'metadata.csv'))
        else:
            f = open(os.path.join(self.results_dir, 'metadata.csv'), 'a')
            metadata.to_csv(f, header=False)
            f.close()

        outage_table.to_csv(
            os.path.join(self.results_dir, 'outage_table_simulation%i.csv' % self.number))
        total_costs.to_csv(
            os.path.join(self.results_dir, 'system_costs_simulation%i.csv' % self.number))

    def calc_failure_rates(self):
        failure_rates = pd.Series(np.nan, index=self.grid_network.network['line'].index)
        for l_type, rate in self.rates.iteritems():
            failure_rates.loc[self.idx_l_type[l_type]] = rate
        return failure_rates

    def generate_outage_realization(self):
        time_steps = np.arange(0, self.event_duration, self.dt)
        number_of_faults = pd.Series(0, index=time_steps)

        for l_type, rate in self.rates.iteritems():
            number_of_faults += pd.Series(
                stats.poisson.rvs(
                    rate * self.grid_network.network['line'].loc[self.idx_l_type[l_type]]['length_km'].sum(),
                    size=len(number_of_faults)), index=time_steps)

        dfs = []
        for t in number_of_faults.index:
            # TODO: Here we have a division by zero!!!
            try:
                p = (self.line_failure_rates / self.line_failure_rates.sum()).tolist()
            except:
                p = self.line_failure_rates.tolist()
            lines_out = np.random.choice(self.grid_network.network['line'].index,
                                         size=number_of_faults.loc[t],
                                         p=p)

            df = pd.DataFrame(index=lines_out)
            df['start'] = t
            df['repair_time'] = np.nan
            df['end'] = np.nan
            df['customers'] = np.nan
            df['load'] = np.nan
            df['cost'] = np.nan
            df['line_type'] = self.line_types.loc[lines_out]
            for l_type, repair_time in self.repair_times.iteritems():
                idx = df['line_type'] == l_type
                #                 df['repair_time'].loc[idx] = stats.lognorm.rvs(repair_time[0], repair_time[1], size=sum(idx)
                df['repair_time'].loc[idx] = [repair_time() for x in range(sum(idx))]
            dfs.append(df)
        outage_table = pd.concat(dfs, ignore_index=False)
        return outage_table.reset_index().rename(columns={'index': 'line'}).set_index('line')

    def compute_unsupplied_totals(self):
        print('----- Compute unsupplied lines -----')
        unsupplied_buses = pp.topology.unsupplied_buses(self.grid_network.network)
        unsupplied_load_buses = unsupplied_buses & set(self.grid_network.bus_outages_cost.index)
        total_consequences = self.grid_network.bus_outages_cost.loc[unsupplied_load_buses].sum()
        return total_consequences

    def compute_outage_cost(self, lines_out, new_outages, t, k):
        no_change = (len(new_outages) == 0) & sum(
            ((self.outage_table['end'] >= t - self.dt) & (self.outage_table['end'] < t)) == 0)
        if no_change:
            return self.outage_table, None
        topo_searches = 0

        # first calculate the total unserved load associated with the current outages
        total_consequences = self.compute_unsupplied_totals()
        topo_searches += 1

        # first cut at computing damages looks at precomputed values for N-1 topologies
        #    this approx should work most of the time, except if two redundant lines fail
        estimated_consequences = self.grid_network.line_outages.loc[new_outages]

        damage_keys = total_consequences.index
        for outage in new_outages:
            for key in damage_keys:
                self.outage_table[key].loc[outage] = estimated_consequences[key].loc[outage]

        #         for key in damage_keys:
        #             idx = pd.isnull(estimated_consequences[key])
        #             self.outage_table[key].loc[new_outages].loc[idx] = 0

        # now check to see if the approximation is accurate
        #     (there's a high probability that it is)
        close_enough = np.abs(total_consequences.loc['load'] - self.outage_table['load'].loc[lines_out].sum()) < 1e-3
        if close_enough:
            for key in self.grid_network.line_outages.keys():
                self.outage_table[key].loc[new_outages] = estimated_consequences[key].loc[new_outages]
            pass
        else:
            # if that didn't work, then iterate through all the lines that failed
            #     start with the ones that didn't affect any customers in the N-1 scenarios

            idx = pd.isnull(self.outage_table.loc[lines_out]['load'])
            process_lines = (self.outage_table.loc[lines_out][idx].index.tolist()
                             + self.outage_table.loc[lines_out][~idx].sort_values('load',
                                                                                  ascending=False).index.tolist())

            while ~close_enough and len(process_lines) > 0:
                line = process_lines.pop(0)

                self.grid_network.network['line']['in_service'].loc[line] = True
                restoration_benefits = self.compute_unsupplied_totals()
                marginal_change = total_consequences - restoration_benefits
                self.grid_network.network['line']['in_service'].loc[line] = False

                self.outage_table[marginal_change.index].loc[line] = marginal_change

                # TODO: idx here is never used ...
                idx = pd.notnull(self.outage_table.loc[lines_out]['load'])
                close_enough = np.abs(total_consequences.loc['load']
                                      - self.outage_table['load'].loc[lines_out].sum()) < 1e-3

                topo_searches += 1
                logging.info('search iteration: %s', topo_searches)

        return self.outage_table, total_consequences

    def simulate_event(self, prioritization_criteria='load'):
        lines_out = []
        t = 0
        k = int(self.k)
        total_costs = pd.DataFrame(0, columns=self.grid_network.bus_outages_cost.keys(), index=[t, ])
        while (t <= self.event_duration) or (len(lines_out) > 0):
            new_outages = self.outage_table[self.outage_table['start'] == t].index.tolist()

            if any([new in lines_out for new in new_outages]):
                idx = self.outage_table.index.isin(new_outages)
                self.outage_table['end'].loc[idx] = np.nan

            lines_out.extend(new_outages)
            lines_out = list(set(lines_out))

            print('lines_out: ', lines_out)
            # lines_out is never updated ...

            if len(lines_out) == 0:
                pass
            else:
                self.grid_network.network['line']['in_service'].loc[lines_out] = False
                self.outage_table, total_cost = self.compute_outage_cost(lines_out, new_outages, t, k)

                if type(total_cost) == type(None):
                    # I don't understand this condition
                    total_costs.loc[t] = total_costs.loc[total_costs.index.max()]
                else:
                    total_costs.loc[t] = total_cost

            self.outage_table = self.outage_table.sort_values(prioritization_criteria, ascending=False)

            idx = pd.isnull(self.outage_table['end'])
            restore_lines = self.outage_table[idx].index[:k]

            self.outage_table['end'].loc[restore_lines] = t + self.outage_table['repair_time'].loc[restore_lines]

            idx = (self.outage_table['end'] >= t) & (self.outage_table['end'] < t + self.dt)
            repairs_completed = self.outage_table[idx].index
            k += len(repairs_completed)
            self.grid_network.network['line']['in_service'].loc[repairs_completed] = True
            lines_out = self.grid_network.network['line'][
                self.grid_network.network['line']['in_service'] == False].index.tolist()

            t += self.dt

        return self.outage_table, total_costs
