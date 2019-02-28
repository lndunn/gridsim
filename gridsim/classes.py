import os
import pandas as pd
import numpy as np
import pandapower as pp
from gridsim import utils
import datetime
from scipy import stats


class GridNetwork:
    def __init__(self, path_to_network):
        """

        :param path_to_network:
        """
        print('ingesting network info')
        self.network = pp.from_json(path_to_network)
        self.bus_index = self.bus_names_to_numbers()
        self.loads = self.map_loads_to_busses()

        #         calculating cost functions (by bus)
        groups = self.loads.groupby('bus')
        self.bus_outages = pd.DataFrame(columns=['customers', 'load', 'cost'], index=list(groups.groups.keys()))
        self.bus_outages['customers'] = groups['zone'].count()
        self.bus_outages['load'] = groups['kW'].sum()

        #         read in cost functions (by line)
        self.affected_busses = self.get_affected_busses()

    def bus_names_to_numbers(self):
        bus_index = self.network['bus'][['name', 'zone']].reset_index()
        bus_index = bus_index.rename(columns={'name': 'bus_name', 'index': 'bus'})
        bus_index = bus_index.set_index(['zone', 'bus_name'])
        return bus_index

    def map_loads_to_busses(self):
        loads = utils.get_loads()
        loads = loads.rename(columns={'bus': 'bus_name'})
        loads = loads.join(self.bus_index, on=['zone', 'bus_name'])
        return loads

    def get_affected_busses(self):
        affected_busses = pd.read_csv(os.path.join('inputs', 'affected_busses.csv'), header=None)
        affected_busses = affected_busses.drop_duplicates().set_index(0)
        return affected_busses[1]

    def calc_line_consequences(self):
        if 'line_outage_consequences.csv' in os.listdir('inputs'):
            line_outages = pd.read_csv(os.path.join('inputs', 'line_outage_consequences.csv'), index_col='line')
        else:
            line_outages = pd.DataFrame(columns=['customers', 'load', 'cost'], index=self.network['line'].index)

        it = 0
        idx = pd.isnull(self.line_outages['customers'])
        lines_to_process = self.line_outages[idx].index
        for l, bus_list in self.affected_busses.loc[lines_to_process].items():
            if pd.isnull(bus_list):
                pass
            else:
                bus_list = np.array(bus_list.split('-')).astype(int)
                bus_list = set(bus_list) & set(self.bus_outages.index)

                line_outages.loc[l] = self.bus_outages.loc[bus_list].sum()
            it += 1
            if it % 1000 == 0:
                print()
                it, 'of', len(self.affected_busses)
                line_outages.to_csv(os.path.join('inputs', 'line_outage_consequences.csv'), index_label='line')
        line_outages.to_csv(os.path.join('inputs', 'line_outage_consequences.csv'), index_label='line')


class Simulation:
    def __init__(self, system, line_types, outage_rates, repair_times, event_duration=30, multiplier=(1, 1), k=10, dt=15. / 60,
                 results_dir='simulations'):
        """

        :param system:
        :param line_types:
        :param outage_rates:
        :param repair_times:
        :param event_duration:
        :param multiplier:
        :param k:
        :param dt:
        :param results_dir:
        """

        #TODO(Mathilde): store in a smart way the configurations for the simulation

        self.number = sum(['simulation' in f for f in os.listdir(os.path.join(results_dir, 'outage_tables'))])

        self.grid_network = GridNetwork(os.path.join('inputs', 'greensboro.json'))

        self.k = k
        self.dt = dt
        self.event_duration = event_duration
        self.multiplier = multiplier
        self.results_dir = results_dir
        if not hasattr(outage_rates, '__len__'):
            unique_linetypes = line_types.unique()
            self.rates = pd.Series(outage_rates, index=unique_linetypes)
            self.repair_times = pd.Series(repair_times, index=unique_linetypes)
        else:
            self.rates = outage_rates
            self.repair_times = repair_times

        self.line_types = line_types
        self.idx_l_type = dict(zip(self.rates.index, [line_types == l_type for l_type in outage_rates.index]))

        self.line_failure_rates = self.calc_failure_rates(system)
        self.outage_table = self.generate_outage_realization(system)

        #TODO(Mathilde): remove that part a create a methode
        self.outage_table, self.total_costs = self.simulate_event(self.grid_network)

        self.save_results()

    def save_results(self):
        keys = [('number', self.number),
                ('timestamp', str(datetime.now()).split('.')[0]),
                ('event_duration', self.event_duration),
                ('k', self.k),
                ('percent_underground',
                 (self.grid_network['line']['length_km'][self.line_types == 'underground']).sum() / self.grid_network['line'][
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

        self.outage_table.to_csv(
            os.path.join(self.results_dir, 'outage_tables', 'outage_table-simulation%i.csv' % (self.number)))
        self.total_costs.to_csv(
            os.path.join(self.results_dir, 'time_series', 'system_costs-simulation%i.csv' % (self.number)))

    def calc_failure_rates(self, system):
        failure_rates = pd.Series(np.nan, index=system.network['line'].index)
        for l_type, rate in self.rates.iteritems():
            failure_rates.loc[self.idx_l_type[l_type]] = rate
        return failure_rates

    def generate_outage_realization(self, system):
        time_steps = np.arange(0, self.event_duration, self.dt)
        number_of_faults = pd.Series(0, index=time_steps)

        for l_type, rate in self.rates.iteritems():
            number_of_faults += pd.Series(
                stats.poisson.rvs(rate * system.network['line'].loc[self.idx_l_type[l_type]]['length_km'].sum(),
                                  size=len(number_of_faults)), index=time_steps)

        dfs = []
        unsupplied_busses = {}
        for t in number_of_faults.index:
            lines_out = np.random.choice(system.network['line'].index,
                                         size=number_of_faults.loc[t],
                                         p=(self.line_failure_rates / self.line_failure_rates.sum()).tolist())

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

    def compute_unsupplied_totals(self, system):
        unsupplied_busses = pp.topology.unsupplied_buses(system.network)
        unsupplied_load_busses = unsupplied_busses & set(system.bus_outages.index)
        total_consequences = system.bus_outages.loc[unsupplied_load_busses].sum()
        return total_consequences

    def compute_outage_cost(self, system, lines_out, new_outages, t, k):
        no_change = (len(new_outages) == 0) & sum(
            ((self.outage_table['end'] >= t - self.dt) & (self.outage_table['end'] < t)) == 0)
        if no_change:
            return self.outage_table, None
        topo_searches = 0

        # first calculate the total unserved load associated with the current outages
        total_consequences = self.compute_unsupplied_totals(system)
        topo_searches += 1

        # first cut at computing damages looks at precomputed values for N-1 topologies
        #    this approx should work most of the time, except if two redundant lines fail
        estimated_consequences = system.line_outages.loc[new_outages]

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
            for key in system.line_outages.keys():
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

                system.network['line']['in_service'].loc[line] = True
                restoration_benefits = self.compute_unsupplied_totals(system)
                marginal_change = total_consequences - restoration_benefits
                system.network['line']['in_service'].loc[line] = False

                self.outage_table[marginal_change.index].loc[line] = marginal_change

                idx = pd.notnull(self.outage_table.loc[lines_out]['load'])
                close_enough = np.abs(total_consequences.loc['load']
                                      - self.outage_table['load'].loc[lines_out].sum()) < 1e-3

                topo_searches += 1
                print
                'searches: ', topo_searches

        return self.outage_table, total_consequences

    def simulate_event(self, system, prioritization_criteria='load'):
        lines_out = []
        t = 0
        k = int(self.k)
        self.total_costs = pd.DataFrame(0, columns=system.bus_outages.keys(), index=[t, ])
        while (t <= self.event_duration) or (len(lines_out) > 0):
            new_outages = self.outage_table[self.outage_table['start'] == t].index.tolist()

            if any([new in lines_out for new in new_outages]):
                idx = self.outage_table.index.isin(new_outages)
                self.outage_table['end'].loc[idx] = np.nan

            lines_out.extend(new_outages)
            lines_out = list(set(lines_out))

            if len(lines_out) == 0:
                pass
            else:
                system.network['line']['in_service'].loc[lines_out] = False
                self.outage_table, total_cost = self.compute_outage_cost(system, lines_out, new_outages, t, k)

                if type(total_cost) == type(None):
                    self.total_costs.loc[t] = self.total_costs.loc[self.total_costs.index.max()]
                else:
                    self.total_costs.loc[t] = total_cost

            self.outage_table = self.outage_table.sort_values(prioritization_criteria, ascending=False)

            idx = pd.isnull(self.outage_table['end'])
            restore_lines = self.outage_table[idx].index[:k]

            self.outage_table['end'].loc[restore_lines] = t + self.outage_table['repair_time'].loc[restore_lines]

            idx = (self.outage_table['end'] >= t) & (self.outage_table['end'] < t + self.dt)
            repairs_completed = self.outage_table[idx].index
            k += len(repairs_completed)
            system.network['line']['in_service'].loc[repairs_completed] = True
            lines_out = system.network['line'][system.network['line']['in_service'] == False].index.tolist()

            t += self.dt

        return self.outage_table, self.total_costs

