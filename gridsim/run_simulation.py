import os
import pandas as pd
import numpy as np
from scipy import stats
import logging

from gridsim.classes import Simulation, GridNetwork

# save locally
MAIN_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIRECTORY = os.path.join(MAIN_DIRECTORY, 'simulations')
METADATA_FILE = 'metadata.csv'
METADATA_PATH = os.path.join(RESULTS_DIRECTORY, METADATA_FILE)
INPUT_DIRECTORY = os.path.join(MAIN_DIRECTORY, 'inputs')


def repair_time_distribution(sig, mu, distribution=stats.lognorm):
    probs = pd.Series(index=range(0, 48 * 60, ))
    probs.loc[0] = 0
    for x0, x1 in zip(probs.index[:-1], probs.index[1:]):
        probs.loc[x1] = (distribution.cdf(x1, sig, mu - sig ** 2 / 2.)
                         - distribution.cdf(x0, sig, mu - sig ** 2 / 2.))
    return probs / probs.sum()


def run_simulation(path_network):
    system = GridNetwork(os.path.join('inputs', 'greensboro.json'))
    dt = 15.
    n_sim = 10
    multiplier = (1, 1)  # increase in prob for OH and UG lines (respectively)
    event_duration = 30  # minutes

    for percent_underground in [100, 20, 40, 60, 80, 0]:
        # overground is 0 underground is 1
        if percent_underground == 0:
            line_types = pd.Series('overhead', index=system.network['line'].index)
        elif percent_underground == 100:
            line_types = pd.Series('underground', index=system.network['line'].index)
        else:
            line_miles = system.network['line'][['length_km', ]]
            line_miles['load'] = system.line_outages['load'] / system.network['line']['length_km']
            line_miles = line_miles.sort_values('load')
            line_miles['cumulative_sum'] = 100 * line_miles['length_km'].cumsum() / line_miles['length_km'].sum()
            idx = line_miles['cumulative_sum'] <= percent_underground
            line_types = pd.Series('overhead', index=system.network['line'].index)
            line_types.loc[idx] = 1

        metadata = pd.read_csv(METADATA_PATH)

        idx = ((metadata['event_duration'] == event_duration)
               & (metadata['percent_underground'] == percent_underground)
               & (metadata['lambda_overhead'] == multiplier[0])
               & (metadata['lambda_underground'] == multiplier[1]))

        remaining_sims = n_sim - sum(idx)

        outage_rates = pd.Series([multiplier[0] * dt * 0.6 / 8760. / 60., multiplier[1] * dt * 0.3 / 8760. / 60.],
                                 index=['overhead', 'underground'])

        system.network['line']['in_service'] = True

        probs = {'underground': repair_time_distribution(3, 145),
                 'overhead': repair_time_distribution(3, 92)}

        repair_times = pd.Series([lambda: np.random.choice(probs['overhead'].index, p=probs['overhead'].tolist()),
                                  lambda: np.random.choice(probs['overhead'].index, p=probs['overhead'].tolist())],
                                 index=['overhead', 'underground'])
        # What is k??
        k = 10

        idx = system.line_outages
        # TODO: Remove idx because I think some of them are for different purposes
        count = 0
        while count < remaining_sims:
            print('----- Simulation %s for percentage underground %s -----' % (count, percent_underground))
            sim = Simulation(system, line_types, outage_rates, repair_times, path_network,
                             event_duration=event_duration,
                             multiplier=multiplier, k=k,
                             dt=dt,
                             results_dir=RESULTS_DIRECTORY)
            outage_table, total_costs = sim.simulate_event()
            sim.save_results(outage_table, total_costs)
            count += 1


if __name__ == '__main__':
    path_network = os.path.join(INPUT_DIRECTORY, 'greensboro.json')
    run_simulation(path_network)
