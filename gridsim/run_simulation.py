import os
import pandas as pd
import numpy as np
from scipy import stats

from classes import Simulation, GridNetwork
import utils as u


def repair_time_distribution(sig, mu, distribution=stats.lognorm):
    probs = pd.Series(index=range(0, 48 * 60, ))
    probs.loc[0] = 0
    for x0, x1 in zip(probs.index[:-1], probs.index[1:]):
        probs.loc[x1] = (distribution.cdf(x1, sig, mu - sig ** 2 / 2.)
                         - distribution.cdf(x0, sig, mu - sig ** 2 / 2.))
    return probs / probs.sum()


def run_simulation(path_network):
    grid_network = GridNetwork(os.path.join('inputs', 'greensboro.json'))
    dt = 15.
    n_sim = 10
    multiplier = (1, 1)  # increase in prob for OH and UG lines (respectively)
    event_duration = 30  # minutes
    list_percent_underground = [100, 20, 40, 60, 80, 0]

    # TODO: Change 'underground' and 'overhead' in 0 or 1

    # Compute the line type dataframe for underground or overground lines
    for percent_underground in list_percent_underground:
        line_miles = grid_network.network['line'][['length_km', ]]
        line_miles['load'] = grid_network.line_outages['load'] / grid_network.network['line']['length_km']
        line_miles = line_miles.sort_values('load')
        line_miles['cumulative_sum'] = 100 * line_miles['length_km'].cumsum() / line_miles['length_km'].sum()
        idx_underground_lines = line_miles['cumulative_sum'] <= percent_underground
        line_types = pd.Series('overhead', index=grid_network.network['line'].index)
        line_types.loc[idx_underground_lines] = 'underground'

        # Check is simulations are done
        metadata = pd.read_csv(u.METADATA_PATH)
        simulations_done = ((metadata['event_duration'] == event_duration)
                            & (metadata['percent_underground'] == percent_underground)
                            & (metadata['lambda_overhead'] == multiplier[0])
                            & (metadata['lambda_underground'] == multiplier[1]))
        remaining_sims = n_sim - sum(simulations_done)

        outage_rates = pd.Series([multiplier[0] * dt * 0.6 / 8760. / 60., multiplier[1] * dt * 0.3 / 8760. / 60.],
                                 index=['overhead', 'underground'])

        probability_repair_time = {'underground': repair_time_distribution(3, 145),
                                   'overhead': repair_time_distribution(3, 92)}

        repair_times = pd.Series([lambda: np.random.choice(probability_repair_time['overhead'].index,
                                                           p=probability_repair_time['overhead'].tolist()),
                                  lambda: np.random.choice(probability_repair_time['overhead'].index,
                                                           p=probability_repair_time['overhead'].tolist())],
                                 index=['overhead', 'underground'])

        # TODO(Laurel): What is k??
        k = 10

        for i in range(remaining_sims):
            print('----- Simulation %s for percentage underground %s -----' % (i, percent_underground))
            number = int(len(os.listdir(u.RESULTS_DIRECTORY))/2)
            sim = Simulation(number, grid_network, line_types, outage_rates, repair_times, path_network,
                             event_duration=event_duration,
                             multiplier=multiplier, k=k,
                             dt=dt,
                             results_dir=u.RESULTS_DIRECTORY)
            outage_table, total_costs = sim.simulate_event()
            sim.save_results(outage_table, total_costs)


if __name__ == '__main__':
    path_network = os.path.join(u.INPUT_DIRECTORY, 'greensboro.json')
    run_simulation(path_network)

    # # temporary only when using the
    # import pstats

    # p = pstats.Stats('output_file')
    # p.sort_stats('time').print_stats(10)
    # # p.strip_dirs().sort_stats(-1).print_stats()
