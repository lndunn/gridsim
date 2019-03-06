import unittest
import pandapower as pp
import os
from gridsim.classes import Simulation, GridNetwork
import pandas as pd


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.network = pp.from_json(os.path.join('inputs', 'greensboro.json'))

    def test_compute_unsupplied_buses(self):
        # not a real test, it was to check the time to compute
        # pp.topology.unsupplied_buses(self.network)
        pass

    def test_generate_outage_realization(self):
        outage_table = Simulation.generate_outage_realization()
        self.assertIsInstance(outage_table, pd.Series)


class TestGridNetwork(unittest.TestCase):
    def setUp(self):
        self.path_network = os.path.join('inputs', 'greensboro.json')

    def test_get_loads(self):
        loads = GridNetwork._get_loads()
        self.assertIsInstance(loads, pd.DataFrame)

    def test_init(self):
        grid_network = GridNetwork(self.path_network)
        self.assertIsInstance(grid_network.line_outages, pd.DataFrame)
        self.assertIsInstance(grid_network.loads, pd.DataFrame)
        self.assertIsInstance(grid_network.affected_buses, pd.Series)
        self.assertIsInstance(grid_network.bus_outages_cost, pd.DataFrame)
        self.assertIsInstance(grid_network.network, pp.auxiliary.pandapowerNet)
