import unittest
import pandapower as pp
import os
from gridsim.classes import Simulation, GridNetwork
import time


class TestSimulation(unittest.TestCase):
    def setUp(self):
        pass

    def test_compute_unsupplied_totals(self):
        t = time.clock()
        grid_network = pp.from_json(os.path.join('inputs', 'greensboro.json'))
        pp.topology.unsupplied_buses(grid_network)
        print(time.clock() - t)

        self.assertTrue(False)

