import unittest
import math
from src.mst.pamst import pamst
from networkx import Graph
from src.mst.mst_utilities import generate_random_complete_graph

class pamst_tests(unittest.TestCase):
    def test_pamst(self):
        """"
        Reproduces a bug where the MST value increases for ery large noise levels.
        Let's find the reason!
        """
        n = 100
        sensitivity = 1
        G = generate_random_complete_graph(n, 1)
        noise_level = lambda rho: (1/2 * sensitivity * math.sqrt( (n-1)/(2 * rho))) # Should be ok

        rhos = range(1,1000000,5000)
        for rho in rhos:
            edges = pamst(G, noise_level(rho))
            print("Noise " + str(rho) + " " + str(sum(e[2]["weight"] for e in edges)))
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
