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

        rhos = range(1000000,10000000,10000)
        for rho in rhos:
            scaling = noise_level(rho)
            edges = pamst(G.copy(), scaling)
            print("Noise Scaling " + str(scaling))
            res = sorted(edges, key=lambda x: x[2].get("weight", 1))
            print(res)
            print("Noise " + str(rho) + ": Weight " + str(sum(e[2]["weight"] for e in res)))
            print("\n")
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
