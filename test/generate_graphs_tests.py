import unittest
import src.mst.mst_utilities as mst_util

class MyTestCase(unittest.TestCase):
    def test_mi_graph(self):
        n = 3
        G = mst_util.generate_mi_instance(n, 0.1)
        print(G)
        self.assertEqual(n, G.number_of_nodes())  # add assertion here
        for (u, v, w) in G.edges(data=True):
            self.assertIsNotNone(w['weight'])
    def test_smoke_testing_mi_dist(self):
        w = mst_util.mutual_information(0.1, 1)
        print(w)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
