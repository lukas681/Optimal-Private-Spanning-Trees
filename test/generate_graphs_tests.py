import unittest
from src.mst.mst_utilities import generate_mi_instance

class MyTestCase(unittest.TestCase):
    def test_mi_graph(self):
        n = 3
        G = generate_mi_instance(n, 0.1)
        print(G)
        self.assertEqual(n, G.number_of_nodes())  # add assertion here
        for (u, v, w) in G.edges(data=True):
            self.assertIsNotNone(w['weight'])


if __name__ == '__main__':
    unittest.main()
