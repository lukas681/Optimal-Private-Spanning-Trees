import math
import numpy as np
from itertools import count
# DEPRECATED.
class FastDPPQ:
    """
    Implements a novel Priority Queue that quickly allows sampling from the top group.
    A: Describes an additional adjacency matrix where we store the current position of an edge and the referenced weight class,
        Takes two attributs (index_weight_class, index_in_C)
    W: All weight classes, used for fast sampling of these edges
        Takes 5-Tuple (s_value, counter, _ start_of_equence_C, end_of_sequence_C)
    C: An ascending list of all edges
        (e1,e2,e3, ..) initially ordered by weight, but gets shiftet around.
    B: A Block array for fast traversal, Additional Data sttructure
    """
    @classmethod
    def __init__(self, g:Graph, s):
        """
        :param graph:
        :param s: gap size, for 0 leq mi leq 1, we need 1/s leq (d*d)
        """
        if 1/s > len(g.edges):
            raise "Warning! Discretization parameter would be worse then treating edges seperately!"

        # Step 1: Put all of the edges into an array which will then be references by our weight classes
        num_weight_classes = math.ceil(1/s) # Asuming Mutual Information as boundaries
        print("Setting W...")
        self.W = np.empty(num_weight_classes, dtype=object)
        for (idx, w) in enumerate(np.arange(0,1,s)): # The last element contains the whole last range
            self.W[idx] = [0,-1,-1] # (Counter, start, end), -1 represent not used, -1 represent not used

        self.block_size = math.ceil(math.sqrt(num_weight_classes))
        print(f'Block size {self.block_size}')
        print("Setting A...")
        self.A = np.empty((g.number_of_nodes(), g.number_of_nodes()), dtype=object)
        print("Setting B...")
        self.B = np.empty(self.block_size, dtype=int)

        print("Setting C... Depending on the size of the graph, this operation might be slow.")
        #        self.C = sorted(g.edges(data=True), key=lambda e: e[2].get('weight', 0)) # 2 d d  log d sorting
        self.C = list(
            map(lambda e:e[:3],
                sorted(G.edges.data("weight", default=0), key=lambda e: e[2])
                ))
        # G.edges.data("weight", default=1)

        idx_weight = 0
        idx_offset = 0
        group_start = 0
        idx = 0
        for e in self.C:
            if(idx%100000==0):
                print(f'Preparing edge: {idx}')

            if (idx_weight < num_weight_classes-1  ## We need a new weight class
                    and e[2] > (idx_weight+1) * s):
                self.W[idx_weight][1] = group_start
                self.W[idx_weight][2] = idx - 1
                idx_weight = math.floor(e[2] / s)
                group_start=idx
                idx_offset=0

            # Check weather we have to increase the pointer to idx_weight TODO Check wether still correct
            self.A[e[0]][e[1]] = [idx_weight, idx_offset] # set position in C
            self.A[e[1]][e[0]] = [idx_weight, idx_offset]

            idx_offset+=1
            idx+=1
        # Last one:
        self.W[idx_weight][1] = group_start
        self.W[idx_weight][2] = len(self.C)-1

    @classmethod
    def find_largest_non_empty_weight_block(self):
        largest_block_idx = len(self.B) - 1
        # Finding Max Block
        while  largest_block_idx >= 0:
            if self.B[largest_block_idx] != 0:
                return largest_block_idx
            largest_block_idx-=1
        return None

    @classmethod
    def find_max_weight(self):
        largest_block = self.find_largest_non_empty_weight_block()
        # Now, find the concrete weight class
        for idx in reversed(
                range(largest_block * self.block_size, min(len(self.W), (largest_block+1) * self.block_size))): # last block might not be filled up
            if self.W[idx][0] != 0:
                return self.W[idx]  # Checking for count
        return None

    @classmethod
    def insert(self, e):
        # Not using variable unpacking for speednes
        v = e[0]
        w = e[1]
        index_W = self.A[v][w][0]
        index_C = self.A[v][w][1]
        (count, start, end) = self.W[index_W]
        len = end - start # maybe storing instead of end?

        if (index_C < count
                or count >= len+1
                or self.A[v][w] is None):
            return # Membership tests and Fast fails

        self.swap(self, start+count, start+index_C)
        self.A[v][w][1] = count # not sure wether its a #FIXME
        self.A[w][v][1] = count
        self.W[index_W][0] = count+1
        self.B[self.find_block_number(self,  index_W)]+=1

    @classmethod
    def delete(self, e):
        v = e[0]
        w = e[1]
        [index_W, index_C] = self.A[v][w]
        (count, start, end) = self.W[index_W]

        if index_C > count or count == 0: # membership test, was not inserted
            print("Something went wrong")
            return

        position_to = count
        self.A[v][w][1] = count
        self.A[w][v][1] = count
        self.swap(self, start+index_C, start+position_to)
        self.B[self.find_block_number(self,  index_W)]-=1
        self.W[index_W][0] = count - 1

    @classmethod
    def set_edge_position_C(self, v, w, position_to):
        self.A[v][w][1] = position_to
        self.A[w][v][1] = position_to

    @classmethod
    def print_all_inserted_edges(self):
        for (idx, (count, start, end)) in enumerate(self.W):
            print(f"Weight class {idx}: {[' '.join(f'({str(tups[0])},{str(tups[1])})') for tups in self.C[start:start+count]]}")

    def find_block_number(self, weight_index):
        return math.floor((weight_index) / self.block_size) # shifts counting to correct oint arithmetic

    def swap(self, pos_from_C, pos_to_C):
        intermediate = self.C[pos_from_C]
        self.C[pos_from_C]  = self.C[pos_to_C]
        self.C[pos_to_C] = intermediate
        #print(f'swapped {pos_to_C} with {pos_from_C}')
