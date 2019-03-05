from igraph import *
import pandas as pd
from tabulate import tabulate
from scipy import spatial
import numpy as np
import operator
import sys
import copy
import argparse

class MyGraph(object):
    """MyGraph class"""
    def __init__(self):
        """Constructor for MyGraph class"""
        self.vertices = {}
        self.label2index = {}
        self.index = 0

    def add_vertex(self, label):
        """Add a vertex to the graph"""
        vertex = MyVertex(label, self.index)
        self.vertices[self.index] = vertex
        self.label2index[label] = self.index
        self.index += 1

    def add_edge(self, from_label, to_label, weight=sys.maxint):
        """Add an edge to the graph with an optional weight"""
        from_index = self.label2index[from_label]
        to_index = self.label2index[to_label]
        edge = MyEdge(self.vertices[from_index], self.vertices[to_index], weight)
        from_vertex = self.vertices[from_index]
        from_vertex.add_edge(edge)

    def remove_vertex(self, label):
        """Remove a vertex from the graph"""
        index = self.label2index[label]
        vertex = self.vertices[index]
        for vertex in [v for v in self.vertices.values() if v.get_index() != index]:
            vertex.remove_edge(index)
        del self.vertices[index]
        del self.label2index[label]

    def contains(self, vertex_label):
        """Check to see if vertex is in the graph"""
        return vertex_label in self.label2index.keys()

    def get_size(self):
        """Return number of vertices |V|"""
        return len(self.vertices)


    def get_edges(self, set1, set2):
        """Return the edges between two vertices"""
        retval = []
        for edge in reduce(operator.concat, [vert.get_edges() for vert in self.vertices.values()]):
            if edge.get_from_vertex().get_label() in set1 and \
                    edge.get_to_vertex().get_label() in set2:
                retval.append(edge)
        return retval

    def __str__(self):
        """Convert graph to string representation"""
        retval = ""
        for vertex in self.vertices.values():
            retval = retval + str(vertex) + "\n"
        return retval


class MyVertex(object):
    """MyVertex class"""

    def __init__(self, label, index):
        """Constructor for the MyVertex class"""
        self.index = index
        self.label = label
        self.edges = []

    def __eq__(self, other):
        """Equality operator"""
        return self.label == other.label

    def get_label(self):
        """Return the label associated with the vertex"""
        return self.label

    def set_label(self, label):
        """Set the label for the vertex"""
        self.label = label

    def get_index(self):
        """Return the index value"""
        return self.index

    def add_edge(self, edge):
        """Add an edge to the vertex"""
        self.edges.append(edge)

    def get_edges(self):
        """Get edges from vertex"""
        return self.edges

    def remove_edge(self, to_index):
        """Remove an edge from a vertex"""
        idx = -1
        pos = 0
        for edge in self.edges:
            if to_index in edge.get_to_vertex().get_index():
                idx = pos
                break
            pos += 1
        if idx != -1:
            del self.edges[idx]

    def get_number_edges(self):
        """Return the number of edges"""
        return len(self.edges)

    def has_edge(self, vertex):
        retval = False
        for edge in self.edges:
            if edge.get_to_vertex() == vertex:
                retval = True
                break
        return retval

    def __str__(self):
        """Return a string representation of a MyVertex"""
        retval = "MyVertex: label={}; index={};  edges=[". \
            format(self.label, self.index)
        for edge in self.edges:
            retval = retval + str(edge) + ","
        retval = retval + "]"
        return retval


class MyEdge(object):
    """MyEdge class"""

    def __init__(self, from_vertex, to_vertex, weight=sys.maxint):
        """Constructor for the MyEdge class"""
        self.from_vertex = from_vertex
        self.to_vertex = to_vertex
        self.weight = weight

    def __eq__(self, other):
        """Equality operator"""
        return self.from_vertex == other.from_vertex and self.to_vertex == other.to_vertex

    def get_from_vertex(self):
        """Return the from vertex"""
        return self.from_vertex

    def get_to_vertex(self):
        """Return the to vertex"""
        return self.to_vertex

    def get_weight(self):
        """Return the weight of the edge"""
        return self.weight

    def __str__(self):
        """Return string representation of an MyEdge"""
        return "MyEdge: from={}; to={}; weight={}". \
            format(self.from_vertex.get_label(), self.to_vertex.get_label(), self.get_weight())


class Communities(object):
    def __init__(self, g1):
        self.communities = []
        self.communities_map = {}
        for vertex in g1.vertices.values():
            community = MyCommunity()
            community.__add_vertex__(vertex)
            self.communities_map[vertex.get_label()] = community
            self.communities.append(community)
        self.current = 0

    def find_community(self, vertex):
        return self.communities_map[vertex.get_label()]

    def add_vertex(self, vertex, community):
        community.__add_vertex__(vertex)
        self.communities_map[vertex.get_label()] = community

    def remove_vertex(self, vertex):
        self.communities_map[vertex.get_label()].__remove_vertex__(vertex)
        del self.communities_map[vertex.get_label()]

    def size(self):
        retval = 0
        for community in self.communities:
            retval += len(community.vertices)
        return retval


class MyCommunity(object):
    def __init__(self):
        self.vertices = []

    def __add_vertex__(self, vertex):
        self.vertices.append(vertex)

    def __remove_vertex__(self, vertex):
        idx = -1
        for i in range(0, len(self.vertices)):
            if vertex.get_label() == self.vertices[i].get_label():
                idx = i
                break
        if idx >= 0:
            del  self.vertices[idx]

    def simA(self, vertex_a, vertex_b, attr_df):
        index_a = int(vertex_a.get_label())
        index_b = int(vertex_b.get_label())
        from_attr = np.array(attr_df.iloc[index_a].values.tolist())
        to_attr = np.array(attr_df.iloc[index_b].values.tolist())
        weight = 1 - spatial.distance.cosine(from_attr, to_attr)
        return weight

    def similarity_measure(self, vertex, attr_df, alpha):
        retval = 0.0
        verts =  self.vertices
        for i in range(0, len(verts)):
            Gij = 1 if verts[i].has_edge(vertex) else 0
            Sij = (alpha * Gij) + ((1 - alpha) * self.simA(verts[i], vertex, attr_df))
            retval += Sij
        return retval / (2.0 * (len(verts) + 1))

    def similarity_measure_comm(self, community, attr_df, alpha):
        retval = 0.0
        if community:
            verts =  self.vertices + community.vertices
        else:
            verts = self.vertices

        for i in range(0, len(verts)):
            for j in range(0, len(verts)):
                Gij = 1 if verts[i].has_edge(verts[j]) else 0
                Sij = (alpha * Gij) + ((1 - alpha) * self.simA(verts[i], verts[j], attr_df))
                retval += Sij
        return retval / (2.0 * (len(verts) * len(verts)))

    def merge(self, community):
        self.vertices =  sorted(self.vertices + copy.deepcopy(community.vertices), key=lambda x: int(x.label))

    def empty(self):
        self.vertices = []

    def is_empty(self):
        return False if self.vertices else True

    def __str__(self):
        """Return string representation of an MyCommunity"""
        return "{}".format(str([int(vertex.get_label()) for vertex in self.vertices]))


if __name__ == '__main__':
    MAX_ITERATIONS = 15
    c_args = argparse.ArgumentParser(description='Attributed Graph Community Detection')
    c_args.add_argument('--alpha', help='Optional alpha parameter', default="0.0")
    cli_args = c_args.parse_args()
    alpha = float(cli_args.alpha)

    edge_df = pd.read_csv('data/fb_caltech_small_edgelist.txt',header=None, delimiter=r"\s+")
    attr_df = pd.read_csv('data/fb_caltech_small_attrlist.csv')

    g1 = MyGraph()
    verts = set([str(row[0]) for idx, row in edge_df.iterrows()] + [str(row[1]) for idx, row in edge_df.iterrows()])
    for vert in verts:
        g1.add_vertex(vert)

    for idx, row in edge_df.iterrows():
        g1.add_edge(str(row[0]), str(row[1]))

    # Phase One
    communities = Communities(g1)
    vertices = sorted(g1.vertices.values(), key=lambda x: int(x.label))
    for i in range(0, len(vertices)):
        target = vertices[i]
        communities.remove_vertex(target)
        max_qattr = -sys.maxint - 1
        max_community = None
        for community in communities.communities:
            if community.is_empty():
                continue
            qattr = community.similarity_measure(target, attr_df, alpha)
            if qattr > max_qattr:
                max_qattr = qattr
                max_community = community
        if max_community:
            communities.add_vertex(target, max_community)

    # Phase Two
    for k in range(0, MAX_ITERATIONS):
        num_merged = 0
        for i in range(0, len(communities.communities) - 1):
            community_a = communities.communities[i]
            if community_a.is_empty():
                continue
            max_qattr = 0
            max_community = None
            for j in range(i, len(communities.communities)):
                community_b = communities.communities[j]
                if community_b.is_empty():
                    continue

                qattr = community_a.similarity_measure_comm(community_b, attr_df, alpha)  - community_a.similarity_measure_comm(None, attr_df, alpha)
                if qattr > max_qattr and qattr > 0.005:
                    max_qattr = qattr
                    max_community = community_b

            if max_community:
                num_merged += 1
                max_community.merge(community_a)
                community_a.empty()

        if not num_merged:
            break

    for community in communities.communities:
        if not community.is_empty():
            print str(community)[1:-1]

