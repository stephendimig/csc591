#
# File: degreeTest.py
# Author: Stephen Dimig (sdimig@netapp.com)
# Description:
#

import unittest
import degree as deg
import sys
import pandas as pd
import networkx as nx
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from graphframes import *
from tabulate import tabulate
import unittest


#
# Class:
# Description:
#
class degreeTest(unittest.TestCase):
    #
    # Name: test
    # Description:
    #
    # Parameters:
    # None
    #
    # Returns:
    # None
    #
    def test_readFile(self):
        filename = "test.csv"
        sc = SparkContext("local", "degreeTest.py")
        sqlContext = SQLContext(sc)
        g = deg.readFile(filename, False, sc, sqlContext)
        print(g.vertices.count())
        print(g.edges.count())



if __name__ == '__main__':
    unittest.main()