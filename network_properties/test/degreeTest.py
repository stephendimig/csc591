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

    @classmethod
    def setUpClass(cls):
        cls.sc = SparkContext("local", "degreeTest.py")
        cls.sqlContext = SQLContext(sc)

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
        g = deg.readFile(filename, False, degreeTest.sc, degreeTest.sqlContext)
        self.assertEquals(g.edges.count(), 1000)


    def test_simple(self):
        filename = "test.csv"
        g = deg.readFile(filename, False, degreeTest.sc, degreeTest.sqlContext)
        self.assertEquals(g.edges.count(), 1000)
        simple_g = deg.simple(g)

if __name__ == '__main__':
    unittest.main()