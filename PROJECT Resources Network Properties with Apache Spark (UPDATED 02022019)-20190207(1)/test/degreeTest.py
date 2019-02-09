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
import findspark

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
        findspark.init()
        filename = "9_11_edgelist.txt"
        sc = SparkContext("local", "degree.py")
        sqlContext = SQLContext(sc)

        deg.readFile(filename, False, sc, sqlContext=sqlContext)



if __name__ == '__main__':
    unittest.main()