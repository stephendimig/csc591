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
from pyspark.sql import SparkSession

import unittest
import logging
from pyspark.sql import SparkSession


class PySparkTest(unittest.TestCase):

    @classmethod
    def suppress_py4j_logging(cls):
        logger = logging.getLogger('py4j')
        logger.setLevel(logging.WARN)

    @classmethod
    def create_testing_pyspark_session(cls):
        return (SparkSession.builder
            .master('local[2]')
            .appName('degreeTest.py')
            .getOrCreate())

        @classmethod
        def setUpClass(cls):
            cls.suppress_py4j_logging()

        cls.spark = cls.create_testing_pyspark_session()

        @classmethod
        def tearDownClass(cls):
            cls.spark.stop()

#
# Class:
# Description:
#
class degreeTest(PySparkTest):
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
        filename = "9_11_edgelist.txt"
        sc = PySparkTest.create_testing_pyspark_session()
        sqlContext = None

        deg.readFile(filename, False, sc, None)



if __name__ == '__main__':
    unittest.main()