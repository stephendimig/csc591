#
# File: degreeTest.py
# Author: Stephen Dimig (sdimig@netapp.com)
# Description:
#

import unittest
import degree as deg
from mock import MagicMock

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
        filename = "9_11_edgelist.txt"
        sc = SparkContext("local", "degree.py")
        sqlContext = SQLContext(sc)

        deg.readFile(filename, False, sc, sqlContext=sqlContext)



if __name__ == '__main__':
    unittest.main()