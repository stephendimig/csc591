import sys
import pandas as pd
import networkx as nx
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from graphframes import *
from tabulate import tabulate
from plfit import plfit
from itertools import islice

# return the simple closure of the graph as a graphframe.
def simple(g, sc, sqlContext):
	# Extract edges and make a data frame of "flipped" edges
	# YOUR CODE HERE
	eschema = StructType([StructField("src", IntegerType()), StructField("dst", IntegerType())])
	flipped = g.edges.rdd.map(lambda x: (x[1], x[0]))

	# Combine old and new edges. Distinctify to eliminate multi-edges
	# Filter to eliminate self-loops.
	# A multigraph with loops will be closured to a simple graph
	# If we try to undirect an undirected graph, no harm done
	# YOUR CODE HERE
	combined = flipped.union(g.edges.rdd).distinct()
	flipped_df = sqlContext.createDataFrame(combined, eschema)
	simple_g = GraphFrame(g.vertices, flipped_df)
	return simple_g



# Return a data frame of the degree distribution of each edge in the provided graphframe
def degreedist(g):
	# Generate a DF with degree,count
    # YOUR CODE HERE
	retval = g.degrees.groupBy(['degree']).count()
	return retval


# Read in an edgelist file with lines of the format id1<delim>id2
# and return a corresponding graphframe. If "large" we assume
# a header row and that delim = " ", otherwise no header and
# delim = ","
def readFile(filename, large, sc, sqlContext):
	vschema = StructType([StructField("id", IntegerType())])
	eschema = StructType([StructField("src", IntegerType()), StructField("dst", IntegerType())])

	lines = sc.textFile(filename)
	g = None
	if large:
		delim=" "
		# Strip off header row.
		lines = lines.mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it)
		#lines = lines.mapPartitionsWithIndex(lambda ind,it: iter(list(it)[1:]) if ind==0 else it)
	else:
		delim=","

	# Extract pairs from input file and convert to data frame matching
	# schema for graphframe edges.
	# YOUR CODE HERE
	print lines.filter(lambda line: line.split(delim)[0].isdigit() and line.split(delim)[1].isdigit()).collect()
	e = lines.filter(lambda line: line.split(delim)[0].isdigit() and line.split(delim)[1].isdigit()).\
		map(lambda line: (int(line.split(delim)[0]), int(line.split(delim)[1])))

	edf = sqlContext.createDataFrame(e, eschema)

	# Extract all endpoints from input file (hence flatmap) and create
	# data frame containing all those node names in schema matching
	# graphframe vertices
	# YOUR CODE HERE
	v  = lines.filter(lambda line: line.split(delim)[0].isdigit() and line.split(delim)[1].isdigit()).\
		flatMap(lambda line: line.split(delim)).\
		map(lambda x: int(x)).\
		distinct().\
		map(lambda x: (x, ))

	vdf = sqlContext.createDataFrame(v, vschema)

	# Create graphframe g from the vertices and edges.
	g = GraphFrame(vdf, edf)
	return g


if __name__ == '__main__':
	# main stuff
	sc=SparkContext("local", "degree.py")
	sqlContext = SQLContext(sc)

	# If you got a file, yo, I'll parse it.
	if len(sys.argv) > 1:
		filename = sys.argv[1]
		if len(sys.argv) > 2 and sys.argv[2]=='large':
			large=True
		else:
			large=False

		print("Processing input file " + filename)
		g = readFile(filename, large, sc, sqlContext)

		print("Original graph has " + str(g.edges.count()) + " directed edges and " + str(g.vertices.count()) + " vertices.")

		g2 = simple(g, sc, sqlContext)
		print("Simple graph has " + str(g2.edges.count()/2) + " undirected edges.")

		distrib = degreedist(g2)
		distrib.show()
		nodecount = g2.vertices.count()
		print("Graph has " + str(nodecount) + " vertices.")

		out = filename.split("/")[-1]
		print("Writing distribution to file " + out + ".csv")
		distrib.toPandas().to_csv(out + ".csv")

		fit = plfit([row['count'] for row in distrib.collect()])
		print fit.test_pl()
	# Otherwise, generate some random graphs.
	else:
		print("Generating random graphs.")
		vschema = StructType([StructField("id", IntegerType())])
		eschema = StructType([StructField("src", IntegerType()),StructField("dst", IntegerType())])

		gnp1 = nx.gnp_random_graph(100, 0.05, seed=1234)
		gnp2 = nx.gnp_random_graph(2000, 0.01, seed=5130303)
		gnm1 = nx.gnm_random_graph(100,1000, seed=27695)
		gnm2 = nx.gnm_random_graph(1000,100000, seed=9999)

		todo = {"gnp1": gnp1, "gnp2": gnp2, "gnm1": gnm1, "gnm2": gnm2}
		for gx in todo:
			print("Processing graph " + gx)
			v = sqlContext.createDataFrame(sc.parallelize(todo[gx].nodes()), vschema)
			e = sqlContext.createDataFrame(sc.parallelize(todo[gx].edges()), eschema)
			g = simple(GraphFrame(v,e), sc, sqlContext)
			distrib = degreedist(g)
			print("Writing distribution to file " + gx + ".csv")
			distrib.toPandas().to_csv(gx + ".csv")

			print [row['count'] for row in distrib.collect()]
			fit = plfit([row['count'] for row in distrib.collect()])
			print fit.test_pl()

