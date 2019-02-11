import sys
import time
import networkx as nx
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from copy import deepcopy
from pyspark.sql.types import *

def articulations(g, sc, sqlContext, usegraphframe=False):
	# Get the starting count of connected components
	# YOUR CODE HERE
	result = g.connectedComponents()
	number_connected = result.groupby(result.component).count().distinct().count()
	print "number_connected={}".format(number_connected)

	# Default version sparkifies the connected components process 
	# and serializes node iteration.
	if usegraphframe:
		# Get vertex list for serial iteration
		# YOUR CODE HERE
		vertices = [row['id'] for row in g.vertices.collect()]

		# For each vertex, generate a new graphframe missing that vertex
		# and calculate connected component count. Then append count to
		# the output
		# YOUR CODE HERE
		rows = []
		for vertex in vertices:
			new_vertices = [v for v in vertices if v != vertex]
			new_edges = g.edges.map(lambda edge: (edge['src'], edge['dst'])).\
				filter(lambda edge: edge[0] in new_vertices and edge[1] in new_vertices).\
				collect()
			e = sqlContext.createDataFrame(new_edges, ['src', 'dst'])

			# Extract all endpoints from input file and make a single column frame.
			v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()

			# Create graphframe from the vertices and edges.
			new_g = GraphFrame(v, e)
			new_number_connected = result.groupby(result.component).count().distinct().count()
			row = (vertex, 1 if new_number_connected > number_connected else 0)
			rows.append(row)

		schema = StructType([StructField("id", StringType()), StructField("articulation", IntegerType())])
		df = sqlContext.createDataFrame(rows, schema)
		return df

	# Non-default version sparkifies node iteration and uses networkx 
	# for connected components count.
	else:
		# YOUR CODE HERE
		pass

if __name__ == '__main__':
	sc = SparkContext("local", "articulation.py")
	sqlContext = SQLContext(sc)
	filename = sys.argv[1]
	lines = sc.textFile(filename)

	pairs = lines.map(lambda s: s.split(","))
	e = sqlContext.createDataFrame(pairs,['src','dst'])
	e = e.unionAll(e.selectExpr('src as dst','dst as src')).distinct() # Ensure undirectedness

	# Extract all endpoints from input file and make a single column frame.
	v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()

	# Create graphframe from the vertices and edges.
	g = GraphFrame(v,e)

	#Runtime approximately 5 minutes
	print("---------------------------")
	print("Processing graph using Spark iteration over nodes and serial (networkx) connectedness calculations")
	init = time.time()
	#df = articulations(g, sc, sqlContext, False)
	print("Execution time: %s seconds" % (time.time() - init))
	print("Articulation points:")
	#df.filter('articulation = 1').show(truncate=False)
	print("---------------------------")

	#Runtime for below is more than 2 hours
	print("Processing graph using serial iteration over nodes and GraphFrame connectedness calculations")
	init = time.time()
	df = articulations(g, sc, sqlContext, True)
	print("Execution time: %s seconds" % (time.time() - init))
	print("Articulation points:")
	df.filter('articulation = 1').show(truncate=False)
