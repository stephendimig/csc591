from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from pyspark.sql.functions import explode

def closeness(g, sc, sqlContext):
	# Get list of vertices. We'll generate all the shortest paths at
	# once using this list.
	# YOUR CODE HERE
	vertices = [row['id'] for row in g.vertices.collect()]
	print(vertices)

	#vertices.map(lambda vertex: g.shortestPaths(landmarks=["a", "d"])
	# first get all the path lengths.


	# Break up the map and group by ID for summing

	# Sum by ID

	# Get the inverses and generate desired dataframe.

if __name__ == '__main__':
	sc = SparkContext("local", "degree.py")
	sqlContext = SQLContext(sc)

	print("Reading in graph for problem 2.")
	graph = sc.parallelize([('A','B'),('A','C'),('A','D'),
		('B','A'),('B','C'),('B','D'),('B','E'),
		('C','A'),('C','B'),('C','D'),('C','F'),('C','H'),
		('D','A'),('D','B'),('D','C'),('D','E'),('D','F'),('D','G'),
		('E','B'),('E','D'),('E','F'),('E','G'),
		('F','C'),('F','D'),('F','E'),('F','G'),('F','H'),
		('G','D'),('G','E'),('G','F'),
		('H','C'),('H','F'),('H','I'),
		('I','H'),('I','J'),
		('J','I')])

	e = sqlContext.createDataFrame(graph,['src','dst'])
	v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()
	print("Generating GraphFrame.")
	g = GraphFrame(v,e)

	print("Calculating closeness.")
	closeness(g, sc, sqlContext).sort('closeness',ascending=False).show()
