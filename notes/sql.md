# SparkSQL

## Important definitions

1. **Data Model:**
   - A data model is a conceptual representation of how data is organized and structured in a database. It defines the relationships between different data elements, the constraints on the data, and the rules governing data manipulation and integrity.

2. **Schema:**
   - In the context of databases, a schema is a blueprint or structural representation that defines the organization of data within a database. It includes information about the tables, fields, data types, and relationships, providing a framework for storing and retrieving data.

3. **Relational Data Model:**
   - The relational data model is a type of data model that represents data as tables (relations) consisting of rows and columns. It is based on mathematical set theory and defines relationships between tables. The data in this model is organized to support easy querying, retrieval, and manipulation using structured query language (SQL).

4. **Structured vs. Semi-Structured vs. Unstructured Data:**
   - **Structured Data:** Data that is organized in a fixed format, usually within a database, with a well-defined schema. Examples include relational databases where data is stored in tables.
   - **Semi-Structured Data:** Data that does not conform to a rigid structure like structured data but has some level of organization. It may be represented in formats such as JSON or XML, allowing for flexibility in data representation.
   - **Unstructured Data:** Data that lacks a predefined data model or structure. It includes data like text documents, images, audio, and video files, which do not fit neatly into traditional database tables.

5. **Relational Database:**
   - A relational database is a type of database that uses a relational data model to organize and store data. It consists of tables, where each table represents an entity, and relationships between tables are defined by common fields. Relational databases are managed using a relational database management system (RDBMS), and SQL is commonly used for querying and manipulating data.

6. **Two Parts of a Relation: Schema, Instance:**
   - **Schema:** The schema of a relation defines the structure of the data, including the names of the attributes (columns), the data types of each attribute, and any constraints on the data. It provides a blueprint for how the data should be organized.
   - **Instance:** The instance of a relation refers to the actual data stored in the relation at a particular point in time. It consists of rows, each representing a record or tuple, and each column containing the values corresponding to the attributes defined in the schema.

## 10 things to know about SparkSQL
1.  Spark SQL use cases
	- Ad-hoc querying of data in files
	- Interaction with external Databases
	- Live SQL analytics over streaming data
	- ETL capabilities alongside familiar SQL
	- Scalable clusters query performance with larger clusters
    
2.  Loading data: in the cloud vs locally, RDDs vs DataFrames

3.  SQL vs the DataFrame API. What’s the difference?
    
4.  Schemas: implicit vs explicit schemas, data types - Schemas can be inferred, i.e. guessed, by spark. With inferred schemas, you usually end up with a bunch of strings and ints. If you have more specific needs, supply your own schema.
	```python
import pyspark.sql.types as types
schema = types.StructType([ types.StructField(‘id',types.IntegerType(), False), types.StructField(‘name',types.StringType()),						   types.StructField(‘company',types.StringType()),
types.StructField(‘state',types.StringType())
				  ])
peopleDF = spark.createDataFrame(peopleRDD, schema)
```
1.  Loading & saving results
    
6.  What SQL functionality works and what doesn’t?
    
7.  Using SQL for ETL
	- Tip 1: In production, break your applications into Tip 1: In production, break your applications into smaller apps as steps. I.e. “Pipeline pattern” 
	- Tip 2: When tinkering locally, save a small version of the dataset via spark and test against that.
	- Tip 3: If using EMR, create a cluster with your desired steps, prove it works, then export a CLI desired steps, prove it works, then export a CLI command to reproduce it, and run it in Data Pipeline to start recurring pipelines / jobs.
8.  Working with JSON data
	- spark.read.json("file:/json.explode.json").createOrReplaceTempView("json")
	- spark.sql("SELECT * FROM json").show()
	- spark.sql("SELECT x, explode(y) FROM json").show()
9.  Reading and Writing to an external SQL databases
	- To read from an  from an external database, you’ve got to  have your JDBC connectors (jar) in handy. In order to pass a jar package into spark, you'd use the --jars flag when starting pyspark.
	- df.write.jdbc(url=url, table="baz", mode=mode, properties=properties)
	- mysql_url="jdbc:mysql://localhost:3306/sakila?user=root&password=Ramvam01"
	- df.write.jdbc(mysql_url,table="actor1"mode="append")
10.  Testing your work in the real world
	- Construct your applications as much as you can in advance. Cloudy clusters are expensive.
	- Get really comfortable using .parallelize() to create dummy data.
	- If testing locally, do not load data from S3 or other similar types of cloud storage.
	- If you’re using big data, and many nodes, don’t use .collect() unless you intend to
	- In the cloud you can test a lot of your code reliably with a 1-node cluster.

