# Miscellaneos

### What is spark context in pyspark ?
In the context of Apache Spark, the term "SparkContext" refers to the entry point and the main interface for programming Spark with the programming language API (Application Programming Interface). 

SparkContext is responsible for coordinating the execution of Spark jobs within a cluster. It represents the connection to a Spark cluster and can be used to create RDDs (Resilient Distributed Datasets), which are the fundamental data structures in Spark. RDDs are immutable distributed collections of objects that can be processed in parallel.

When you create a Spark application, the first thing you typically do is create a SparkContext. SparkContext is used to set up various configuration parameters, such as the master URL (the address of the cluster manager), and it coordinates the execution of operations on the cluster.

**SparkContext is the "connection" to the Spark Cluster.**

1. **Connection to Cluster:** When you run a Spark application, it doesn't execute directly on your local machine; it often runs on a cluster of computers. SparkContext is the object that connects your Spark application to this cluster. It tells Spark how to access the cluster.

2. **Coordinates Operations:** Once connected, SparkContext coordinates the execution of operations on the cluster. When you perform Spark operations (like transformations and actions) in your code, SparkContext is responsible for distributing those tasks across the cluster.

3. **Resource Allocation:** SparkContext manages resources in the cluster. It decides how much memory and CPU each task in your Spark application can use.

4. **Creation of RDDs:** It's also responsible for creating Resilient Distributed Datasets (RDDs), which are fundamental data structures in Spark. RDDs are distributed collections of data, and SparkContext helps in creating, distributing, and managing these RDDs.

Here's a simplified analogy: Think of SparkContext as the control tower at an airport. It coordinates the takeoff and landing of planes (your Spark operations) and ensures that resources are allocated efficiently. It's the central authority that makes sure everything runs smoothly on the distributed cluster.

### What is SparkSession ?
A `SparkSession` is the entry point for reading data, performing transformations, and executing Spark SQL queries in Apache Spark. It was introduced in Spark 2.0 to provide a unified interface for working with data in Spark, combining functionalities that were previously provided by different contexts such as `SparkContext`, `SQLContext`, and `HiveContext`.

`SparkSession` is designed to simplify the usage of Spark and make it more user-friendly. It encapsulates the functionality of the older contexts and provides a single, unified API. The main features and purposes of `SparkSession` include:

1. **Unified Entry Point:** `SparkSession` serves as a single entry point for reading data, executing SQL queries, and working with various Spark features. It combines the functionality of `SparkContext`, `SQLContext`, and `HiveContext`.

2. **DataFrame and SQL API:** `SparkSession` provides a DataFrame and SQL API for working with structured and semi-structured data. DataFrames are distributed collections of data organized into named columns, similar to a table in a relational database.

3. **Reading and Writing Data:** `SparkSession` simplifies the process of reading data from external sources (like Parquet, Avro, JSON, CSV, etc.) and writing data back to external storage systems. It supports a wide range of data formats.

4. **Configuration:** `SparkSession` allows you to configure various Spark settings and properties.

Here's a simple example of creating a `SparkSession` in Python:

```python
from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder \
    .appName("MySparkApp") \
    .getOrCreate()

# Your Spark code goes here

# Stop the SparkSession when done
spark.stop()
```

In this example, `appName` sets the name of the application, and `getOrCreate` either retrieves an existing `SparkSession` or creates a new one if it doesn't exist.

Using `SparkSession`, you can easily transition between Spark's SQL, DataFrame, and RDD (Resilient Distributed Dataset) APIs, making it a powerful tool for data processing and analysis in Spark applications.

### SparkSession vs SparkContext ?
`SparkContext` and `SparkSession` serve different purposes in Apache Spark, and they are used at different levels of abstraction.

1. **SparkContext:**
   - **Purpose:** `SparkContext` is the entry point and represents the connection to a Spark cluster. It is responsible for coordinating the execution of Spark jobs on a cluster.
   - **Functionality:** It is used to create Resilient Distributed Datasets (RDDs), the fundamental data structure in Spark, and to perform low-level Spark operations.
   - **Usage:** Prior to Spark 2.0, `SparkContext` was the main entry point for Spark applications. You would create a `SparkContext` to interact with the Spark cluster and perform distributed data processing.

   Example (in Python):
   ```python
   from pyspark import SparkContext

   # Create a SparkContext
   sc = SparkContext("local", "MySparkApp")

   # Your Spark code goes here

   # Stop the SparkContext when done
   sc.stop()
   ```

2. **SparkSession:**
   - **Purpose:** `SparkSession` is introduced in Spark 2.0 to provide a unified entry point for reading data, executing SQL queries, and performing various Spark operations. It encapsulates the functionality of `SparkContext`, `SQLContext`, and `HiveContext`.
   - **Functionality:** It simplifies the API for users by providing a higher-level abstraction. `SparkSession` includes methods for creating DataFrames, executing SQL queries, and working with structured data.
   - **Usage:** In Spark 2.0 and later versions, it is recommended to use `SparkSession` as the entry point for Spark applications.

   Example (in Python):
   ```python
   from pyspark.sql import SparkSession

   # Create a SparkSession
   spark = SparkSession.builder.appName("MySparkApp").getOrCreate()

   # Your Spark code using DataFrames and SQL goes here

   # Stop the SparkSession when done
   spark.stop()
   ```

In summary, while `SparkContext` is lower-level and focuses on the distributed processing capabilities of Spark, `SparkSession` is a higher-level abstraction that provides a more convenient interface, especially for working with structured data using DataFrames and executing SQL queries. In modern Spark applications, it's common to use `SparkSession` for most tasks, and it internally manages the `SparkContext`.

### How to use spark when multiple nodes are involved like ec2 instances
### Do RDDs have a data type. If not, how do we find out which sort of operation is valid for a given rdd. Like I should have some sort of check to stop someone from applying dictionary operations on an rdd that was given lists.
### 