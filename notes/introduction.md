# Introduction
- Learning Objectives
	- Distributed Computing Principles
	- Computationally Distribute ML Algos
	- Spark
- Course Website - https://www.andrew.cmu.edu/user/lakoglu/courses/95869/index.htm
- “Big Data” is data whose scale, diversity, and complexity require new architecture, algorithms, tools, and analytic techniques to manage it and extract value and hidden knowledge from it.
- Big Data 5Vs
	- Volume - TBs of existing data to process
	- Velocity - Streaming Data
	- Variety - Data in many forms (structured,unstructured, text)
	- Veracity - Data in doubt, incosistency due to incompleteness ambiguities etc.
	- Value - Business Value
- Hadoop
	- a distributed system that offers a way to parallelize and execute programs on a cluster of machines over substantial amounts of data
	- a platform that provides both distributed storage and distributed computation capabilities.
	- has a distributed master-slave architecture that consists of
		1.  Hadoop Distributed File System ( HDFS ) for storage    
		2.  and MapReduce for computational capabilities
- Explain hadoop master slave architecture
	Hadoop is an open-source framework designed for distributed storage and processing of large data sets using a cluster of commodity hardware. The master-slave architecture is a key component of Hadoop, providing a scalable and fault-tolerant solution for big data processing. Here's an overview of the Hadoop master-slave architecture:
	
	1. **Master Node (NameNode):**
	   - The master node in Hadoop is called the NameNode. It manages the metadata and keeps track of the file system namespace and the structure of the data stored in Hadoop Distributed File System (HDFS).
	   - The NameNode is a single point of failure in a Hadoop cluster. Therefore, it is crucial to have mechanisms like secondary NameNode and HA (High Availability) configurations to address this vulnerability.
	
	2. **Secondary Master Node (Secondary NameNode):**
	   - The Secondary NameNode is not a backup or failover NameNode. Instead, it periodically merges the namespace and edits from the main NameNode, creating a new, updated file system image. This helps in preventing the main NameNode's metadata from becoming too large and provides a checkpoint that can be used to recover the file system metadata.
	
	3. **Slave Nodes (DataNodes):**
	   - The slave nodes in Hadoop are called DataNodes. These nodes are responsible for storing and managing the actual data. Each DataNode is responsible for managing the storage attached to the node and performing read and write operations as instructed by the NameNode.
	   - DataNodes are distributed across the Hadoop cluster and work in conjunction with the NameNode to ensure data reliability and fault tolerance.
	
	4. **JobTracker:**
	   - In addition to HDFS, Hadoop also includes a distributed data processing framework known as MapReduce. The JobTracker is responsible for managing and coordinating the MapReduce jobs in the cluster.
	   - The JobTracker keeps track of the available TaskTrackers in the cluster and assigns tasks to them. It monitors the progress of tasks and handles task failures by rescheduling them on available nodes.
	
	5. **TaskTracker:**
	   - TaskTrackers run on each slave node and are responsible for executing tasks assigned to them by the JobTracker. These tasks typically include Map and Reduce tasks as part of a MapReduce job.
	   - TaskTrackers report their status back to the JobTracker, allowing it to monitor progress and detect task failures.
- Apache Spark
	- a fast, general-purpose, open-source cluster computing engine
	- well suited for machine learning 
	- data objects stored in resilient distributed datasets (RDDs)
	- provides high-level APIs and a rich set of higher-level tools:
		- Spark SQL for SQL and structured data processing
		- MLlib for Machine Learning § GraphX for graph processing
		- GraphX for graph processing
		- Spark Streaming
- Data Processing
	- Interactive Queries on historical data - Apache HBASE
	- Batch - Hadoop, Hive
	- Streaming - Spark
- Spark vs Hadoop
| Feature                       | Hadoop                                   | Spark                                      |
|-------------------------------|------------------------------------------|--------------------------------------------|
| **Processing Paradigm**       | Batch-oriented processing (MapReduce)    | Batch and real-time processing              |
| **Data Processing Engine**     | MapReduce                                | Spark Core (supports batch processing)     |
|                               |                                          | Spark SQL (supports SQL queries)           |
|                               |                                          | Spark Streaming (real-time data processing)|
|                               |                                          | MLlib (machine learning library)           |
|                               |                                          | GraphX (graph processing library)          |
| **Ease of Use**               | Steeper learning curve                   | Easier to use and programmatic APIs        |
| **Performance**               | Slower due to disk-based processing      | Faster in-memory processing               |
| **Data Processing Speed**     | Typically slower due to disk I/O         | Faster due to in-memory computation        |
| **Fault Tolerance**           | High fault tolerance (HDFS replication)  | Fault tolerance through lineage information|
|                               |                                          | and recomputation                          |
| **Storage System**            | Hadoop Distributed File System (HDFS)    | Can use HDFS or other storage systems      |
| **Caching**                   | Limited caching capabilities             | Extensive support for in-memory caching    |
| **Use Cases**                 | Suitable for batch processing of large   | Suitable for batch, real-time, and         |
|                               | datasets and ETL processes               | iterative processing tasks                 |
| **Integration**               | Integrates well with Hadoop ecosystem     | Can run independently or integrate with    |
|                               |                                          | Hadoop ecosystem                           |
| **Community Support**         | Large and mature community               | Growing community with active development |
| **Programming Languages**     | Primarily Java, supports other languages | Supports Java, Scala, Python, and R        |
| **Resource Management**       | MapReduce relies on Hadoop's resource    | Built-in cluster manager (Spark Standalone,|
|                               | manager (YARN or Hadoop 1.0's JobTracker)| supports Apache Mesos, Hadoop YARN)        |
