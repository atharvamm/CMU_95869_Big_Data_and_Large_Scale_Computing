# Quiz Notes


**Data Analysis Procedures Requiring (Near) Real-Time Response:**
- Interactive querying
- Stream processing

**Hadoop Components:**
- MapReduce: Used for computation
- Hadoop Distributed File System (HDFS): Used for storage

**Hadoop vs Spark:**
- **Spark:**
  - Built-In ML Library
  - Fast in-memory parallelization
  - Easier to program

- **Hadoop:**
  - Comes with built-in storage system

**5Vs of Big Data:**
- **Volume:** Large Scale Computing
- **Velocity:** Online Analytics
- **Veracity:** Noise
- **Variety:** Multiple Data Sources
- **Value:** Business Insights

**Big Data Challenges:**
- Processing big data is a bigger challenge than storing it.
- True
  - Disk storage costs and sizes are decreasing.
  - Acquiring and storing large amounts of data is not the biggest challenge.
  - The challenge lies in extracting meaningful insights and models through large-scale processing of big data.

- **Challenges from Distributed Processing:**
  - Task split and aggregation
  - Machine failure likelihood
  - Load balancing

- **Why is slow disk access speed not a challenge for distributed systems?**
  - Disk access is not a specific issue for distributed platforms; it is also an issue for a single-machine server.
  - Network access speed is an issue for distributed systems as machines in a cluster need to communicate intermediate results over the network.

- **In a Hadoop map-reduce job, there are multiple map tasks and a single reduce task?**
  - False
    - Map-reduce jobs typically contain multiple map tasks and multiple reduce tasks, with results residing across multiple (reducer) machines.

- **Basic differences between relational database (RDBMS) and HDFS:**
  - Hadoop scales better than RDBMS for large-scale data (scale-out).
  - Hadoop can process data (like complex calculation) while RDBMS can only answer data queries.
  - Hadoop can store any kind of data while RDBMS only works on structured data.

- **Hadoop supports real-time online data processing while RDBMS can only do offline batch.**
  - False
    - Hadoop does not support online data processing; it was built for batch processing.

- **Both Hadoop and RDBMS are open-source applications.**
  - False
    - Most RDBMS are not open source.

- **With already fast enough network and storage speed, enlarging storage is a major bottleneck for handling big data.**
  - False
    - Disk is cheap, with prices dropping exponentially, and adding more disk to any machine is feasible without much hurdle.

- **Features of Hadoop:**
  - Large-scale batch processing
  - Handling structured and unstructured data

- **What are not features of Hadoop:**
  - Low fault tolerance
  - Interactive queries

- **Hadoop's MapReduce system handles worker node failure by simply re-scheduling all tasks of the failed node to one or more other workers.**
	  - True

- **When scaling up hits a wall and/or is too expensive, scale out!**

**Comparison of Spark with Hadoop's MapReduce:**
- Key advantage of Spark over Hadoop's MapReduce: Uses memory cache instead of disk, speeding up computation through in-memory processing and minimizing disk accesses.

**Features of RDDs (Resilient Distributed Datasets):**
- Partitioned data is immutable and distributed.
- Object types are opaque to Spark.
- Transformations are lazily evaluated.
- Multiple ways to construct RDDs, with common methods being parallelized collections and reading in files.

**Comparison of DataFrame and RDD:**
- False statement: DataFrame is more efficient than RDD for managing unstructured data like multimedia. RDD works over unstructured data, while DataFrame is designed for structured data, making processing on DataFrames more efficient due to the known schema.

**Properties of Binary Function $f(a, b) = \frac{a+b}{2}$:**
- Commutative: $f(a, b) = f(b, a)$ for all $a$ and $b$.
- Not associative: $f(f(a, b), c) \neq f(a, f(b, c))$ for some values of $a$, $b$, and $c$. Understanding these properties is crucial in algebra and mathematical structures.

**RDD Operations in Spark:**
- Example RDD: `rdd = sc.parallelize([1, 2, 3, 4])`
  - Get odd values: `rdd.filter(lambda x: x%2 != 0)`
  - Get first 3 elements: `rdd.take(3)`
  - Add 1 to even numbers: `rdd.map(lambda x: x+1 if x%2 == 0 else x)`

**RDD Operations with `reduceByKey` in Spark:**
- Example RDD: `rdd = sc.parallelize([1, 4, 2, 3, 4, 4, 2, 4])`
  - Transform into key-value pairs: `a = rdd.map(lambda x: (x, 1))`
  - Reduce by key (sum values): `b = a.reduceByKey(lambda x, y: x+y)`
  - Reduce by key (sum values plus 1): `c = a.reduceByKey(lambda x, y: x+y+1)`
    - Note: `reduceByKey` operates over pairs at a time, and it is highlighted that for certain keys, there may not be an add operation. Additionally, when counting 4, the add operation is executed three times as 4 records with key 4 are paired three times to obtain the total sum of values.

**Unsupervised Learning:**
- Unsupervised learning is often considered more suitable for handling complicated input data types such as images and audio when compared to supervised learning.
- The statement that unsupervised learning is more suitable for complicated input data types is false.

**Feature Extraction:**
- Feature extraction is a crucial step in the learning process.
- The goal is to define and measure good descriptions of observations that are most relevant to the given task.
- Feature engineering plays a key role in tasks like spam or fraud detection, where extracting key descriptors of fraud is essential for effective learning.
- If the input data is of poor quality, the learning outcome is likely to be poor as well—following the principle of garbage in, garbage out.

**Training and Test Sets:**
- During training, data is typically divided into a training set and a validation set.
- During testing, a separate test set is used to evaluate the model's performance.

**Supervised vs. Unsupervised Learning Examples:**
- Examples of supervised learning include predicting the number of new daily confirmed cases of Covid-19 and estimating the time until death for heart-attack patients.
- An example of unsupervised learning is identifying underlying topics from a large collection of news articles.

**Linear Algebra Calculation:**
- Let M be the outer product of $x_1 = [-1, -3, -3, -1, 1]$ and $x_2 = [1, 2, 3]$, i.e., $x_1^Tx_2$.
- The resulting matrix is:
  $$
\begin{bmatrix}
  -1 & -2 & -3 \\
  -3 & -6 & -9 \\
  -3 & -6 & -9 \\
  -1 & -2 & -3 \\
   1 &  2 &  3 \\
\end{bmatrix}
$$

  
- The sum of the third row is $[-3, -6, -9] = 126$.

**Big O Notation:**
- Rank the following expressions in terms of increasing order of growth: $f_1(n) = 2^{\log_{50}{n}}$, $f_2(n) = 2^{\log_2{n}} = n$, $f_3(n) = n\log_2{n}$.
  - $f_1 < f_2 < f_3$
- The expression $f_1(n) = 2^{\frac{\log_2{n}}{\log_2{50}}}$ is simplified using the change of base formula, resulting in $f_1(n) = n^{1/\ln{50}}$, where $\ln{50}$ is a constant.

**Big O Misconceptions:**
- $2^{2n} = O(2^n)$ is incorrect. It should be $Cg(n)$, as the left-hand side grows quickly with increasing $n$.
- For positive values, $f(n) = O(g(n))$ implies $f(n)^2 = O(g(n)^2)$, which is correct. Quadratic growth applies directly to big O.
- $2^{n+1} = O(2^n)$ is correct since $2^{n+1} = 2 \cdot 2^n$ and is effectively $O(2^n)$.
- $3^n = O(2^n)$ is incorrect because $2^n$ cannot bound $3^n$.


**Why Use Gradient Descent for Optimization Problems:**
- Closed form solution may not exist.
- Linear complexity makes it scalable.
- Additive nature facilitates easy distribution.

**Reducing Network Communication Cost on a Distributed Platform:**
- Leverage distributed disk storage for parallel computing.
- Choose algorithms with linear computation and storage requirements.
- Minimize network communication by computing locally and combining results.

**Catch in Using Gradient Descent for Linear Regression:**
- Advantage: Scalable with linear time and space requirements.
- Trade-off: Iterative and requires multiple rounds of network communication, potentially leading to suboptimal solutions.

**Logistic Regression:**

- Logistic regression is a classification method that aims to model the relationship between input features and the dependent variable, but it differs from linear regression in both methodology and objectives.
  
  - **False Statement:**
    - The assertion that logistic regression shares the same learning objective with linear regression is false. Despite their common goal of modeling relationships, they employ different methodologies.

    - **Reasons for Inaccuracy:**
      1. **Objective Function:**
         - Linear regression minimizes mean squared error, while logistic regression maximizes likelihood or minimizes log-loss of predicted probabilities.
      2. **Output Type:**
         - Linear regression produces continuous output, while logistic regression predicts probabilities for classification.
      3. **Modeling Binary Outcomes:**
         - Logistic regression is designed for binary classification, unlike linear regression, which might not be suitable for such tasks.
      4. **Decision Boundary:**
         - Logistic regression employs a sigmoid function, introducing non-linearity and capturing complex relationships, whereas linear regression uses a linear decision boundary.


**Classification Models and Categorical Features:**

- Which of the following classification models can readily handle categorical features as input?

  - - Logistic regression - No
    - Neural networks  - No
    - Decision trees   - Yes
    - Naive Bayes classifier - Yes


**Binary Classification Thresholds:**

- In binary classification, the threshold for classifier output scores or probabilities is not always set to 0.5. This adjustment is made to control the trade-off between true positives and false positives when assigning discrete yes/no labels to test observations.


- **Feature Hashing Example:**
  - For m=4, the feature representation of an observation (flushot:yes, exercise:often):
    - Initially, [0 0 0 0]
      - Vowels (flushot:yes) % 4 = 3 % 4 = 3 (index 3 incremented by 1)
      - Vowels (exercise:often) % 4 = 6 % 4 = 2 (index 2 incremented by 1)
    - Resulting representation: [0 0 1 1]


**The principal components found by PCA are straight lines from the origin, such that each component is perpendicular to all the components before it.**

**Find principal components of a matrix X with size nXd:**
### Steps for PCA:
1. **Standardize the Data:**
   - If the features in your matrix X are measured in different units or have different scales, it's a good practice to standardize the data. Subtract the mean and divide by the standard deviation for each feature.

2. **Compute the Covariance Matrix:**
   - Calculate the covariance matrix Σ for the standardized data. The covariance matrix is a d × d matrix where each element σij represents the covariance between features i and j.
   - Σ = (1/n)X^TX

3. **Compute the Eigenvectors and Eigenvalues:**
   - Compute the eigenvectors and eigenvalues of the covariance matrix Σ. The eigenvectors represent the principal components, and the corresponding eigenvalues indicate the amount of variance explained by each principal component.
   - Σvi = λivi
   - Sort the eigenvectors based on their corresponding eigenvalues in descending order. The eigenvector with the highest eigenvalue corresponds to the first principal component, the second-highest to the second principal component, and so on.

4. **Choose the Number of Principal Components:**
   - Decide how many principal components to retain. This is often determined based on the percentage of total variance you want to capture. A common choice is to retain enough components to capture, for example, 95% or 99% of the total variance.

5. **Form a Projection Matrix:**
   - Construct a projection matrix P using the selected eigenvectors. If you choose k principal components, the projection matrix P will consist of the first k eigenvectors as columns.
   - P = [v1, v2, …, vk]

6. **Project the Data onto the New Feature Space:**
   - Multiply the original standardized data matrix X by the projection matrix P to obtain the transformed data matrix Xnew, which has reduced dimensionality.
   - Xnew = X ⋅ P
   - Each row in Xnew represents the data points in the new feature space defined by the principal components.

**How to compute eigenvalues and eigenvectors?**
Computing eigenvalues and eigenvectors involves solving the characteristic equation for a given matrix. Given a square matrix A, the characteristic equation is defined as:

det(A - λI) = 0

where λ is the eigenvalue, I is the identity matrix, and det(⋅) denotes the determinant.

Once you find the eigenvalues, you can compute the corresponding eigenvectors using the equation:

(A - λI)v = 0

### Computing Eigenvalues:
1. **Set Up the Characteristic Equation:**
   - Form the matrix A - λI.
   - Compute the determinant of A - λI and set it equal to zero.
   - det(A - λI) = 0

2. **Solve for λ:**
   - Solve the characteristic equation to find the eigenvalues λ.

### Computing Eigenvectors:
3. **For Each Eigenvalue λ:**
   - Substitute λ back into the equation (A - λI)v = 0.
   - Solve the system of linear equations to find the corresponding eigenvector v.

- **Passes Over Data for PCA:**
  - Question: How many passes over the data do we need to perform PCA?
  - Answer: 4 passes
  - Details:
    1. Compute Mean
    2. Normalize
    3. Covariance Matrix
    4. Final Scores

- **Non-Iterative PCA Approach on Spark Cluster:**
  - Statement: Using commodity machines in a Spark cluster with 1GB RAM each, for a dataset with large n and d=10,000 features, the non-iterative PCA approach can reduce dimensionality to k=100.
  - Fact: True
  - Explanation:
    - Covariance matrix is 10Kx10K
    - RAM space required: $0.1 * 8 * 10^9$ bytes = 0.8GB (less than 1GB)
    - A single machine can store and process the covariance matrix, enabling the non-iterative approach.

- **Disadvantages of Iterative PCA Approach:**
    - More network communication is needed in the iterative approach.
    - Iterative approach necessitates a larger number of passes over the data.

- **Importance of Caching/Persisting in Spark:**
  - For implementing solutions on Spark making multiple passes over the data, it is critical to cache/persist.

- **Implementing Iterative ML Algorithms with MLlib:**
  - Explanation: MLlib's `.train()` and `.fit()` functions handle the training specifics implicitly, abstracting away low-level operations like iterations, map, reduce, etc. No need to write a 'for' loop for training models.

**Hadoop Architecture and Secondary NameNode:**
- It does not manage memory resources for the NameNode.
- It takes snapshots of the NameNode for backup.
- In case of NameNode failure, Hadoop can create a new copy based on Secondary NameNode.
- It does not communicate with other DataNodes to replicate data blocks for redundancy.

**Differences Between RDBMS and Hadoop:**
- RDBMS follows a schema while hadoop hold key value pairs
- RDBMS follows a particular schema and handles only structured data, while Hadoop can handle both structured and unstructured data.

**Closed Form Solution of Linear Regression:**
- Adding the term $\lambda*I_d$ to $X^TX$ in the closed form solution of linear regression makes the determinant of the matrix to be inverted non-zero.

**Dimensionality of One-Hot-Encoding:**
- The dimensionality of One-Hot-Encoding is determined by the total number of distinct categorical feature values.

**Mini-Batch Gradient Descent:**
- It helps reduce network communication costs.
- It is not guaranteed to produce the optimal solution.
- It does not yield the lowest communication cost.
- It helps in performing more computation locally at the worker nodes.

**Use of Caching:** Caching persists frequently used data in main memory (RAM) to reduce subsequent access times.

**Distributed PCA and Matrix Multiplication:** For distributed PCA, we use the outer product to compute matrix multiplication in a distributed fashion.
