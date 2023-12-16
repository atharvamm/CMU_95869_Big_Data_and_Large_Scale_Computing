# Code
- sc = SparkContext()
## HW 1
- sc.parallelize = Pass python object and number of nodes to generate rdd
- rdd.map = Use a lambda function to do some sort of operation on each rdd element.
- A pair RDD is an RDD where each element is a pair tuple `(k, v)` where `k` is the key and `v` is the value. To create pairrdd rdd.map(lambda x: (x,1))
- groupByKey()
	- Basically group all pairs generated using pair RDD in the previous step to get a  group where the first element is the key and maybe a list of value elements.
	- Two problems with using `groupByKey()`: 
		- The operation requires of data movement to move all values into appropriate partitions.
		- The lists can be very large. Consider a word count of English Wikipedia: the lists for common words (the, a) would be huge and could exhaust available memory in a worker.
- reduceByKey()
	- gathers together pairs that have the same key and applies the function provided to two values at a time, iteratively reducing all of the values to a single value
	- operates by applying the function first within each partition on a per-key basis and then across the partitions
	- significantly reduces the amount of data being shuffled because only unique keys with aggregated counts are shuffled. It should be the most efficient method among three methods(collect and then counter, groupbykey and then length, reducebykey()).
	- rdd.reduceByKey(lambda x,y:x+y/some other operation)
	- [reduce and reduceByKey need commutative and associatve functions](https://stackoverflow.com/questions/35205107/spark-difference-of-semantics-between-reduce-and-reducebykey)
	- [Commuative and Associative](https://www.mathsisfun.com/associative-commutative-distributive.html)
- rdd multiple operation in one step
  ```
  finalrdd = (rdd
                .map(lambda x : [x,1])
                .reduceByKey(lambda x,y:x+y)
                .collect())
	```
- Find longest word using reduce
	- `longestWord = (rdd.reduce(lambda a,b : a if len(a) > len(b) else b))`
	- Basically you can also return another element using reduce function
- What will happen if reduce/reduceByKey function is not associative and commutative ? = You can get unpredictable results based on how the data is divided across partitions/nodes.
- To load text file `rdd = sc.textFile(fileName, numNodes)`
- `topN = rdd.takeOrdered(N, key=lambda x: -x[1])`, with takeOrdered you can sort the rdd and take topN elements
- `getN = rdd.take(N)`, get N elements.[doc](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.RDD.take.html)
## HW 2
- .count() = Count the number of object instances in the rdd
- [LabeledPoint(label,features)](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.regression.LabeledPoint.html) = In MLlib, labeled training instances are stored using the object.
- Lets say you made a labeled point and then want to change some feature's value how would you do that ? - Basically make a new function where you take in the point and do all operations and then return the point object. In this way you will not have to create a new instance of labeled point. 
```
def func(pt):
	    pt.label = pt.label - 3
	    return pt
newrdd = rdd.map(lambda pt: func(pt))  
```
- Use the [randomSplit method](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.RDD.randomSplit.html) with the specified weights and seed to create RDDs storing each of these datasets. `train, val, test = rdd.randomSplit(weights(1*3 arr),seed)`
- What is a dense vector ? - A dense vector is a data structure used to represent an array of values where most of the elements are non-zero. It is part of the MLlib library.
- Gradient Summand for Linear Regression = $(w^Tx - y)x$ where the gradient update equation is $w_{i+1} = w_{i} - \alpha_{i}\sum_{j}{(w_i^Tx_j - y_j)x_j}$
- Learning rate update rule : $\alpha_i = \frac{\alpha}{n * \sqrt{i+1}}$
- Linear Regression Gradient Update Rules
	- No Penalty - $w_{i+1} = w_{i} - \alpha_{i}(\sum_{j}{(w_i^Tx_j - y_j)x_j})$
	- L1 - $w_{i+1} = w_{i} - \alpha_{i}(\sum_{j}{(w_i^Tx_j - y_j)x_j}+ \eta sign(w_i))$
	- L2 -$w_{i+1} = w_{i} - \alpha_{i}(\sum_{j}{(w_i^Tx_j - y_j)x_j} + \eta w_i)$
- Linear Regression Update Rules = [Link](https://aunnnn.github.io/ml-tutorial/html/blog_content/linear_regression/linear_regression_regularized.html)
- Federated Learning = [Link](https://federated.withgoogle.com)
## HW 3
- [SparseVector](https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.mllib.linalg.SparseVector.html) Example
```
from pyspark import SparkContext
from pyspark.mllib.linalg import SparseVector
sc = SparkContext("local", "RDDExample")
x = SparseVector(5,[0,3],[2,4])
print(x)
print(x.toArray())

>>> Output
(5,[0,3],[2.0,4.0])
[2. 0. 0. 4. 0.]

y = [0,3,4,1,0]
y = SparseVector(5,[(ind,ele) for ind,ele in enumerate(y) if ele != 0])
print(y)
print(y.toArray())
>>> Output
(5,[1,2,3],[3.0,4.0,1.0])
[0. 3. 4. 1. 0.]
```
- **How to create OHE Feature for multivariate data?** One column each for (feature,feature_val) pair. For example we have 2 features Animal,Color and each had 2 vals Dog,Cat and Orange and Yellow. Then 4 columns for (Animal,Dog),(Animal,Cat),etc. 
- `Sometimes some features values not seen in training data can be seen in test data. For example in the above example maybe training data never had Animal,Tiger as a pair. To ensure this does not crash the code take advice from a subject matter expert or update the function to ignore unseens feature values.`
- [Logistic Regression Loss](https://developers.google.com/machine-learning/crash-course/logistic-regression/model-training) = $\sum_{(x,y) \in D}{ -ylog(y') - (1-y)log(1-y')}$
- [ROC Curve]
	The Receiver Operating Characteristic (ROC) curve is a graphical representation that illustrates the performance of a binary classification model across different classification thresholds. It is a widely used tool in machine learning and diagnostic medicine to evaluate the trade-off between sensitivity and specificity of a model.
	
	Here are the key concepts associated with the ROC curve:
	
	1. **Binary Classification:**
	   - The ROC curve is typically applied to binary classification problems where the outcome can be classified into two categories, often referred to as positive and negative classes.
	
	2. **Sensitivity and Specificity:**
	   - Sensitivity (True Positive Rate) is the proportion of true positives (correctly predicted positive instances) among all actual positive instances.
	   - Specificity (True Negative Rate) is the proportion of true negatives (correctly predicted negative instances) among all actual negative instances.
	
	3. **Threshold Variation:**
	   - In a binary classification model, the algorithm assigns a probability or score to each instance. A threshold is then applied to convert these scores into binary predictions (positive or negative). The ROC curve is created by varying this threshold.
	
	4. **True Positive Rate vs. False Positive Rate:**
	   - The ROC curve plots the True Positive Rate (sensitivity) on the y-axis against the False Positive Rate on the x-axis across different threshold values.
	   - False Positive Rate (FPR) is the proportion of actual negatives that are incorrectly predicted as positives.
	
	5. **AUC (Area Under the Curve):**
	   - The area under the ROC curve (AUC) is a scalar value that quantifies the overall performance of the classification model. A higher AUC indicates better discrimination between positive and negative classes.
	   - AUC ranges from 0 to 1, where 0.5 corresponds to a random classifier, and 1.0 indicates a perfect classifier.
	
	6. **Ideal Scenario:**
	   - In an ideal scenario, the ROC curve would hug the top-left corner of the plot, indicating high sensitivity and low false positive rate across all threshold values.
	
	Here's a brief summary of the terms used in ROC analysis:
	
	- **True Positive (TP):** Instances that are correctly predicted as positive.
	- **False Positive (FP):** Instances that are incorrectly predicted as positive.
	- **True Negative (TN):** Instances that are correctly predicted as negative.
	- **False Negative (FN):** Instances that are incorrectly predicted as negative.
- [Feature Hashing](https://alex.smola.org/papers/2009/Weinbergeretal09.pdf)
	- Define a hash function to take (feature,feature_val) pair and map it to one of m buckets. Essentially reducing n dimensions to m dimensions. (Used hashlib.md5 hashing)
	- Hashing Categorial Features - Concatenate string representations of feature_name/id and feature_val. Then use hashing algorithm to get an integer value and take modulus by number of buckets to get features.
	- Hashing Numeric Features - In this case we take string representation of feature_name/id and then use a hashing algorithm to get an integer value, followed by taking modulus by the number of buckets. Based on the bucket index we add the numeric value at that bucket index. `Note: It is recommended to normalize numeric values to avoid having a bias for numeric features which have bigger magnitudes.`
## HW 4
- [Sample Variance](https://www.youtube.com/watch?v=iHXdzfF7UEs)
	The variance is a statistical measure that quantifies the spread or dispersion of a set of data points. The formula for variance is given by:
	
	$\text{Var}(X) = \frac{1}{n} \sum_{i=1}^{n} (X_i - \bar{X})^2$
	
	Here, $X$ represents the individual data points in a dataset, $n$ is the number of data points, $\bar{X}$ is the mean (average) of the data points, and $\sum$ denotes the sum across all data points.
	
	The intuition behind the variance formula lies in understanding how each data point deviates from the mean. Here's a step-by-step breakdown:
	
	1. **Deviation from the Mean:** $X_i - \bar{X}$ represents the difference between each individual data point ($X_i$) and the mean ($\bar{X}$). This part measures how far each data point is from the average.
	
	2. **Square of Deviation:** Squaring the differences has two main effects:
	   - It ensures that all deviations are positive (eliminating the possibility of cancellation when summing deviations).
	   - It gives more weight to larger deviations, emphasizing the impact of data points that are further from the mean.
	
	3. **Summation and Average:** The sum of the squared deviations is divided by the number of data points ($n$). This step calculates the average squared deviation, providing a measure of the average "spread" or "dispersion" of the data.
	
	In summary, the variance captures how much individual data points deviate from the mean, and the squaring emphasizes larger deviations. By averaging these squared deviations, the variance provides a measure of the overall dispersion of the dataset. Keep in mind that the square root of the variance gives the standard deviation, which is often preferred as it is in the same units as the original data.
- [Covariance](https://www.youtube.com/watch?v=qtaqvPAeEJY)
	- Sensitive to the scale of the data, which makes the values difficult to interpret. Best to only understand the direction based on the sign of the values.
- [Sample Covariance](https://www.youtube.com/watch?v=SbOyFf-B5OQ)
- [Sample Covariance Matrix](https://www.youtube.com/watch?v=locZabK4Als)
- PCA Scores - These are the lower dimensional representation of the actual data.
- In general, it is advisable to normalize data before applying PCA, especially when the features have different units. This precaution is taken to prevent features with larger scales from unduly influencing the principal components, which could introduce bias into the results. Working with unnormalized data in this case, there is an increase in the percentage of explained variances, and the first component image is more meaningful compared to its normalized counterpart. In cases where our primary objective is to interpret and comprehend relationships between variables, the use of unnormalized data, which offers more meaningful images and higher explained variances, may be regarded as positive. On the contrary, if the aim is to achieve a balanced representation of features and mitigate biases stemming from scale differences, the decision not to normalize the data could be perceived as a potential drawback. In summary, whether this outcome is deemed advantageous or disadvantageous hinges on the specific goals of the analysis and the significance of normalization in attaining those objectives.
- [np.kron](https://numpy.org/doc/stable/reference/generated/numpy.kron.html) - [Kronecker Product](https://www.youtube.com/watch?v=e1UJXvu8VZk)
## HW 5
- Power Law
	The power law is a mathematical relationship between two quantities, where one quantity varies as a power of another. In other words, it describes a functional form where a change in one variable results in a proportional change in another variable raised to a fixed exponent. The power law is also known as a scaling law or a Pareto distribution.
	
	The general form of a power law relationship is given by:
	
	$Y = a \cdot X^b$
	
	Here:
	- $Y$ and $X$ are the two variables involved.
	- $a$ is a constant.
	- $b$ is the exponent, which determines the degree of the power law.
	
	The power law is often used to describe relationships in various fields, including physics, economics, biology, social sciences, and network theory. Some key characteristics of power laws include:
	
	1. **Scale Invariance:** Power laws exhibit scale invariance, meaning that the relationship holds regardless of the scale at which it is observed. This property makes power laws suitable for describing phenomena that occur at different magnitudes.
	
	2. **Heavy-Tailed Distribution:** Power laws are associated with heavy-tailed distributions, where a small number of events or entities have significantly larger magnitudes than the majority. This characteristic is often referred to as the "80-20 rule" or the "Pareto principle."
	
	3. **Common Examples:**
	   - In network theory, the degree distribution of many real-world networks (such as the internet or social networks) follows a power-law distribution.
	   - In economics, the distribution of wealth or income in a population is often modeled using a power law.
	
	4. **Robustness:** Systems described by power laws can be robust in the face of random changes but vulnerable to targeted attacks. This property is relevant in the study of complex systems and networks.
- Log Log Scale Plots
	Using a log-log scale to plot values is beneficial in certain situations, especially when dealing with data that spans several orders of magnitude. Here are some reasons why log-log scales are commonly used:
	
	1. **Visualization of Wide Ranges**: Log-log scales are useful when you have data that covers a wide range of values. In a linear scale, small values may be compressed at the bottom of the plot, making it difficult to discern patterns or trends in that range. A log-log scale can spread out the data, making it easier to visualize the entire range.
	
	2. **Power Law Relationships**: Many natural phenomena and complex systems exhibit power-law relationships, where one variable is proportional to a power of another variable. In such cases, plotting the data on a log-log scale transforms power-law relationships into straight lines. This simplifies analysis and makes it easier to identify the nature of the relationship between variables.
	
	3. **Highlighting Small Changes**: In a log-log plot, equal distances on the axes represent equal ratios, not equal differences. This means that small changes in values are more clearly visible, especially for values that are initially small. This can be important when examining fine details in the data.
	
	4. **Data Compression**: Logarithmic scales can compress large ranges of data into a more manageable size. This is particularly useful when presenting data in a limited space, such as in publications or presentations.
	
	5. **Sensitivity to Extreme Values**: Log-log plots can be more sensitive to extreme values or outliers. In linear scales, extreme values can dominate the plot and make it challenging to see patterns in the rest of the data. In log-log scales, extreme values have less visual impact.
	
	6. **Analysis of Exponential Growth or Decay**: When dealing with processes that involve exponential growth or decay, a log-log plot can help linearize the data, making it easier to analyze trends.
- Tree Ensemble Methods
	- Decision Tree
		Basic Concept:
		
		1. **Tree Structure:**
		   - Decision trees have a tree-like structure with nodes, branches, and leaves.
		   - Nodes represent decisions or test conditions.
		   - Branches represent the possible outcomes of a decision.
		   - Leaves represent the final decision or the predicted outcome.
		
		2. **Decision Making:**
		   - At each internal node, a decision is made based on a specific feature or attribute.
		   - The decision leads to one of the branches, and the process continues until a leaf node is reached.
		
		### Building a Decision Tree:
		
		1. **Root Node:**
		   - The first decision node at the top is called the root node.
		   - It represents the best feature to split the data based on certain criteria (e.g., Gini impurity, information gain).
		
		2. **Internal Nodes:**
		   - Subsequent nodes are internal nodes that represent decisions based on specific features.
		
		3. **Leaves:**
		   - Terminal nodes (leaves) represent the final output, which could be a class label (in classification) or a numerical value (in regression).
		
		### Splitting Criteria:
		
		1. **Gini Impurity:**
		   - Commonly used in classification problems.
		   - Measures the probability of incorrectly classifying a randomly chosen element in the dataset.
		
		2. **Information Gain:**
		   - Also used in classification problems.
		   - Measures the reduction in entropy or uncertainty after a dataset is split.
		
		3. **Mean Squared Error (MSE):**
		   - Used in regression problems.
		   - Measures the average squared difference between the predicted and actual values.
		
		### Advantages:
		
		1. **Interpretability:**
		   - Decision trees are easy to understand and interpret.
		   - They can be visualized graphically.
		
		2. **No Need for Feature Scaling:**
		   - Decision trees are not sensitive to the scale of the features.
		
		3. **Handles Non-linear Relationships:**
		   - Decision trees can capture non-linear relationships in the data.
		
		### Disadvantages:
		
		1. **Overfitting:**
		   - Decision trees are prone to overfitting, especially when the tree is deep.
		
		2. **Instability:**
		   - Small changes in the data can lead to a completely different tree.
		
		3. **Biased Toward Dominant Classes:**
		   - In classification problems with imbalanced class distribution, decision trees may be biased toward the dominant class.
	- Random Forest
		A Random Forest is an ensemble learning method that combines multiple decision trees to create a more robust and accurate model. It is used for both classification and regression tasks. The key idea behind a Random Forest is to build a multitude of decision trees and merge their predictions to obtain a more stable and accurate result. Here are the main features and concepts associated with Random Forests:
		### Key Features:
		
		1. **Ensemble Learning:**
		   - Random Forest is an ensemble method that builds multiple decision trees and combines their outputs.
		   - The ensemble approach helps to reduce overfitting and increase the overall model performance.
		
		2. **Bagging (Bootstrap Aggregating):**
		   - Each tree in the Random Forest is trained on a different subset of the training data.
		   - The subsets are created through bootstrapping, which involves randomly sampling with replacement from the original dataset.
		
		3. **Random Feature Selection:**
		   - At each node in a decision tree, a random subset of features is considered for splitting.
		   - This randomness helps to decorrelate the individual trees, making the overall model more robust.
		
		4. **Voting or Averaging:**
		   - For classification tasks, the final prediction is often determined by a majority vote among the trees.
		   - For regression tasks, the final prediction is the average of the predictions from individual trees.
		
		### Advantages:
		
		1. **Reduced Overfitting:**
		   - The use of multiple trees and random feature selection helps mitigate overfitting.
		
		2. **High Accuracy:**
		   - Random Forests often achieve high accuracy and generalization performance.
		
		3. **Robustness:**
		   - Random Forests are less sensitive to noise and outliers in the data due to the averaging effect of multiple trees.
		
		4. **Feature Importance:**
		   - Random Forests can provide a measure of feature importance, indicating which features contribute more to the model's predictions.
		
		### Disadvantages:
		
		1. **Complexity:**
		   - Random Forests are more complex than individual decision trees, making them harder to interpret.
		
		2. **Computational Resources:**
		   - Training and evaluating a large number of trees can be computationally expensive.
		
		### Applications:
		
		1. **Classification:**
		   - Commonly used for tasks such as spam detection, image classification, and medical diagnosis.
		
		2. **Regression:**
		   - Effective for predicting continuous variables, such as house prices or stock prices.
		
		3. **Feature Importance Analysis:**
		   - Used to identify the most influential features in a dataset.
	- Gradient Boosted Trees
		Gradient Boosting is another ensemble learning technique, like Random Forests, but it builds a series of weak learners, typically decision trees, sequentially. The key idea is to train each new tree to correct the errors made by the combination of existing trees. When specifically using decision trees as the weak learners, the algorithm is referred to as Gradient Boosted Trees or Gradient Boosting Machines (GBM). Here are the main concepts associated with Gradient Boosted Trees:
		### Key Concepts:
		
		1. **Sequential Learning:**
		   - Trees are added sequentially, with each new tree addressing the mistakes of the combined ensemble.
		
		2. **Residuals:**
		   - Each new tree is trained on the residuals (the differences between actual and predicted values) of the combined ensemble.
		
		3. **Weighted Voting:**
		   - The predictions of individual trees are weighted and summed to make the final prediction.
		   - Weights are determined during the training process based on the performance of each tree.
		
		4. **Shrinkage (Learning Rate):**
		   - A shrinkage parameter is used to control the contribution of each tree to the final prediction.
		   - Lower learning rates often lead to more robust models but require more trees to achieve the same level of accuracy.
		
		### Building the Model:
		
		1. **Initialize with a Constant:**
		   - The model starts with a simple prediction, usually the mean of the target variable.
		
		2. **Sequential Tree Building:**
		   - Build a decision tree to predict the residuals (errors) of the current model.
		
		3. **Update the Model:**
		   - Update the model by adding the new tree's predictions, scaled by a learning rate.
		
		4. **Repeat:**
		   - Repeat the process for a specified number of iterations or until a certain level of performance is achieved.
		
		### Advantages:
		
		1. **High Predictive Accuracy:**
		   - Gradient Boosted Trees often provide high predictive accuracy, comparable to Random Forests.
		
		2. **Handles Complex Relationships:**
		   - Can capture complex relationships in the data due to the sequential nature of building trees.
		
		3. **Feature Importance:**
		   - Like Random Forests, Gradient Boosted Trees can provide information about feature importance.
		
		### Disadvantages:
		
		1. **Computational Complexity:**
		   - Training can be computationally expensive, especially with a large number of trees.
		
		2. **Sensitive to Noisy Data:**
		   - Gradient Boosted Trees can be sensitive to noisy data and outliers.
		
		### Applications:
		
		1. **Regression and Classification:**
		   - Commonly used for a wide range of tasks, including predicting house prices, customer churn, and click-through rates.
		
		2. **Ranking:**
		   - Suitable for ranking tasks, such as search engine result ranking.
- Collaborative Filtering
	- https://www.youtube.com/watch?v=Fmtorg_dmM0
- ALS
	Alternating Least Squares (ALS) is an optimization algorithm commonly used in collaborative filtering techniques, particularly in the context of matrix factorization for recommendation systems. It is used to approximate a given matrix by decomposing it into two lower-rank matrices. This technique is often employed in recommendation systems to predict missing values in a user-item interaction matrix. Here are the key concepts related to Alternating Least Squares:
	
	### Matrix Factorization:
	
	1. **User-Item Interaction Matrix:**
	   - In the context of recommendation systems, the user-item interaction matrix represents user preferences for items (e.g., ratings or interactions).
	   
	2. **Matrix Factorization:**
	   - Matrix factorization is a technique used to approximate the original matrix by decomposing it into two or more lower-rank matrices.
	   
	3. **Low-Rank Approximation:**
	   - The goal is to find low-rank matrices that, when multiplied, approximate the original matrix with minimal error.
	
	### Alternating Least Squares:
	
	1. **Alternating Optimization:**
	   - ALS alternates between fixing one set of variables while optimizing the other set.
	   - In recommendation systems, it alternates between optimizing the user factors and item factors.
	
	2. **User Factors and Item Factors:**
	   - User factors and item factors are latent vectors that represent users and items in a lower-dimensional space.
	   - The user-item interaction matrix is approximated by the dot product of these latent vectors.
	
	3. **Least Squares Optimization:**
	   - ALS minimizes the least squares error between the observed ratings and the predicted ratings.
	   - It iteratively updates user factors and item factors to minimize the reconstruction error.
	
	### Steps in ALS:
	
	1. **Initialize User and Item Factors:**
	   - Start with random or predefined values for user and item factors.
	
	2. **Fix User Factors, Optimize Item Factors:**
	   - Treat user factors as constants and optimize item factors to minimize the least squares error.
	
	3. **Fix Item Factors, Optimize User Factors:**
	   - Treat item factors as constants and optimize user factors to minimize the least squares error.
	
	4. **Repeat:**
	   - Alternate between steps 2 and 3 for a fixed number of iterations or until convergence.
	
	### Advantages:
	
	1. **Parallelization:**
	   - ALS is inherently parallelizable, making it suitable for large-scale implementations.
	
	2. **Scalability:**
	   - ALS can efficiently handle sparse matrices, which is common in recommendation systems.
	
	### Limitations:
	
	1. **Cold Start Problem:**
	   - ALS may struggle with new users or items with limited interaction history.
	
	2. **Hyperparameter Sensitivity:**
	   - The performance of ALS can be sensitive to hyperparameters, such as the rank of the factor matrices.
- [Matrix Factorization](https://www.youtube.com/watch?v=ZspR5PZemcs)
- [Topic Modelling](https://www.youtube.com/watch?v=IUAHUEy1V0Q)
- [LDA](https://www.youtube.com/watch?v=T05t-SqKArY)
- [Word2Vec](https://www.youtube.com/watch?v=viZrOnJclY0)
- Pattern Mining
	Pattern mining refers to the process of discovering meaningful, interesting, and previously unknown patterns in large datasets. This technique is commonly used in data mining and machine learning to identify relationships, trends, or structures within data. There are various types of patterns that can be mined, including association rules, sequences, clusters, and anomalies.
	
	Here are a few key concepts related to pattern mining:
	
	1. **Association Rule Mining:**
	   - **Definition:** Association rule mining is focused on finding associations or relationships between variables in a dataset.
	   - **Example:** If customers who purchase product A also tend to purchase product B, an association rule might be discovered, such as "If A, then B."
	
	2. **Sequential Pattern Mining:**
	   - **Definition:** Sequential pattern mining is used when the order and timing of events are important. It identifies patterns in sequences of events.
	   - **Example:** Analyzing the sequence of web pages visited by users to understand the navigation patterns on a website.
	
	3. **Cluster Analysis:**
	   - **Definition:** Cluster analysis involves grouping similar items or data points together based on certain characteristics.
	   - **Example:** Grouping customers based on their purchasing behavior to identify different market segments.
	
	4. **Anomaly Detection:**
	   - **Definition:** Anomaly detection aims to identify patterns that do not conform to expected behavior or patterns.
	   - **Example:** Detecting unusual patterns in network traffic that may indicate a security threat.
	
	5. **Frequent Pattern Mining:**
	   - **Definition:** Frequent pattern mining identifies patterns that occur frequently in a dataset.
	   - **Example:** Discovering that a particular combination of items is frequently purchased together in a retail dataset.
- [FP-Growth Algorithm](https://www.youtube.com/watch?v=kK6yRznGTdo)
	The FP-Growth (Frequent Pattern Growth) algorithm is a popular data mining algorithm used for frequent itemset mining and association rule learning in datasets. It was proposed by Jiawei Han, Jian Pei, Yiwen Yin, and Runying Mao in their paper "Mining Frequent Patterns without Candidate Generation" in 2000.
	
	Here's a brief overview of how the FP-Growth algorithm works:
	
	1. **Database Scan (First Pass):**
	   - Scan the database to determine the frequency of each item (single items or 1-itemsets).
	   - Sort the items in non-increasing order of frequency.
	
	2. **Building the FP-Tree (Second Pass):**
	   - Scan the database again and for each transaction, reorder the items based on the frequency order obtained in the first pass.
	   - Build an FP-tree (Frequent Pattern tree) structure that represents the relationships and frequencies of itemsets in a compact manner.
	
	3. **Mining Frequent Itemsets (Recursive Third Pass):**
	   - Traverse the FP-tree to find frequent itemsets by considering different combinations of items.
	   - The algorithm uses a recursive approach to mine frequent itemsets efficiently.
	
	4. **Generating Association Rules:**
	   - Once the frequent itemsets are identified, association rules can be generated based on user-specified minimum support and confidence thresholds.
	
	One of the key advantages of FP-Growth is that it avoids the expensive step of candidate generation, which is a characteristic of other frequent itemset mining algorithms like Apriori. The FP-tree structure allows for efficient and compact representation of frequent patterns in the dataset.