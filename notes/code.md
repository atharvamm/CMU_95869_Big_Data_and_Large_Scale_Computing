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
- Gaussian Mixture Model
- Sample Covariance
- Sample Variance
- Sample Covariance Matrix
- Eigen Decomposition
- PCA Scores
- PCA on GMM
- PCA Explained Variance
- In general, it is advisable to normalize data before applying PCA, especially when the features have different units. This precaution is taken to prevent features with larger scales from unduly influencing the principal components, which could introduce bias into the results. Working with unnormalized data in this case, there is an increase in the percentage of explained variances, and the first component image is more meaningful compared to its normalized counterpart. In cases where our primary objective is to interpret and comprehend relationships between variables, the use of unnormalized data, which offers more meaningful images and higher explained variances, may be regarded as positive. On the contrary, if the aim is to achieve a balanced representation of features and mitigate biases stemming from scale differences, the decision not to normalize the data could be perceived as a potential drawback. In summary, whether this outcome is deemed advantageous or disadvantageous hinges on the specific goals of the analysis and the significance of normalization in attaining those objectives.
- [np.kron](https://numpy.org/doc/stable/reference/generated/numpy.kron.html) - [Kronecker Product](https://www.youtube.com/watch?v=e1UJXvu8VZk)
## HW 5
- Power Law
- Log Log Scale Plots
- Tree Ensemble Methods
	- Decision Tree
	- Random Forest
	- Gradient Boosted Trees
- Collaborative Filtering
- ALS
- Matrix Factorization
- Topic Modelling
- LDA
- Word2Vec
- Frequent Pattern Mining
- Pattern Mining
- FP-Growth Algorithm
- 