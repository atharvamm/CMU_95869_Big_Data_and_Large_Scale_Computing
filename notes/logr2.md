# Logistic Regression 2
- Classification Errors
	In classification problems, errors can occur when the model makes predictions. The two main types of errors are false positives and false negatives, and they are related to the concepts of false positive rate (FPR) and true positive rate (TPR), also known as sensitivity or recall. Let's define each of these terms:
	
	1. **False Positive (FP):**
	   - A false positive occurs when the model predicts the positive class, but the true class is negative. In other words, the model incorrectly classifies an instance as positive when it is actually negative.
	
	2. **False Negative (FN):**
	   - A false negative occurs when the model predicts the negative class, but the true class is positive. In other words, the model incorrectly classifies an instance as negative when it is actually positive.
	
	3. **False Positive Rate (FPR):**
	   - The false positive rate is the ratio of false positives to the total number of actual negatives. It is calculated as:
	     $FPR = \frac{FP}{FP + TN}$
	     where $TN$ is the number of true negatives.
	
	4. **True Positive Rate (TPR):**
	   - The true positive rate, also known as sensitivity or recall, is the ratio of true positives to the total number of actual positives. It is calculated as:
	     $TPR = \frac{TP}{TP + FN}$
	     where $TP$ is the number of true positives.
	
	These metrics are often used to evaluate the performance of a classification model. In addition to FPR and TPR, other common metrics include:
	
	- **Precision:**
	  $Precision = \frac{TP}{TP + FP}$
	  Precision measures the accuracy of the positive predictions made by the model.
	
	- **Specificity:**
	  $Specificity = \frac{TN}{TN + FP}$
	  Specificity measures the accuracy of the negative predictions made by the model.
	
	- **Accuracy:**
	  $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
	  Accuracy is the overall correctness of the model's predictions.
	
	- **F1 Score:**
	  $F1 \text{ Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
	  The F1 score is the harmonic mean of precision and recall and is useful when there is an imbalance between the classes.
	
	These metrics provide a comprehensive view of a classification model's performance by considering different aspects of its predictions. The choice of which metric to prioritize depends on the specific goals and requirements of the application.
- ROC and AUC Curve

- Logistic Loss
	The logistic loss, also known as log loss or cross-entropy loss, is a commonly used loss function in logistic regression and other binary classification models. It measures the performance of a classification model whose output is a probability value between 0 and 1. The goal is to minimize the logistic loss, which represents the dissimilarity between the predicted probabilities and the actual class labels.
	
	For a binary classification problem with true labels $y_i$ (0 or 1) and predicted probabilities $p_i$, the logistic loss is defined as follows:
	
	$\text{Logistic Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$
	
	Here:
	- $N$ is the number of samples.
	- $y_i$ is the true label for the $i$-th sample.
	- $p_i$ is the predicted probability that the $i$-th sample belongs to the positive class.
	
	The logistic loss penalizes the model more when it confidently predicts the wrong class. If the predicted probability is close to the true label, the loss is lower. However, as the predicted probability deviates from the true label, the loss increases.
	
	The logistic loss is commonly used as the objective function to be minimized during the training of logistic regression models and other binary classifiers. The optimization process involves adjusting the model parameters (coefficients) to minimize the logistic loss over the training data.
	
	In mathematical terms, for a logistic regression model with parameters $\beta$, the logistic loss can be expressed as:
	
	$\text{Logistic Loss}(\beta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log\left(\frac{1}{1 + e^{-\beta^T x_i}}\right) + (1 - y_i) \log\left(1 - \frac{1}{1 + e^{-\beta^T x_i}}\right) \right]$
	
	Here:
	- $x_i$ is the feature vector for the $i$-th sample.
	- $e$ is the base of the natural logarithm.
	
	Minimizing the logistic loss is often performed using optimization algorithms such as gradient descent, where the gradients of the loss with respect to the model parameters are used to iteratively update the parameters until convergence.
- Types of Features
	In statistical modeling and machine learning, features (also known as variables or attributes) are categorized into different types based on the nature of the data they represent. Three common types of features are numeric, categorical, and ordinal.
	
	1. **Numeric Features:**
	   - Numeric features represent numerical values and can be continuous or discrete. Examples include age, height, income, temperature, etc. Numeric features are often used in regression models or algorithms that require mathematical operations.
	
	2. **Categorical Features:**
	   - Categorical features represent categories or labels and do not have a meaningful numerical interpretation. They can be further divided into two subtypes: nominal and ordinal.
	     - **Nominal Features:** Nominal features represent categories with no inherent order or ranking. Examples include colors, types of fruits, or country names.
	     - **Ordinal Features:** Ordinal features, on the other hand, represent categories with a meaningful order or ranking. Examples include education levels (e.g., high school, bachelor's, master's) or customer satisfaction ratings (e.g., low, medium, high).
	
	3. **Ordinal Features:**
	   - Ordinal features are a specific type of categorical feature where the categories have a meaningful order or ranking. Unlike nominal features, the order of categories matters. However, the intervals between the categories may not be uniform or well-defined.
	
	Here's a summary:
	
	- **Numeric Features:** Represent numerical values and can be continuous or discrete.
	- **Categorical Features:** Represent categories and can be nominal (no order) or ordinal (with order).
	- **Ordinal Features:** A subtype of categorical features where the categories have a meaningful order.
	
	In a dataset, you may encounter a mix of these feature types. It's important to handle each type appropriately during data preprocessing and model development. For instance:
	
	- **Numeric Features:** May require scaling if they have different ranges or units.
	- **Categorical Features:** May need encoding techniques (e.g., one-hot encoding for nominal features) for machine learning algorithms that require numeric input.
	- **Ordinal Features:** May be treated as categorical but might benefit from encoding that preserves the order.
	
	Understanding the nature of your features is crucial for selecting appropriate models, preprocessing steps, and interpreting the results of your analyses or predictions.
- Encoding Non-Numeric to Numeric Features
	Converting non-numeric features to numeric form is a common preprocessing step in machine learning. Different types of non-numeric features (categorical, ordinal, or textual) may require different encoding methods. Here are several common approaches:
	
	1. **Label Encoding:**
	   - For ordinal features (categories with a meaningful order), label encoding assigns a unique integer to each category based on their order. Libraries like scikit-learn in Python provide `LabelEncoder` for this purpose.
	
	    ```python
	    from sklearn.preprocessing import LabelEncoder
	
	    le = LabelEncoder()
	    encoded_labels = le.fit_transform(['low', 'medium', 'high'])
	    ```
	
	   - The challenge with label encoding is that it may imply an ordinal relationship between the categories, which might not always be appropriate.
	
	2. **One-Hot Encoding:**
	   - For nominal features (categories with no inherent order), one-hot encoding creates binary columns for each category. Each column represents the presence or absence of a particular category. Pandas provides a `get_dummies` function for this purpose.
	
	    ```python
	    import pandas as pd
	
	    df = pd.DataFrame({'Color': ['Red', 'Green', 'Blue']})
	    one_hot_encoded = pd.get_dummies(df['Color'])
	    ```
	
	3. **Ordinal Encoding:**
	   - If the ordinal feature has a specific order, you can create a mapping dictionary and replace the values accordingly.
	
	    ```python
	    ordinal_mapping = {'low': 1, 'medium': 2, 'high': 3}
	    df['OrdinalColumn'] = df['OrdinalColumn'].map(ordinal_mapping)
	    ```
	
	4. **Binary Encoding:**
	   - For categorical features with high cardinality (many unique values), binary encoding can be used. It represents each unique category with binary code.
	
	    ```python
	    import category_encoders as ce
	
	    encoder = ce.BinaryEncoder(cols=['CategoryColumn'])
	    df_binary = encoder.fit_transform(df)
	    ```
	
	5. **Embedding for Text Data:**
	   - If dealing with textual data, natural language processing (NLP) techniques like word embeddings (e.g., Word2Vec, GloVe) can be used to convert text into numeric vectors.
	
	    ```python
	    # Using Word2Vec as an example
	    from gensim.models import Word2Vec
	
	    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
	    word_vectors = model.wv
	    ```
	
	Choose the appropriate encoding method based on the nature of your data and the requirements of your machine learning model. Always keep in mind the assumptions and implications of the encoding technique you choose, as well as the potential impact on the model's performance.
- Sparse Representations
	Let's break down the example and understand the storage requirements for dense and sparse representations of a matrix with 10 million observations and 1,000 features, assuming only 1% of the entries are non-zeros.
	
	### Dense Representation:
	1. **Number of Entries:**
	   - $10,000,000 \times 1,000 = 10,000,000,000$ entries.
	
	2. **Storage Requirements:**
	   - Storing each entry as a double (8 bytes): $10,000,000,000 \times 8 = 80,000,000,000$ bytes.
	   - In terms of gigabytes (GB): $80,000,000,000 \, \text{bytes} \, / \, (1024 \times 1024 \times 1024) \approx 74.51 \, \text{GB}$.
	
	### Sparse Representation:
	1. **Number of Non-Zero Entries (Assuming 1%):**
	   - $10,000,000,000 \times 0.01 = 100,000,000$ non-zero entries.
	
	2. **Storage Requirements for Non-Zero Entries:**
	   - Storing value and location (2 doubles per entry): $100,000,000 \times 2 \times 8 = 1,600,000,000$ bytes.
	   - In terms of gigabytes (GB): $1,600,000,000 \, \text{bytes} \, / \, (1024 \times 1024 \times 1024) \approx 1.49 \, \text{GB}$.
	
	### Comparison:
	- **Dense Representation Storage:** Approximately 74.51 GB.
	- **Sparse Representation Storage:** Approximately 1.49 GB.
	- **Storage Savings:** $\approx$ 50x.
	
	### Computational Savings:
	- In addition to storage savings, sparse representations can lead to computational savings in matrix operations because operations involving zeros need not be computed.
	
	- For example, in a dense matrix multiplication, you would perform $10,000,000 \times 1,000 \times 1,000 \times 10,000,000$ operations, while in a sparse matrix multiplication, you would only need to perform operations for the non-zero entries.
	
	Sparse representations are particularly beneficial when dealing with large datasets with many zeros, as they allow for more efficient storage and computation. Sparse matrix formats, such as Compressed Sparse Row (CSR) or Compressed Sparse Column (CSC), are commonly used to efficiently represent sparse matrices in practice.
- OHE Cons
	One-Hot Encoding (OHE) is a popular technique for representing categorical variables in a binary format, but it comes with some drawbacks:
	
	1. **High Dimensionality:**
	   - OHE can significantly increase the dimensionality of the dataset, especially when dealing with categorical features with a large number of unique values. Each unique category becomes its own binary column, leading to a sparse and high-dimensional feature space. This can lead to the curse of dimensionality, making the dataset more challenging to manage and analyze.
	
	2. **Collinearity:**
	   - The binary columns created by OHE are often highly correlated because if all other binary columns are 0, the last one must be 1 (and vice versa). This can introduce multicollinearity in the dataset, which might adversely affect certain machine learning models that assume independent features.
	
	3. **Increased Storage Requirements:**
	   - OHE increases the storage requirements for the dataset, especially when dealing with large datasets. This is because each binary column requires additional storage compared to a single categorical column.
	
	4. **Loss of Information about Magnitude:**
	   - OHE discards information about the magnitude or order of categories within a feature. The resulting binary representation treats all categories as equal, which might not be suitable for ordinal categorical variables.
	
	5. **Difficulty Handling New Categories:**
	   - OHE assumes that the set of categories in the test set or future data is the same as in the training set. If new categories appear in the test set, which were not present in the training set, handling them can be challenging. They would need to be encoded as zeros for the existing columns, leading to a potential loss of information.
	
	6. **Sparse Matrix Representation:**
	   - The binary representation generated by OHE results in a sparse matrix, where most of the entries are zero. While this is memory-efficient, it may not be optimal for all machine learning algorithms, especially those that don't take advantage of sparse data structures.
	
	7. **Curse of Dimensionality:**
	   - The high dimensionality introduced by OHE can exacerbate the curse of dimensionality, making certain algorithms, such as k-nearest neighbors (k-NN), computationally expensive and prone to overfitting.
- Reduce OHE Dimension
	When dealing with One-Hot Encoding (OHE) and facing issues related to high dimensionality, two common approaches to reduce dimensions are:
	
	### 1. Discarding Rare Features:
	
	**Process:**
	- Identify and discard features that have very low occurrence or frequency in the dataset.
	- This involves setting a threshold below which features are considered rare and then removing them.
	
	**Pros:**
	- Reduces dimensionality, making the dataset more manageable.
	- Eliminates features with little predictive power or contribution.
	
	**Cons:**
	- Loss of information if rare features turn out to be important.
	- The threshold selection might be subjective and require careful consideration.
	
	### 2. Feature Hashing:
	
	**Process:**
	- Apply feature hashing (also known as the hash trick) to convert categorical features into a fixed-size vector representation.
	- The hashing function maps the original feature values into a fixed number of buckets, effectively reducing dimensionality.
	
	**Pros:**
	- Efficient and memory-friendly, especially when dealing with high-cardinality categorical features.
	- Can handle new or unseen categories in the test set without issues.
	
	**Cons:**
	- Loss of interpretability, as hashed features do not retain the original category information.
	- Potential for hash collisions, where different original features map to the same hash value.
	
	**Example using Python and scikit-learn:**
	
	```python
	from sklearn.feature_extraction import FeatureHasher
	import pandas as pd
	
	# Example DataFrame with a categorical feature
	data = {'Category': ['A', 'B', 'C', 'A', 'B']}
	df = pd.DataFrame(data)
	
	# Feature hashing with 3 hash bits (resulting in 2^3 = 8 features)
	hasher = FeatureHasher(n_features=8, input_type='string')
	hashed_features = hasher.fit_transform(df['Category']).toarray()
	
	# Display hashed features
	print(hashed_features)
	```
	
	In this example, `n_features` is the desired number of output features, and `input_type='string'` indicates that the input is categorical.
	
	### Considerations:
	- The choice between discarding rare features and feature hashing depends on the specific characteristics of the dataset and the modeling goals.
	- Feature hashing is particularly useful when dealing with high-cardinality categorical features, while discarding rare features might be more appropriate when interpretability is crucial.
- Feature Hashing
	Feature hashing, also known as the hash trick or the hashing trick, is a technique used to convert categorical features into a fixed-size vector representation, thereby reducing the dimensionality of the feature space. This technique is particularly reasonable in scenarios where you have high-cardinality categorical features and want to mitigate the issues associated with high dimensionality.
	
	### Reasons Feature Hashing is Reasonable:
	
	1. **Memory Efficiency:**
	   - Feature hashing is memory-efficient, especially when dealing with datasets containing a large number of unique categorical values. Instead of creating a one-hot encoded representation, which can be sparse and memory-intensive, feature hashing maps categories to a fixed number of hash buckets, resulting in a more compact representation.
	
	2. **Scalability:**
	   - Feature hashing is scalable and can handle large datasets. The computation involved in feature hashing can be distributed across multiple computing nodes, making it suitable for big data or distributed computing environments.
	
	3. **Handling New Categories:**
	   - Feature hashing can naturally handle new or unseen categories in the test set without requiring retraining or pre-specification of all possible categories. Each category is hashed independently of the others.
	
	4. **Dimensionality Reduction:**
	   - The technique effectively reduces the dimensionality of the feature space, which is advantageous in scenarios where dimensionality is a concern, such as when using memory-constrained environments or when dealing with algorithms sensitive to high-dimensional spaces.
	
	### Performing Feature Hashing in a Distributed Computing Environment:
	
	When working in a distributed computing environment, you can perform feature hashing by leveraging frameworks or libraries that support distributed processing. Some considerations include:
	
	1. **Use Distributed Libraries:**
	   - Utilize distributed computing libraries or frameworks, such as Apache Spark, Dask, or Hadoop, to parallelize the feature hashing process across multiple nodes. These frameworks are designed to handle large-scale distributed data processing.
	
	2. **Parallelization Across Nodes:**
	   - Distribute the data across nodes in your computing cluster and perform feature hashing independently on each node. Ensure that the hashing function is consistent across nodes to maintain the integrity of the hashing process.
	
	3. **Combine Hashed Features:**
	   - After hashing is performed on individual nodes, combine or aggregate the hashed features to obtain the final feature representation. This may involve additional processing steps depending on the specific requirements of your analysis or machine learning model.
	
	4. **Consider Hash Function Consistency:**
	   - Ensure that the hash function used for feature hashing is consistent across distributed nodes. Consistency is crucial to ensure that the same category is hashed to the same bucket across different nodes, preserving the integrity of the feature representation.
	
	Here's a simplified example using PySpark (Apache Spark's Python API):
	
	```python
	from pyspark.ml.feature import FeatureHasher
	from pyspark.sql import SparkSession
	
	# Initialize a Spark session
	spark = SparkSession.builder.master("local").appName("FeatureHashingExample").getOrCreate()
	
	# Example DataFrame with a categorical feature
	data = [('A',), ('B',), ('C',), ('A',), ('B',)]
	columns = ['Category']
	df = spark.createDataFrame(data, columns)
	
	# Feature hashing with 3 hash bits (resulting in 2^3 = 8 features)
	hasher = FeatureHasher(inputCols=['Category'], outputCol='hashed_features', numFeatures=8)
	hashed_df = hasher.transform(df)
	
	# Display the resulting DataFrame with hashed features
	hashed_df.show(truncate=False)
	```
	
	In a distributed environment, you would scale this process across a cluster of nodes, and Spark would handle the parallelization of the feature hashing task.
	
	Keep in mind that while feature hashing provides advantages in terms of memory efficiency and scalability, it comes with the trade-off of potential hash collisions, where different categories map to the same hash bucket. The choice of the number of hash buckets (`numFeatures`) influences the trade-off between collision risk and dimensionality reduction.
- [Feature Hashing Example](https://github.com/atharvamm/CMU_95869_Big_Data_and_Large_Scale_Computing/blob/main/code/hw3/amhaskar_hw3_ctr_student-2023.ipynb)
```python
from pyspark import SparkContext
from collections import defaultdict
import hashlib

  

sc = SparkContext("local", "RDDExample")

  
def hashFunction(numBuckets, rawFeats, printMapping=False):
	mapping = {}
	for ind, category in rawFeats:
		featureString = category + str(ind)
		mapping[featureString] = int(int(hashlib.md5(featureString.encode('utf-8')).hexdigest(), 16) % numBuckets)
	if(printMapping): print ("Mapping: ",mapping)
	sparseFeatures = defaultdict(float)
	for bucket in mapping.values():
			sparseFeatures[bucket] += 1.0
	return dict(sparseFeatures)

def hashNumericFunction(numBuckets, rawFeats, printMapping=False):
	mapping = {}
	for ind,_ in enumerate(rawFeats):
		featureString = str(ind)
		hashedValue = int(int(hashlib.md5(featureString.encode('utf-8')).hexdigest(), 16) % numBuckets)
		mapping[featureString] = hashedValue
	if printMapping:
		print("Mapping: ",mapping)
	sparseFeatures = defaultdict(float)
	
	for ind, bucket in mapping.items():
		numericValue = rawFeats[int(ind)]
		sparseFeatures[bucket] += numericValue
	return dict(sparseFeatures)

  
  
sampleThree = [(0, 'bear'), (1, 'black'), (2, 'salmon'),(3, 'samon')]
sampleThreeFiveBuckets = hashFunction(5, sampleThree, True)
print(sampleThreeFiveBuckets)

sampleNumericOne = [8,10]
sampOneFiveBuckets = hashNumericFunction(5, sampleNumericOne, True)
print(sampOneFiveBuckets)

sc.stop()

>>> Output
>>> Mapping:  {'bear0': 2, 'black1': 4, 'salmon2': 0, 'samon3': 2}
>>> {2: 2.0, 4: 1.0, 0: 1.0}
>>> Mapping:  {'0': 0, '1': 1}
>>> {0: 8.0, 1: 10.0}
```
- 