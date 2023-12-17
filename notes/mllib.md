# MLLib

## What's in MLLib
As of my last knowledge update in January 2022, Apache Spark's MLlib library provided a variety of machine learning algorithms and optimization techniques. Keep in mind that library updates may have occurred since then, so it's a good idea to check the official documentation for the latest information. As of my last update, some key models and optimization algorithms in Spark MLlib included:

1. **Linear Models:**
   - Linear Regression
   - Logistic Regression
   - Support Vector Machines (SVM)

2. **Tree-based Models:**
   - Decision Trees
   - Random Forests
   - Gradient-Boosted Trees

3. **Clustering:**
   - K-means

4. **Collaborative Filtering:**
   - Alternating Least Squares (ALS) for collaborative filtering

5. **Dimensionality Reduction:**
   - Singular Value Decomposition (SVD)

6. **Optimization Algorithms:**
   - Stochastic Gradient Descent (SGD)
   - L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) for logistic regression

7. **Ensemble Methods:**
   - Bagging
   - Boosting

8. **Cross-Validation:**
   - k-fold Cross-Validation

9. **Pipelines:**
   - Pipeline API for constructing ML workflows

10. **Feature Extraction and Transformation:**
    - TF-IDF
    - Word2Vec

## Benefits of MLLib
-  Many algorithms and utilities 
- Performance gains ¡ Scalability
- Part of Spark
	- Integrated with Spark and its other components
	- Various languages (Python, Scala, Java, R)

## Example using MLLib and SparkSQL

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create a Spark session
spark = SparkSession.builder.appName("MLlibExample").getOrCreate()

# Generate a synthetic dataset
data = [(0, 2.0, 1.0),
        (1, 1.0, 0.0),
        (0, 2.0, 1.0),
        (1, 1.0, 1.0),
        (1, 1.0, 0.0),
        (0, 2.0, 1.0)]

columns = ["label", "feature1", "feature2"]
df = spark.createDataFrame(data, columns)

# Register the DataFrame as a temporary SQL table
df.createOrReplaceTempView("my_table")

# Define a SQL query to select the features and labels
sql_query = "SELECT label, feature1, feature2 FROM my_table"

# Use Spark SQL to create a new DataFrame with the selected columns
selected_data = spark.sql(sql_query)

# Define the feature assembler
feature_cols = ['feature1', 'feature2']
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

# Define the logistic regression model
lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)

# Create the pipeline
pipeline = Pipeline(stages=[assembler, lr])

# Split the data into training and testing sets
train_data, test_data = selected_data.randomSplit([0.8, 0.2], seed=123)

# Fit the pipeline to the training data
model = pipeline.fit(train_data)

# Make predictions on the testing data
predictions = model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='label', metricName='areaUnderROC')
auc = evaluator.evaluate(predictions)
print(f'Area under ROC curve: {auc}')

# Stop the Spark session
spark.stop()
```


## ML Pipeline in Spark
Using the `pyspark.ml` library, you can create machine learning pipelines in Apache Spark. A machine learning pipeline consists of a sequence of stages, where each stage is either a Transformer or an Estimator. A Transformer is an abstraction that includes feature transformers and learned models, while an Estimator is an algorithm or any algorithm that can be fit on data. Here's a basic example of creating a machine learning pipeline in PySpark:

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create a Spark session
spark = SparkSession.builder.appName("example").getOrCreate()

# Assume you have a DataFrame named 'data' with columns 'feature1', 'feature2', and 'label'
# 'label' is the target variable, and 'feature1', 'feature2' are input features

# Define the feature assembler
feature_cols = ['feature1', 'feature2']
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

# Define the logistic regression model
lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)

# Create the pipeline
pipeline = Pipeline(stages=[assembler, lr])

# Split the data into training and testing sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=123)

# Fit the pipeline to the training data
model = pipeline.fit(train_data)

# Make predictions on the testing data
predictions = model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='label', metricName='areaUnderROC')
auc = evaluator.evaluate(predictions)
print(f'Area under ROC curve: {auc}')
```