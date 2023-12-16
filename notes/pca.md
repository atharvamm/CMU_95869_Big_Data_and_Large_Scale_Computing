
# Principal Component Analysis
## Raw Data
- Complex and high dimensional
- Measure redundant signals
- Represent data via the method by which it was gathered

## PCA Goal 
- Minimize reconstruction error
- Maximize Variance between points

## Linear Regression vs PCA
1. **Objective:**
   - **Linear Regression:** It is a supervised learning algorithm used for predicting a dependent variable based on one or more independent variables. The goal is to find the best-fitting linear relationship between the input features and the target variable.
   - **PCA:** It is an unsupervised technique used for dimensionality reduction. The objective is to transform the original features into a new set of uncorrelated variables (principal components) that capture the maximum variance in the data.

2. **Use Case:**
   - **Linear Regression:** It is commonly used for tasks such as predicting house prices, stock prices, or any other continuous numeric value.
   - **PCA:** It is often used when dealing with high-dimensional data to reduce the number of features while retaining as much information as possible.

3. **Output:**
   - **Linear Regression:** The output is a regression equation that can be used for making predictions.
   - **PCA:** The output is a set of principal components, which are linear combinations of the original features.

4. **Supervised vs. Unsupervised:**
   - **Linear Regression:** It is a supervised learning technique because it requires a labeled dataset with a target variable for training.
   - **PCA:** It is an unsupervised technique as it does not require a target variable. It focuses on the variance-capturing structure of the data.

5. **Goal of Analysis:**
   - **Linear Regression:** It aims to understand and quantify the relationship between the input variables and the target variable.
   - **PCA:** It aims to capture the most important information or structure in the data by finding the directions (principal components) along which the data varies the most.

6. **Application in Feature Selection:**
   - **Linear Regression:** It may involve feature selection based on the importance of features for predicting the target variable.
   - **PCA:** It directly transforms the features into a new space, potentially reducing the need for explicit feature selection.

## PCA 
Principal Component Analysis (PCA) is a dimensionality reduction technique widely used in statistics, machine learning, and data analysis. Its primary goal is to transform high-dimensional data into a lower-dimensional representation, capturing as much of the original variance as possible. PCA achieves this by identifying the principal components, which are linear combinations of the original features.

Here's a step-by-step overview of how PCA works:

1. **Standardization:**
   - Standardize the dataset by subtracting the mean from each feature and dividing by the standard deviation. This step is crucial to give all features equal weight in the analysis.

2. **Covariance Matrix Calculation:**
   - Compute the covariance matrix of the standardized data. The covariance matrix represents the relationships between different features and how they vary together.

3. **Eigenvalue and Eigenvector Computation:**
   - Calculate the eigenvalues and eigenvectors of the covariance matrix. The eigenvectors represent the directions of maximum variance in the data, and the corresponding eigenvalues indicate the magnitude of variance along those directions.

4. **Sorting Eigenvalues:**
   - Sort the eigenvalues in descending order. The higher the eigenvalue, the more variance is explained by the corresponding eigenvector.

5. **Selecting Principal Components:**
   - Choose the top k eigenvectors based on the desired dimensionality reduction (where k is the number of principal components to retain). These eigenvectors form the new basis for the data.

6. **Projection:**
   - Project the original data onto the selected principal components to obtain the lower-dimensional representation.

The resulting lower-dimensional representation retains as much variance as possible from the original data, making it useful for visualization, noise reduction, and speeding up subsequent machine learning algorithms.

PCA is not only used for dimensionality reduction but also for other applications:

- **Visualization:** The reduced-dimensional data can be easily visualized, especially when dealing with data in three or more dimensions.
- **Data Compression:** PCA can be used for compressing data while preserving its essential features.
- **Noise Reduction:** By focusing on the principal components associated with high eigenvalues, PCA helps filter out noise in the data.
- **Feature Engineering:** PCA can be used as a feature engineering technique to transform the original features into a set of uncorrelated features.

## Calculate PCA
Principal Component Analysis (PCA) involves several steps for calculating the principal components of a dataset. Here's a step-by-step guide to performing PCA:

Let's assume we have a dataset with $m$ samples and $n$ features, represented by the matrix $X$ (size $m \times n$).

### Step 1: Standardize the Data
Subtract the mean of each feature from the data and divide by the standard deviation. This ensures that all features are on the same scale.

$Z = \frac{(X - \mu)}{\sigma}$

where $\mu$ is the mean vector and $\sigma$ is the standard deviation vector.

### Step 2: Compute the Covariance Matrix
Calculate the covariance matrix $C$ for the standardized data $Z$.

$C = \frac{1}{m-1} \cdot Z^T \cdot Z$

where $T$ denotes the transpose.

### Step 3: Compute Eigenvalues and Eigenvectors
Solve the characteristic equation to find the eigenvalues ($\lambda$) and eigenvectors ($v$) of the covariance matrix $C$.  $C \cdot v = \lambda \cdot v$

This is typically done using numerical methods or linear algebra libraries.

### Step 4: Sort Eigenvalues
Arrange the eigenvalues in descending order, and correspondingly, arrange the eigenvectors. The eigenvalues represent the amount of variance captured by each principal component.

### Step 5: Select Principal Components
Choose the top $k$ eigenvectors based on the desired dimensionality reduction, where $k$ is the number of principal components to retain.

### Step 6: Project the Data onto Principal Components
Multiply the standardized data $Z$ by the matrix of selected eigenvectors to obtain the lower-dimensional representation. $Y = Z \cdot V_k$ where $V_k$ is the matrix of the top $k$ eigenvectors.

The resulting matrix $Y$ represents the dataset in the new coordinate system defined by the principal components.

### Note:
- You can also calculate the explained variance for each principal component, which is the proportion of the total variance explained by that component.
- The percentage of total variance explained by the first $k$ principal components can be calculated as the sum of the $k$ largest eigenvalues divided by the sum of all eigenvalues.

## PCA SVD
### PCA using SVD:

1. **Standardize the Data:**
   - Begin with a dataset $X$ of size $m \times n$, where $m$ is the number of samples and $n$ is the number of features. Standardize the data by subtracting the mean and dividing by the standard deviation for each feature.

2. **Compute the Covariance Matrix:**
   - Compute the covariance matrix $C$ of the standardized data.

3. **Singular Value Decomposition (SVD):**
   - Perform SVD on the covariance matrix $C$:
   
   $C = U \Sigma V^T$

   where:
   - $U$ is a $m \times m$ orthogonal matrix.
   - $\Sigma$ is a diagonal matrix containing the singular values ($\sigma_1, \sigma_2, \ldots, \sigma_n$).
   - $V$ is a $n \times n$ orthogonal matrix.

4. **Select Principal Components:**
   - The columns of $V$ are the principal components. Choose the top $k$ columns to retain the most important components, where $k$ is the desired dimensionality reduction.

5. **Project Data onto Principal Components:**
   - The lower-dimensional representation of the data ($Y$) is obtained by multiplying the standardized data matrix $Z$ by the selected principal components:

   $Y = Z \cdot V_k$

   where $V_k$ is the matrix containing the first $k$ columns of $V$.

### Notes:
- The singular values ($\sigma_i$) represent the square roots of the eigenvalues of the covariance matrix.
- The columns of $V$ are the right singular vectors and correspond to the principal components of the data.
- The product $Z \cdot V_k$ is equivalent to the matrix multiplication used in PCA.

## PCA Assumptions
Principal Component Analysis (PCA) is a versatile and widely used technique, but like any statistical method, it makes certain assumptions to be valid and effective. Here are the key assumptions associated with PCA:

1. **Linearity:**
   - PCA assumes that the relationships among variables are linear. The method is most effective when the relationships between variables can be well approximated by linear transformations. Non-linear relationships may not be adequately captured by PCA.

2. **Standardization:**
   - The variables should be standardized before applying PCA. Standardization involves scaling each variable to have a mean of 0 and a standard deviation of 1. This is crucial because PCA is sensitive to the scale of the variables. Failure to standardize may result in variables with larger scales dominating the principal components.

3. **Orthogonality:**
   - PCA assumes that the principal components are orthogonal (uncorrelated) to each other. This means that the new variables created by PCA are independent linear combinations of the original variables. This assumption simplifies the interpretation of the principal components.

4. **Normality (for statistical inference):**
   - If PCA is used for statistical inference, such as hypothesis testing or confidence intervals, the assumption of multivariate normality may be relevant. This assumption is not strictly required for dimensionality reduction purposes, but it becomes important in certain statistical analyses based on PCA results.

5. **Homoscedasticity:**
   - Homoscedasticity assumes that the variance of the data is constant across all levels of the independent variables. While PCA itself does not explicitly assume homoscedasticity, it is essential if the principal components are later used in regression or other statistical models.

6. **Large Sample Size (for robustness):**
   - PCA tends to work well with larger sample sizes. Asymptotic results suggest that with a sufficiently large sample size, the sample covariance matrix converges to the population covariance matrix. In practice, a common rule of thumb is to have more observations than variables for stable results.

## PCA Cons
While Principal Component Analysis (PCA) is a powerful and widely used technique, it does have some limitations and considerations that should be taken into account:

1. **Linearity Assumption:**
   - PCA assumes linear relationships between variables. If the relationships are non-linear, PCA may not effectively capture the underlying structure of the data.

2. **Sensitivity to Outliers:**
   - PCA is sensitive to outliers, as it aims to maximize the variance in the data. Outliers can disproportionately influence the principal components, potentially leading to a skewed representation of the data.

3. **Interpretability of Components:**
   - While PCA provides a new set of uncorrelated variables (principal components), interpreting these components in the context of the original features may not always be straightforward. The interpretability of principal components can be challenging, especially in high-dimensional spaces.

4. **Orthogonality Assumption:**
   - PCA assumes that the principal components are orthogonal (uncorrelated). In practice, perfect orthogonality may not always be achieved, leading to potential issues in interpretation.

5. **Varimax Rotation:**
   - In cases where interpretability is crucial, researchers may use techniques like Varimax rotation to enhance the interpretability of the principal components. However, this introduces an additional step and may not always yield clear and meaningful results.

6. **Selection of Number of Components (K):**
   - Determining the optimal number of principal components (K) to retain can be subjective. Methods such as the scree plot, cumulative explained variance, or cross-validation may be used, but the choice of K can impact the results.

7. **Assumption of Gaussian Distribution (for statistical inference):**
   - If PCA is used for statistical inference, such as hypothesis testing, it assumes that the data follows a multivariate normal distribution. This assumption may not always hold in practice.

8. **Loss of Information:**
   - While PCA reduces dimensionality, it also involves a loss of information. The reduced-dimensional representation retains the most important information, but some details may be sacrificed.

9. **Not Robust to Non-Gaussian Distributions:**
   - PCA assumes that the data is generated from a Gaussian distribution. If the data has a non-Gaussian distribution or contains outliers, other techniques like Robust PCA may be more appropriate.

10. **Data Scaling Impact:**
    - PCA is sensitive to the scale of the variables. Standardizing variables is a common practice before applying PCA to ensure that variables with larger scales do not dominate the principal components.

## Covariance of Features
Principal Component Analysis (PCA) involves the computation of variance and covariance in the context of dimensionality reduction. Here's a breakdown of how variance and covariance are utilized in PCA:

1. **Covariance Matrix:**
   - In PCA, the first step is often to compute the covariance matrix ($C$) of the original data. The covariance between two features (variables) $X_i$ and $X_j$ is calculated as follows:

     $\text{cov}(X_i, X_j) = \frac{\sum_{k=1}^{m} (X_{ik} - \bar{X_i})(X_{jk} - \bar{X_j})}{m-1}$

   where $m$ is the number of samples, $X_{ik}$ is the $k$-th observation of variable $X_i$, and $\bar{X_i}$ is the mean of $X_i$. The covariance matrix $C$ is then constructed, where $C_{ij}$ is the covariance between features $X_i$ and $X_j$.

   $C = \begin{bmatrix} \text{cov}(X_1, X_1) & \text{cov}(X_1, X_2) & \ldots & \text{cov}(X_1, X_n) \\ \text{cov}(X_2, X_1) & \text{cov}(X_2, X_2) & \ldots & \text{cov}(X_2, X_n) \\ \vdots & \vdots & \ddots & \vdots \\ \text{cov}(X_n, X_1) & \text{cov}(X_n, X_2) & \ldots & \text{cov}(X_n, X_n) \end{bmatrix}$

2. **Eigenvalue Decomposition:**
   - The next step is to perform eigenvalue decomposition on the covariance matrix $C$. This yields a set of eigenvalues ($\lambda_1, \lambda_2, \ldots, \lambda_n$) and corresponding eigenvectors. The eigenvectors represent the principal components.

3. **Variance Explained:**
   - The eigenvalues represent the amount of variance captured by each principal component. The total variance in the data is the sum of all eigenvalues. The proportion of variance explained by the $i$-th principal component is given by:

     $\text{Proportion of Variance} = \frac{\lambda_i}{\sum_{j=1}^{n} \lambda_j}$

   The cumulative proportion of variance up to the $i$-th principal component is also calculated, helping determine the optimal number of components to retain.

4. **Selecting Principal Components:**
   - Principal components are selected based on the eigenvalues. Components with higher eigenvalues capture more variance and are retained, while those with lower eigenvalues may be discarded.

## What constraints make sense in reduced representation?  
- No feature correlation, i.e., all off-diagonals in $C_Z$ are zero
- Rank-ordered features by variance, i.e., sorted diagonals of $C_Z$

## Eigen Decomposition
Eigen decomposition, also known as spectral decomposition, is a factorization of a square matrix into a canonical form, which involves eigenvalues and eigenvectors. It is a fundamental concept in linear algebra and has various applications, including Principal Component Analysis (PCA), diagonalization of matrices, and solving systems of linear differential equations.

Given a square matrix $A$, the eigen decomposition is represented as:

$A = V \Lambda V^{-1}$

where:
- $V$ is the matrix whose columns are the eigenvectors of $A$.
- $\Lambda$ is a diagonal matrix containing the corresponding eigenvalues.

### Steps for Eigen Decomposition:

1. **Compute Eigenvalues and Eigenvectors:**
   - For a given square matrix $A$, solve the characteristic equation to find the eigenvalues $\lambda$:
     $\text{det}(A - \lambda I) = 0$
   - For each eigenvalue $\lambda$, find the corresponding eigenvector $v$ by solving the equation $Av = \lambda v$.

2. **Construct Matrix V:**
   - Assemble the eigenvectors into a matrix $V$, where each column corresponds to an eigenvector.

3. **Construct Diagonal Matrix $\Lambda$:**
   - Form a diagonal matrix $\Lambda$ with the eigenvalues on the main diagonal.

4. **Verify Eigen Decomposition:**
   - Check that $A = V \Lambda V^{-1}$ holds.

### Example:

Given a matrix $A$:
$A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}$

1. **Compute Eigenvalues and Eigenvectors:**
   - Solve the characteristic equation:
     $\text{det}(A - \lambda I) = 0$
   - Find the corresponding eigenvectors.

2. **Construct Matrix $V$:**
   - Assemble the eigenvectors into a matrix $V$.

3. **Construct Diagonal Matrix $\Lambda$:**
   - Form a diagonal matrix $\Lambda$ with the eigenvalues on the main diagonal.

4. **Verify Eigen Decomposition:**
   - Check that $A = V \Lambda V^{-1}$ holds.

### Notes:
- Eigen decomposition is not always possible for all matrices. It requires the matrix to be diagonalizable.
- If $A$ is a symmetric matrix, the eigenvectors can be chosen to be orthogonal, simplifying the decomposition.
- In certain cases, numerical stability issues may arise, and alternative methods like Singular Value Decomposition (SVD) are used.

## PCA Practical Tips
- PCA assumptions (linearity, orthogonality) not always appropriate
  - Various extensions to PCA with different underlying assumptions, e.g., manifold learning, Kernel PCA, ICA
- Centering is crucial, i.e., we must preprocess data so that all features have zero mean before applying PCA
  - PCA results dependent on scaling of data (Data is sometimes rescaled in practice before applying PCA)

## PCA Big Data
- Big n small d
  - O($d^2$) local storage, O($d^3$) local computation, O($dk$) communication  
  - Similar strategy as closed-form linear regression
  - Steps
    - Step 1: Center Data 
      - $\text{O(nd) Distributed Storage}$
      - $\text{O(d) Local Storage} \text{O(d) Local Computation} \text{O(d) Communication}$
    - Step 2: Compute Covariance or Scatter Matrix ($X^TX$)
      - Computer Scatter matrix with outer products
      - $\text{O(nd) Distributed Storage}$
      - O($d^2$) local storage, O($nd^2$) Distributed Computation  
      - O($d^2$) local storage, O($d^2$) Local Computation
    - Step 3: Eigen Decomposition
      - Perform Locally since d is reasonable
      - Communicate k principal components to workers
      - $\text{O(nd) Distributed Storage}$
      - O($d^2$) local storage, O($nd^2$) Distributed Computation  
      - O($d^2$) local storage, O($d^3$) Local Computation O($dk$) Communication
    - Step 4: Compute PCA Scores
      - Multiply each point by principal components, P
      - $\text{O(nd) Distributed Storage}$
      - $\text{O(dk) Local Computation}$

- Big n big d
  - O($dk + n$) local storage, computation; O($dk + n$) communication
  - Iterative Algorithm
    - Sequence of matrix- vector products
  - Krylov subspace or random projection methods
  - Krylov subspace methods (used in MLlib) iteratively compute $X^TXv$ for some $v_i \in R^d$ provided by the method
  - Requires O(k) passes over the data and O(dk) local storage
  - We don’t need to compute the covariance matrix!
  - Iterative Approach
    - Repeat for O(k) iterations:
      1. Communicate $v_i \in R^d$  to all workers
      2. Compute qi = $X^TXv_i$ in a distributed fashion
        - Step 1: $b_i$ = $Xv_i$  =  $[v_i^Tx(1)... v_i^Tx(n)]$
        - Step 2: $q_i$ = $X^Tb_i$ =  $\sum_{j=1}^{n}b_{ij}x^{(j)}$
      3. Driver uses $q_i$ to update estimate of P
  - Steps 
    - Step 1: $b_i$ = $Xv_i$ (each component is dot product then concatenate)
      - O(nd) Distributed Storage
      - O(d) Local Storage O(nd) Distributed Computation
      - O(n) Local Storage O(n) Local Computation O(n) Communication
    - Step 2: $q_i$ = $X^Tb_i$ (sum of rescaled data points)
      - O(nd) Distributed Storage
      - O(n) Local Storage O(nd) Distributed Computation
      - O(d) Local Storage O(d) Local Computation O(d) Communication

## Links
- [StatQuest: Principal Component Analysis (PCA), Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ) 
- [StatQuest: PCA - Practical Tips](https://www.youtube.com/watch?v=oRvgq966yZg)
	- All variables same scale. So divide the variables by their standard deviation
	- Center your data.
	- How many principal components can you expect to find? - There is technically for each feature in the dataset but if the sample is less than the number of features it puts an upper bound on the number of principal components.
  - [Lecture 47 — Singular Value Decomposition | Stanford University](https://www.youtube.com/watch?v=P5mlg91as1c)
  - [StatQuest: PCA main ideas in only 5 minutes!!!](https://www.youtube.com/watch?v=HMOI_lkzW08)