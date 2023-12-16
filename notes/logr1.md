# Logistic Regression 1
- Logistic Regression = Linear Classifier
- SVM hingeloss,adaboost exponential loss, logistic regression logistic loss
- Logistic Regression Gradient Descent
	Logistic Regression is a classification algorithm used to model the probability of a binary outcome. The logistic function (sigmoid function) is often used to transform a linear combination of input features into a value between 0 and 1, representing the probability of belonging to the positive class. Gradient descent is a common optimization algorithm used to find the optimal parameters (coefficients) of the logistic regression model.
	
	Here's the basic outline of how gradient descent works for logistic regression:
	
	### Logistic Regression Model:
	
	The logistic regression model is given by the sigmoid function:
	
	$P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + \beta_2x_2 + \ldots + \beta_nx_n)}}$
	
	Where:
	- $P(Y=1)$ is the probability of the positive class.
	- $e$ is the base of the natural logarithm.
	- $\beta_0, \beta_1, \ldots, \beta_n$ are the coefficients.
	- $x_1, x_2, \ldots, x_n$ are the input features.
	
	### Log-Likelihood Function:
	
	The log-likelihood function is used as the objective function to be minimized. For a dataset with $N$ samples, it is defined as:
	
	$\text{Log-Likelihood} = \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$
	
	Where:
	- $y_i$ is the actual class label (0 or 1) for the $i$-th sample.
	- $p_i$ is the predicted probability of belonging to the positive class for the $i$-th sample.
	
	### Gradient Descent:
	
	The gradient of the log-likelihood with respect to each coefficient $\beta_j$ is calculated:
	
	$\frac{\partial \text{Log-Likelihood}}{\partial \beta_j} = \sum_{i=1}^{N} (p_i - y_i) x_{ij}$
	
	Then, the coefficients are updated iteratively using the gradient descent update rule:
	
	$\beta_j := \beta_j - \alpha \frac{\partial \text{Log-Likelihood}}{\partial \beta_j}$
	
	Where:
	- $\alpha$ is the learning rate, a positive scalar that determines the step size in each iteration.
- 