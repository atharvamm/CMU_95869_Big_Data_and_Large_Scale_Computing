# Linear Regression 1
- Linear Regression
	Linear regression is a statistical method used for modeling the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data. The simplest form of linear regression with one independent variable is known as simple linear regression, while the case with multiple independent variables is called multiple linear regression.
	
	Here's a breakdown of the key components and concepts associated with linear regression:
	
	1. **Dependent Variable (Response Variable):**
	   - The variable that we want to predict or explain. It is typically denoted as $Y$.
	
	2. **Independent Variable(s) (Predictor Variable(s)):**
	   - The variable(s) used to predict or explain the dependent variable. In simple linear regression, there is only one independent variable, denoted as $X$. In multiple linear regression, there are multiple independent variables, denoted as $X_1, X_2, \ldots, X_n$.
	
	3. **Linear Equation:**
	   - The relationship between the dependent variable and the independent variable(s) is represented by a linear equation. For simple linear regression:
	     $Y = \beta_0 + \beta_1X + \varepsilon$
	   - $\beta_0$ is the y-intercept (the value of $Y$ when $X$ is 0).
	   - $\beta_1$ is the slope (the change in $Y$ for a unit change in $X$).
	   - $\varepsilon$ represents the error term, which accounts for the variability in $Y$ that is not explained by the linear relationship with $X$.
	
	4. **Objective:**
	   - The objective in linear regression is to find the values of $\beta_0$ and $\beta_1$ that minimize the sum of squared differences between the observed values of $Y$ and the values predicted by the linear equation.
	
	5. **Least Squares Method:**
	   - The most common approach to finding the optimal values for $\beta_0$ and $\beta_1$ is the least squares method. It minimizes the sum of the squared residuals (differences between observed and predicted values).
	
	6. **Assumptions:**
	   - Linear regression assumes that there is a linear relationship between the dependent and independent variables.
	   - The residuals (the differences between observed and predicted values) are normally distributed.
	   - Homoscedasticity: The variance of the residuals is constant across all levels of the independent variable(s).
	   - Independence of residuals: The residuals are independent of each other.
	
	7. **Interpretation:**
	   - The slope ($\beta_1$) represents the change in the dependent variable for a one-unit change in the independent variable.
	   - The y-intercept ($\beta_0$) is the value of the dependent variable when the independent variable is 0 (which may or may not be meaningful depending on the context).
- Least Squares Regression
	Least Squares Regression is a method used in linear regression analysis to find the best-fitting linear relationship between a dependent variable (often denoted as $Y$) and one or more independent variables (often denoted as $X_1, X_2, \ldots, X_n$). The "best-fitting" line is the one that minimizes the sum of the squared differences (residuals) between the observed values and the values predicted by the linear equation.
	
	In the context of simple linear regression with one independent variable ($X$), the linear equation is represented as:
	
	$Y = \beta_0 + \beta_1X + \varepsilon$
	
	Here, $\beta_0$ is the y-intercept, $\beta_1$ is the slope, and $\varepsilon$ is the error term. The goal is to find the values of $\beta_0$ and $\beta_1$ that minimize the sum of the squared differences between the observed $Y$ values and the values predicted by the equation.
	
	The least squares method achieves this by defining the sum of squared residuals (SSR) as the objective function to be minimized. The residuals ($e_i$) are the differences between the observed $Y$ values ($Y_i$) and the predicted values ($\hat{Y}_i$):
	
	$SSR = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$
	
	The least squares approach aims to find the values of $\beta_0$ and $\beta_1$ that minimize this sum. The formulas for calculating the optimal values are:
	
	$\beta_1 = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^{n} (X_i - \bar{X})^2}$
	
	$\beta_0 = \bar{Y} - \beta_1 \bar{X}$
	
	Where:
	- $\bar{Y}$ is the mean of the observed $Y$ values.
	- $\bar{X}$ is the mean of the observed $X$ values.
	
	The least squares method is widely used because it has a closed-form solution, and it provides unique estimates for the coefficients $\beta_0$ and $\beta_1$ that minimize the sum of squared residuals. It is a fundamental technique in regression analysis and serves as the basis for linear modeling in various fields.
- Closed Form Solution of Least Squares Regression
	In multiple linear regression, where there are multiple independent variables (features), the closed-form solution involves matrix algebra. Let's consider the following notation:
	
	- $Y$: Vector of observed values of the dependent variable (size $n \times 1$, where $n$ is the number of observations).
	- $X$: Matrix of observed values of independent variables (size $n \times (k+1)$, where $k$ is the number of independent variables). The first column is a column of ones to account for the intercept term.
	- $\beta$: Vector of coefficients (size $(k+1) \times 1$).
	- $\varepsilon$: Vector of residuals (size $n \times 1$).
	
	The multiple linear regression model is represented as:
	
	$Y = X\beta + \varepsilon$
	
	The least squares objective is to minimize the sum of squared residuals:
	
	$SSR = \varepsilon^T \varepsilon$
	
	The closed-form solution for the coefficients $\beta$ is obtained by setting the derivative of $SSR$ with respect to $\beta$ equal to zero:
	
	$\frac{\partial SSR}{\partial \beta} = -2X^T(Y - X\beta) = 0$
	
	Solving for $\beta$, we get:
	
	$X^T Y = X^T X \beta$
	
	Now, solving for $\beta$ gives:
	
	$\beta = (X^T X)^{-1} X^T Y$
	
	Here, $(X^T X)^{-1}$ represents the inverse of the matrix $X^T X$.
	
	This closed-form solution provides the values for the coefficients that minimize the sum of squared residuals. However, it's important to note that the closed-form solution might not be feasible or optimal in certain situations, especially when dealing with a large number of features or when the matrix $X^T X$ is not invertible. In such cases, numerical optimization methods like gradient descent are often used to find the optimal coefficients.
- Lasso and Rdige Regression
	Lasso Regression and Ridge Regression are techniques used in linear regression to address some of its limitations and improve the model's performance, especially when dealing with multicollinearity or when the number of features is large compared to the number of observations.
	
	### Ridge Regression:
	
	**Objective Function:**
	In Ridge Regression, the goal is to minimize the following objective function:
	
	$SSR + \lambda \sum_{j=1}^{p} \beta_j^2$
	
	Here, $SSR$ is the sum of squared residuals (similar to ordinary least squares), $\lambda$ is the regularization parameter (a non-negative constant), and the second term is the penalty term. The penalty term is proportional to the square of the magnitude of the coefficients ($\beta_j$).
	
	**Objective:**
	Ridge Regression aims to find the values of the coefficients ($\beta$) that not only fit the data well but also keep the magnitudes of the coefficients small. This helps in preventing overfitting, especially when there are multicollinearity issues.
	
	**Closed-form Solution:**
	The closed-form solution for Ridge Regression is given by:
	
	$\beta = (X^T X + \lambda I)^{-1} X^T Y$
	
	Here, $X$ is the matrix of independent variables, $Y$ is the vector of dependent variable values, $I$ is the identity matrix, and $\lambda$ is the regularization parameter.
	
	### Lasso Regression:
	
	**Objective Function:**
	In Lasso Regression, the objective function to be minimized is slightly different:
	
	$SSR + \lambda \sum_{j=1}^{p} |\beta_j|$
	
	Similar to Ridge, $SSR$ is the sum of squared residuals, $\lambda$ is the regularization parameter, and the second term is the penalty term. In Lasso, the penalty term is proportional to the absolute values of the coefficients.
	
	**Objective:**
	Lasso Regression encourages sparsity in the coefficient vector, meaning it tends to force some of the coefficients to be exactly zero. This property makes Lasso useful for feature selection, as it can effectively shrink some coefficients to zero and exclude corresponding features from the model.
	
	**Optimization:**
	Unlike Ridge Regression, Lasso doesn't have a closed-form solution because of the non-differentiability of the absolute value function. It is typically solved using optimization algorithms, such as coordinate descent.
	
	Both Ridge and Lasso Regression are examples of regularized linear regression techniques that provide a trade-off between fitting the data well and keeping the model simple to avoid overfitting. The choice between Ridge and Lasso depends on the specific characteristics of the data and the goals of the modeling task.`
- Closed Form Complexity
	- Computation: O(nd2 + d3) operations
	- Matrix Multiplication of $X^TX$ is O($nd^2$)
	- Matrix Inverse of $X$ is O($n^3$)
	- Storage - O($nd + d^2$)
- Distributed Linear Regression
	- Big N and Small d
		- Assume O($d^3$) computation and O($d^2$) storage feasible on single machine
		- $O(nd) Distributed Storage$
		- $O(nd^2) Distributed Computation\ O(d^2) \text{Local Storage}$ 
		- $O(d^3) Local Computation\ O(d^2) Local Storage$
		- Calculate outer product of a subset of rows and then bring back the matrices to sum all the matrices and finally invert it.
		- As d grows this operation becomes computationally expensive
	- Big N and Big d
		- Use gradient descent
		- $\text{O(nd) Distributed Storage}$
		- $\text{O(nd) Distributed Computation} \ \text{O(d) Local Storage}$
		- $\text{O(d) Local Computation} \ \text{O(d) Local Storage}$
	