# Linear Regression 2
- Stopping Criterion for gradient descent
	- Updates become small
	- Predetermined maximum number of iterations
- Gradient Descent for Least Squares Regression
	Gradient descent is an iterative optimization algorithm commonly used to find the minimum of a function. In the context of multivariate least squares regression, the goal is to minimize the sum of squared residuals by adjusting the coefficients in the regression equation. Here's an outline of how gradient descent works for this purpose:
	
	### Objective Function:
	
	The objective function for multivariate least squares regression is the sum of squared residuals:
	
	$SSR = \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$
	
	Where $Y_i$ is the observed value, and $\hat{Y}_i$ is the predicted value based on the current coefficients.
	
	### Coefficient Update Rule:
	
	The gradient descent algorithm updates the coefficients iteratively using the gradient of the objective function with respect to each coefficient. The general update rule for the $j$-th coefficient ($\beta_j$) is as follows:
	
	$\beta_j := \beta_j - \alpha \frac{\partial SSR}{\partial \beta_j}$
	
	Where:
	- $\alpha$ is the learning rate, a positive scalar that determines the step size in each iteration.
	- $\frac{\partial SSR}{\partial \beta_j}$ is the partial derivative of the objective function with respect to $\beta_j$.
	
	### Partial Derivative Calculation:
	
	The partial derivative for the $j$-th coefficient is given by:
	
	$\frac{\partial SSR}{\partial \beta_j} = -2\sum_{i=1}^{n} x_{ij}(Y_i - \hat{Y}_i)$
	
	Where:
	- $x_{ij}$ is the $i$-th observation of the $j$-th independent variable.
	
	### Iterative Process:
	
	1. Initialize the coefficients ($\beta_0, \beta_1, \ldots, \beta_k$) randomly or with some initial values.
	2. Repeat until convergence:
	    - Calculate the predicted values $\hat{Y}_i$ using the current coefficients.
	    - Update each coefficient using the gradient descent rule.
	3. Continue the iterations until the change in the objective function is small or after a fixed number of iterations.	
	Note: Proper feature scaling may be necessary for effective convergence, especially when features are on different scales. Additionally, the learning rate ($\alpha$) should be chosen carefully to ensure convergence without overshooting.
- Gradient Descent Summary
	- Pros:  
		- Easily parallelized  
		- Cheap at each iteration
	- Cons:  
		- Slow convergence (especially compared with closed-form)
		- Stochastic variants can make things even cheaper
		- Requires communication across nodes!
- Rules of thumb for distributed ml
	- Computation and storage should be linear (in n, d)
	- Perform parallel and in-memory computation
	- Minimize Network Communication
- 