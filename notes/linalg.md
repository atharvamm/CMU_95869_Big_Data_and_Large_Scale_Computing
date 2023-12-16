# Linear Algebra and Complexity Review
- Bag of Words
	A Bag of Words (BoW) is a representation used in natural language processing and information retrieval, particularly in the context of text data. The idea behind a Bag of Words is to represent a document as an unordered set of words, disregarding grammar and word order but keeping track of word frequency. It is a common and simple way to convert textual data into a numerical format that can be used as input for machine learning algorithms.
	
	Here's how the Bag of Words representation works:
	
	1. **Tokenization:**
	   - The first step is to break down a text into individual words or tokens. This process is called tokenization.
	
	2. **Vocabulary Creation:**
	   - Next, create a vocabulary, which is a unique set of all the words present in the entire corpus (collection of documents).
	
	3. **Document Representation:**
	   - Represent each document in the corpus as a vector, where each element corresponds to a word in the vocabulary, and the value represents the frequency of that word in the document.
	
	4. **Sparse Matrix:**
	   - Since most documents use only a small subset of the entire vocabulary, the resulting representation is often a sparse matrix where most elements are zero.
	
	Let's consider a simple example:
	
	- **Document 1:** "The cat in the hat."
	- **Document 2:** "The quick brown fox."
	
	**Tokenization:**
	- Document 1: ["The", "cat", "in", "the", "hat"]
	- Document 2: ["The", "quick", "brown", "fox"]
	
	**Vocabulary:**
	- ["The", "cat", "in", "the", "hat", "quick", "brown", "fox"]
	
	**Bag of Words Representation:**
	- Document 1: [2, 1, 1, 2, 1, 0, 0, 0]
	- Document 2: [1, 0, 0, 1, 0, 1, 1, 1]
	
	In the representation above, each element in the vector corresponds to a word in the vocabulary, and the value represents the frequency of that word in the document. The order of words is ignored.
	
	**Use Cases:**
	- The Bag of Words model is often used in text classification, sentiment analysis, and document clustering.
	- It simplifies complex documents into a numerical format suitable for machine learning algorithms.
	- While it discards word order and structure, it is computationally efficient and works well in various text analysis scenarios.
	
	**Challenges:**
	- It doesn't capture the semantic meaning of words.
	- The representation can be very high-dimensional, especially for large vocabularies.
	
	To address some of the limitations, more advanced techniques like TF-IDF (Term Frequency-Inverse Document Frequency) and word embeddings (Word2Vec, GloVe) have been developed, providing richer representations for textual data.
- Matrix Transpose
	The transpose of a matrix is an operation that flips the matrix over its main diagonal, which runs from the top-left to the bottom-right. If you have a matrix A with dimensions m x n, the transpose of A, denoted as A^T, is an n x m matrix where the rows and columns are swapped. In other words, if A = [a_ij] (where a_ij represents the element in the i-th row and j-th column of A), then A^T = [b_ij], where b_ij = a_ji.
	
	Here's a simple example:
	
	$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$
	
	The transpose of A (A^T) is:
	
	$A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$
	
	You can see that the rows of A become the columns of A^T, and vice versa.
	
	In mathematical notation, if A = [a_ij] (m x n), then A^T = [b_ij] (n x m), where b_ij = a_ji.
	
	Some properties of matrix transposition include:
	
	1. $(A^T)^T = A$ (Transposing twice returns the original matrix)
	2. $(cA)^T = cA^T$ (Transposing a scalar multiple of a matrix is the same as the scalar multiple of the transposed matrix)
	3. $(A + B)^T = A^T + B^T$ (Transposing a sum of matrices is the same as the sum of their transposes)
	
	Matrix transposition is a fundamental operation in linear algebra and has various applications in fields such as computer science, physics, and engineering.
- Inner/Scalar Product	
	### Inner Product:
	
	An inner product is a mathematical operation that takes two vectors and produces a scalar. It's a way to measure the angle between two vectors or the projection of one vector onto another. In general, for vectors **u** and **v**, the inner product is denoted as ⟨**u**, **v**⟩ or **u · v**.
	
	The inner product has a few key properties:
	
	1. **Linearity**: ⟨a**u** + b**v**, **w**⟩ = a⟨**u**, **w**⟩ + b⟨**v**, **w**⟩
	2. **Symmetry**: ⟨**u**, **v**⟩ = ⟨**v**, **u**⟩
	3. **Positivity**: ⟨**u**, **u**⟩ ≥ 0, and ⟨**u**, **u**⟩ = 0 if and only if **u** is the zero vector.
	
	### Scalar Product:
	
	The scalar product is a more general term that refers to any operation that combines two vectors to produce a scalar. The inner product is a specific type of scalar product. Other examples of scalar products include the dot product and the cross product.
	
	### Dot Product:
	
	The dot product is a specific type of inner product. For two vectors **u** = [u₁, u₂, ..., uₙ] and **v** = [v₁, v₂, ..., vₙ], the dot product (**u · v**) is calculated as:
	
	\[ **u · v** = u₁v₁ + u₂v₂ + ... + uₙvₙ \]
	
	The dot product is a fundamental operation in linear algebra, and it's often used to find the angle between vectors, calculate projections, and solve various geometric and mathematical problems.
	
- Matrix - Vector Product
	The matrix-vector product is a fundamental operation in linear algebra, where a matrix is multiplied by a vector to produce another vector. Let's consider a matrix $A$ and a column vector $\mathbf{v}$:
	
	$A = \begin{bmatrix} a_{11} & a_{12} & \ldots & a_{1n} \\ a_{21} & a_{22} & \ldots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \ldots & a_{mn} \end{bmatrix}$
	
	$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$
	
	The product $\mathbf{w} = A \cdot \mathbf{v}$ is computed as follows:
	
	$\mathbf{w} = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_m \end{bmatrix} = \begin{bmatrix} a_{11}v_1 + a_{12}v_2 + \ldots + a_{1n}v_n \\ a_{21}v_1 + a_{22}v_2 + \ldots + a_{2n}v_n \\ \vdots \\ a_{m1}v_1 + a_{m2}v_2 + \ldots + a_{mn}v_n \end{bmatrix}$
	
	Each element $w_i$ of the resulting vector $\mathbf{w}$ is obtained by taking the dot product of the corresponding row of matrix $A$ and the column vector $\mathbf{v}$.
	
	The matrix-vector product is a common operation in various applications, including computer graphics, physics simulations, and solving systems of linear equations. It's a fundamental building block in linear algebra and is used in many numerical algorithms.
- Scalar product is commonly represented as $X^TW$
- Matrix - Matrix Product
	The matrix-matrix product, also known as matrix multiplication, is a fundamental operation in linear algebra. Let's consider two matrices, $A$ and $B$:
	
	$A = \begin{bmatrix} a_{11} & a_{12} & \ldots & a_{1m} \\ a_{21} & a_{22} & \ldots & a_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ a_{n1} & a_{n2} & \ldots & a_{nm} \end{bmatrix}$
	
	$B = \begin{bmatrix} b_{11} & b_{12} & \ldots & b_{1p} \\ b_{21} & b_{22} & \ldots & b_{2p} \\ \vdots & \vdots & \ddots & \vdots \\ b_{m1} & b_{m2} & \ldots & b_{mp} \end{bmatrix}$
	
	The product $C = A \cdot B$ is a new matrix $C$ with dimensions $n \times p$. The elements of $C$ are computed as follows:
	
	$c_{ij} = a_{i1}b_{1j} + a_{i2}b_{2j} + \ldots + a_{im}b_{mj}$
	
	In other words, each element $c_{ij}$ of the resulting matrix $C$ is obtained by taking the dot product of the $i$-th row of matrix $A$ and the $j$-th column of matrix $B$.
	
	Matrix multiplication is not commutative, meaning $A \cdot B$ is generally not equal to $B \cdot A$. The order of multiplication matters.
	
	Matrix multiplication is a crucial operation used in various fields, including computer graphics, physics simulations, and solving systems of linear equations. It is a fundamental building block for many mathematical and numerical algorithms.
- Outer Product
	The outer product is a mathematical operation that takes two vectors and produces a matrix. Given two vectors $\mathbf{u}$ and $\mathbf{v}$, the outer product is denoted as $\mathbf{u} \otimes \mathbf{v}$ and results in a matrix.
	
	If $\mathbf{u}$ is a column vector of size $m$ and $\mathbf{v}$ is a row vector of size $n$, then the outer product $\mathbf{u} \otimes \mathbf{v}$ will result in an $m \times n$ matrix. Each element $C_{ij}$ of the resulting matrix is obtained by multiplying the $i$-th element of $\mathbf{u}$ by the $j$-th element of $\mathbf{v}$:
	
	$C_{ij} = u_i \cdot v_j$
	
	Mathematically, if:
	
	$\mathbf{u} = \begin{bmatrix} u_1 \\ u_2 \\ \vdots \\ u_m \end{bmatrix}$
	
	and
	
	$\mathbf{v} = \begin{bmatrix} v_1 & v_2 & \ldots & v_n \end{bmatrix}$
	
	then:
	
	$\mathbf{u} \otimes \mathbf{v} = \begin{bmatrix} u_1 v_1 & u_1 v_2 & \ldots & u_1 v_n \\ u_2 v_1 & u_2 v_2 & \ldots & u_2 v_n \\ \vdots & \vdots & \ddots & \vdots \\ u_m v_1 & u_m v_2 & \ldots & u_m v_n \end{bmatrix}$
	
	The outer product is used in various mathematical and scientific applications, such as in the computation of covariance matrices, spectral decompositions, and other linear algebra operations. It provides a way to represent the relationships between elements of two vectors in the form of a matrix.
- Identity Matrices
	An identity matrix, often denoted as $I$ or $I_n$, is a square matrix with ones on its main diagonal (from the top-left to the bottom-right) and zeros elsewhere. In other words, the identity matrix is a special matrix that, when multiplied with another matrix, leaves the other matrix unchanged.
	
	The $n \times n$ identity matrix $I_n$ is defined as follows:
	
	$I_n = \begin{bmatrix} 1 & 0 & \ldots & 0 \\ 0 & 1 & \ldots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \ldots & 1 \end{bmatrix}$
	
	In this matrix, all the elements $I_{ij}$ are zero unless $i = j$, in which case $I_{ij} = 1$. The size of the identity matrix is determined by the value of $n$.
	
	The identity matrix has several important properties:
	
	1. **Multiplicative Identity**: For any square matrix $A$ of appropriate size, $A \cdot I = I \cdot A = A$.
	
	2. **Inverse**: The identity matrix is its own inverse. If $A$ is a square matrix and $A \cdot A^{-1} = A^{-1} \cdot A = I$, then $A^{-1}$ is the inverse of $A$.
	
	Identity matrices are fundamental in linear algebra and various mathematical operations. They play a crucial role in defining the concept of matrix multiplication, and their properties are essential in solving systems of linear equations and many other mathematical applications.
- Inverse Matrix
	The inverse of a square matrix $A$, denoted as $A^{-1}$, is a matrix such that when $A$ is multiplied by its inverse, the result is the identity matrix $I$. Mathematically, this is expressed as:
	
	$A \cdot A^{-1} = A^{-1} \cdot A = I$
	
	Not all matrices have an inverse. For a matrix to have an inverse, it must be a square matrix (having the same number of rows and columns), and its determinant must be non-zero.
	
	If $A$ is a $n \times n$ matrix, the inverse $A^{-1}$ is calculated using the following formula:
	
	$A^{-1} = \frac{1}{\text{det}(A)} \cdot \text{adj}(A)$
	
	Here, $\text{det}(A)$ is the determinant of matrix $A$, and $\text{adj}(A)$ is the adjugate (or adjoint) matrix of $A$.
	
	The elements of the adjugate matrix $\text{adj}(A)$ are obtained by taking the transpose of the cofactor matrix of $A$. The cofactor of an element $a_{ij}$ is given by $(-1)^{i+j} \cdot M_{ij}$, where $M_{ij}$ is the determinant of the matrix obtained by deleting the $i$-th row and $j$-th column of $A$.
	
	It's important to note that not all matrices have an inverse. If the determinant of a matrix is zero, it is said to be singular, and such matrices do not have an inverse.

- Euclidea Norm as scalar product $||X||^2_2 = X^TX$
- Big O Notation
	Big O notation is a mathematical notation that describes the limiting behavior of a function when its argument approaches a particular value or infinity. In computer science, Big O notation is commonly used to analyze and describe the efficiency or complexity of algorithms. It helps express how the runtime or space requirements of an algorithm grow as the size of the input increases.
	
	The notation is called "Big O" because it uses the letter "O" followed by a function, often in terms of the input size $n$. The expression $O(g(n))$ represents the upper bound of the growth rate of a function $f(n)$ in relation to $g(n)$. Specifically, $f(n)$ is said to be $O(g(n))$ if there exist constants $c$ and $n_0$ such that for all $n$ greater than or equal to $n_0$, the following inequality holds:
	
	$0 \leq f(n) \leq c \cdot g(n)$
	
	This notation helps us describe the efficiency of an algorithm in terms of its worst-case behavior. Common time complexities described using Big O notation include:
	
	- $O(1)$: Constant time complexity. The algorithm's runtime does not depend on the size of the input.
	- $O(\log n)$: Logarithmic time complexity. Common in algorithms with divide-and-conquer strategies, like binary search.
	- $O(n)$: Linear time complexity. The runtime grows linearly with the size of the input.
	- $O(n \log n)$: Linearithmic time complexity. Common in efficient sorting algorithms, like mergesort and heapsort.
	- $O(n^2)$, $O(n^3)$, ...: Polynomial time complexity. Common in algorithms with nested loops.
	- $O(2^n)$, $O(n!)$, ...: Exponential and factorial time complexity. Often associated with inefficient algorithms.
- O(1) complexity
	In algorithm analysis, $O(1)$ denotes constant time complexity. An algorithm has $O(1)$ time complexity if its execution time does not depend on the size of the input. In other words, the algorithm's performance remains constant regardless of how large the input becomes.
	
	For example, accessing an element in an array by its index is an $O(1)$ operation because the time it takes to access a specific element is constant, regardless of the array's size. Similarly, performing basic arithmetic operations, assigning a value to a variable, or looking up a value in a hash table (with good hash function and collision resolution) are $O(1)$ operations.
	
	It's important to note that $O(1)$ does not mean the operation takes exactly one unit of time; it simply means the time complexity is constant with respect to the input size.
	
	Examples of $O(1)$ operations:
	
	1. Accessing an element by index in an array.
	2. Assigning a value to a variable.
	3. Checking if a number is even or odd.
	4. Looking up a value in a well-implemented hash table.
	5. Basic arithmetic operations (e.g., addition, subtraction, multiplication) on fixed-size integers.
	
	In contrast to algorithms with higher time complexities (such as linear or quadratic), $O(1)$ algorithms are generally considered very efficient. They provide constant and predictable performance regardless of the input size, making them desirable for certain tasks, especially when dealing with large datasets.
- O(n) Complexity
	$O(n)$ denotes linear time complexity in algorithm analysis. An algorithm has $O(n)$ time complexity if its execution time grows linearly with the size of the input. In other words, as the input size ($n$) increases, the time taken by the algorithm also increases linearly.
	
	For example, if you have an algorithm that iterates through each element in an array exactly once, it has linear time complexity. The number of operations required is directly proportional to the size of the input array.
	
	Examples of $O(n)$ operations:
	
	1. **Linear Search:** Searching for an element in an unsorted array by checking each element one by one.
	
	2. **Iterating Through an Array:** Performing an operation on each element of an array.
	
	3. **Finding the Maximum Element in an Array:** Examining each element in the array to find the maximum.
	
	4. **Counting the Occurrences of an Element in an Array:** Counting how many times a specific element appears in an array.
	
	The efficiency of linear time complexity is generally considered reasonable, especially for small to moderate-sized inputs. However, it might not be optimal for very large datasets, where algorithms with better time complexities (such as $O(\log n)$ or $O(1)$) could be more desirable.
	
	In Big O notation, the actual coefficient and lower-order terms are ignored, focusing on the dominant term that grows the fastest with the input size. For linear time complexity, the algorithm's running time is proportional to $n$ (possibly multiplied by a constant factor), and it scales linearly with the size of the input.
- O($n^2$) Complexity
	$O(n^2)$ denotes quadratic time complexity in algorithm analysis. An algorithm has $O(n^2)$ time complexity if its running time grows quadratically with the size of the input.
	
	In mathematical terms, $O(n^2)$ means that the number of operations required by the algorithm is proportional to the square of the size of the input ($n$). This complexity often arises in algorithms that involve nested iterations, where the number of iterations in one loop is dependent on the number of iterations in another.
	
	Examples of algorithms with $O(n^2)$ time complexity:
	
	1. **Bubble Sort:** A simple sorting algorithm that repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order. The worst-case time complexity is $O(n^2)$.
	
	2. **Insertion Sort:** A sorting algorithm that builds the final sorted array one item at a time. The worst-case time complexity is $O(n^2)$.
	
	3. **Selection Sort:** A sorting algorithm that divides the input list into a sorted and an unsorted region. In each iteration, the minimum element from the unsorted region is selected and moved to the sorted region. The worst-case time complexity is $O(n^2)$.
	
	4. **Matrix Multiplication (Naive Algorithm):** The standard matrix multiplication algorithm involves three nested loops, resulting in $O(n^3)$ time complexity. A naive implementation with three nested loops for $n \times n$ matrices can be categorized as $O(n^2)$ for the square of the input size.
	
	Algorithms with $O(n^2)$ time complexity can become inefficient for large input sizes, and efforts are often made to improve the time complexity to $O(n \log n)$ or even $O(n)$ where possible, especially for larger datasets.
- Matrix - Matrix Product to multiply nXm with mXp. O(npm) time and O(np) disk storage
- Matrix Inversion - n^3 to computer and n^2 to store
- Outer product - n^2 time and space complexity
- Adding Vectors - n time and space complexity