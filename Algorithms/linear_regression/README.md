# Linear Regression

![](doc/linear_regression_example.png)

## What is Linear Regression?

In statistics, linear regression is a linear approach to modelling the relationship between a dependent variable(y) and one or more independent variables(X). In linear regression, the relationships are modeled using linear predictor functions whose unknown model parameters are estimated from the data. Linear Regression is one of the most popular algorithms in Machine Learning. That’s due to its relative simplicity and well known properties.

The best fit line can be calculated in multiple different ways including Least Squares Regression and Gradient Descent. In this guide we'll focus on using gradient descent since this is the most commonly used technique in Machine Learning.

## Simple Linear Regression

Linear Regression is called simple if you are only working with one independent variable.

Formula: <img src="tex/18813fabfad59d1ba84fc901ede9101f.svg?invert_in_darkmode" align=middle width=104.88954134999999pt height=24.65753399999998pt/>

### Cost Function

We can measure the accuracy of our linear regression algorithm using the **mean squared error** (mse) cost function. MSE measures the average squared distance between the predicted output and the actual output (label).

<p align="center"><img src="tex/660ef60b693132606dcc3aae53b147ca.svg?invert_in_darkmode" align=middle width=406.53640665pt height=47.806078649999996pt/></p>

### Optimiztation

To find the coefficients that minimize our error function we will use gradient descent. Gradient descent is a optimization algorithm which iteratively takes steps to the local minimum of the cost function.

To find the way towards the minimum we take the derivative of the error function in respect to our slope m and our y intercept b. Then we take a step in the negative direction of the derivative.

General Gradient Descent Formula:

<p align="center"><img src="tex/e37355cc0b5b07561247c00842519c04.svg?invert_in_darkmode" align=middle width=175.63739985pt height=38.5152603pt/></p>

Gradient Descent Formulas for simple linear regression:

<p align="center"><img src="tex/0822727d1cb885ac043eb8c23c6a8c06.svg?invert_in_darkmode" align=middle width=239.42691134999995pt height=47.806078649999996pt/></p>

<p align="center"><img src="tex/f28aee7ec74570ba081a608f7b5d88bb.svg?invert_in_darkmode" align=middle width=217.1808045pt height=47.806078649999996pt/></p>

## Multivariate Linear Regression

Linear Regression is called multivariate if you are working with at least two independent variables. Each of the independent variables also called features gets multiplied with a weight which is learned by our linear regression algorithm.

<p align="center"><img src="tex/695de53e837a94510d8695f780f764d1.svg?invert_in_darkmode" align=middle width=452.9069325pt height=44.89738935pt/></p>

Loss and optimizer are the same as for simple linear regression. The only difference is that the optimizer is now used for any weight (<img src="tex/4b4518f1b7f0fb1347fa21506ebafb19.svg?invert_in_darkmode" align=middle width=18.32105549999999pt height=14.15524440000002pt/> to <img src="tex/c2a29561d89e139b3c7bffe51570c3ce.svg?invert_in_darkmode" align=middle width=16.41940739999999pt height=14.15524440000002pt/>) instead of only for m and b.

## Normal Equation

Another way to find the optimal coefficients is to use the "Normal Equation". The "Normal Equation" is an analytical approach for finding the optimal coefficients without needing to iterate over the data.

<p align="center"><img src="tex/4bf055a6a961b27706b75bc7e08a0f29.svg?invert_in_darkmode" align=middle width=139.6342233pt height=23.755462499999997pt/></p>

Contrary to Gradient Descent, when using the Normal Equation, features don't need to be scaled. The Normal Equation works well for datasets with few features but can be slow as the number of features increases due to the high computational complexity of computing the inverse <img src="tex/c116dfb62bb6eadf90bac11393f97a66.svg?invert_in_darkmode" align=middle width=43.570210199999984pt height=26.76175259999998pt/>.

Further readings:
* [Lecture 4.6 — Linear Regression With Multiple Variables | Normal Equation — [Andrew Ng]](https://www.youtube.com/watch?v=B-Ks01zR4HY)
* [Derivation of the Normal Equation for linear regression](https://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression)

## Regularization

Regularization are techniques used to reduce overfitting. This is really important to create models that generalize well on new data.

![Regularization](doc/regularization.png)

Mathematically speaking, it adds a regularization term in order to prevent the coefficients to fit so perfectly to overfit. For Linear Regression we can decide between two techniques – L1 and L2 Regularization.

For more information on the difference between L1 and L2 Regularization check out the following article:

* http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/

You can add regularization to Linear Regression by adding regularization term to either the loss function or to the weight update.

L1 regularization:

<p align="center"><img src="tex/ef27eeeeeadc48f3a48118fbf65ff125.svg?invert_in_darkmode" align=middle width=340.32351374999996pt height=59.1786591pt/></p>

L2 regularization:

<p align="center"><img src="tex/ac342f337b60a671151324a7a222d777.svg?invert_in_darkmode" align=middle width=332.09575739999997pt height=59.1786591pt/></p>

## ElasticNet

ElasticNet is a regularization technique that linearly combines the L1 and L2 penalties. 

<p align="center"><img src="tex/eedb3ae6d88cd2296e4c9acfe5658b09.svg?invert_in_darkmode" align=middle width=567.3416066999999pt height=59.1786591pt/></p>

Resources:
* [Regularization Part 3: Elastic Net Regression](https://www.youtube.com/watch?v=1dKRdX9bfIo)

## Polynomial Regression

Polynomial Regression is a form of regression analysis that models the relationship between the independent variables <img src="tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=22.465723500000017pt/> and the dependent variable <img src="tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.649225749999989pt height=14.15524440000002pt/> as an <img src="tex/87a75da6a417d9d9fd57f0b9b24473d2.svg?invert_in_darkmode" align=middle width=25.274089499999988pt height=22.831056599999986pt/> degree polynomial in <img src="tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=22.465723500000017pt/>.

<p align="center"><img src="tex/2d3d16f648bb613710e8ed0a19f2fe17.svg?invert_in_darkmode" align=middle width=415.0087359pt height=18.312383099999998pt/></p>

Resources:
* [Polynomial regression Wikipedia](https://en.wikipedia.org/wiki/Polynomial_regression)
* [Sklearn PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)

## Code

* [Simple Linear Regression](code/simple_linear_regression.py)
* [Multivariate Linear Regression](code/multivariate_linear_regression.py)
* [Lasso Regression](code/lasso_regression.py)
* [Ridge Regression](code/ridge_regression.py)
* [ElasticNet](code/elastic_net.py)
* [Polynomial Regression](code/polynomial_regression.py)
* [Linear Regression Explained](code/linear_regression_explained.ipynb)

## Credit / Other resources

* [Linear Regression (Wikipedia)](https://en.wikipedia.org/wiki/Linear_regression)
* [Simple and Multiple Linear Regression in Python (Adi Bronshtein on Medium)](https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9)
* [Linear Regression (Scikit Learn Documentation)](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)