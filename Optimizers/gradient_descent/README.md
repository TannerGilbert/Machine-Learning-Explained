# Gradient Descent

Gradient descent is an optimization algorithm. Optimization refers to the task of minimizing/maximizing an objective function <img src="tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode" align=middle width=31.655311049999987pt height=24.65753399999998pt/> parameterized by <img src="tex/e1977b3bd8b60ca5e8e3c3b921470696.svg?invert_in_darkmode" align=middle width=46.979921999999995pt height=27.91243950000002pt/>. Gradient Descent minimizes the objective function <img src="tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode" align=middle width=31.655311049999987pt height=24.65753399999998pt/> by iteratively moving in the direction of steepest descent as defined by the negative gradient of the objective function <img src="tex/386e10624041d64770c6785c1034b111.svg?invert_in_darkmode" align=middle width=65.57658359999998pt height=24.65753399999998pt/> with respect to the parameters <img src="tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/>. The size of the steps taken in the negative gradient direction is determined by the learning rate <img src="tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.57650494999999pt height=14.15524440000002pt/>.

<p align="center"><img src="tex/1f52020ae24caeeaeeb316d2525450a2.svg?invert_in_darkmode" align=middle width=138.1617303pt height=16.438356pt/></p>

![Gradient Descent gif](doc/gradient_descent.gif)

Picking a proper value for the learning rate <img src="tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.57650494999999pt height=14.15524440000002pt/> is important as a too small learning rate can lead to slow convergence, while a too large learning rate can hinder convergence and cause the loss function to fluctuate around the minimum or even to diverge.

![Pick learning rate](doc/pick_learning_rate.png)

*Source: [CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/neural-networks-3/)*

## Gradient descent variants

There are three variants of gradient descent. They differ in the amount of data that is used to compute the gradient of the objective function. Depending on the amount of data, there is a trade-off between the accuracy of the parameter update (weight update) and the time it takes to perform an update.

### Batch gradient descent

Batch gradient descent computes the gradient of the objective function / cost function <img src="tex/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode" align=middle width=31.655311049999987pt height=24.65753399999998pt/> with respect to the parameters <img src="tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/> on the entire training data-set.

<p align="center"><img src="tex/1f52020ae24caeeaeeb316d2525450a2.svg?invert_in_darkmode" align=middle width=138.1617303pt height=16.438356pt/></p>

As all the gradients for the whole data-set need to be calculated to perform one single update, batch gradient descent can be very slow. Furthermore it can be impracticable for larger data-sets that don't fit into working memory. 

### Stochastic gradient descent (SGD)

Stochastic gradient descent (SGD), on the other hand, performs a parameter update for each training example <img src="tex/ad769e751231d17313953f80471b27a4.svg?invert_in_darkmode" align=middle width=24.319919249999987pt height=29.190975000000005pt/> and label <img src="tex/708d9d53037c10f462707daa2370b7df.svg?invert_in_darkmode" align=middle width=23.57413739999999pt height=29.190975000000005pt/>.

<p align="center"><img src="tex/d905c0dba806bdd8413af4aefb15d0be.svg?invert_in_darkmode" align=middle width=202.31132954999998pt height=19.526994300000002pt/></p>

Batch gradient descent performs redundant gradient computations as it recomputes gradients for similar examples before performing a parameter update. SGD doesn't have the same redundancy as it updates for each training example, which is why it's usually much faster than batch gradient descend. Furthermore, it can also be used for online learning.

Performing a parameter update with only one training example can lead to high variance, which can cause the objective function to fluctuate.

### Mini-batch gradient descent

Mini-batch gradient descent combines the advantages of SGD and batch gradient descent by performing parameter updates for every mini-batch of <img src="tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/> training examples.

<p align="center"><img src="tex/1c5aa1876430bbdf7dcd8f9e641ac830.svg?invert_in_darkmode" align=middle width=255.49082625pt height=19.526994300000002pt/></p>

By performing the parameter update on a mini-batch, it **a)** reduces the variance of the update, which can lead to more stable convergence, and **b)** can make use of highly optimized matrix calculates commonly found in state-of-the-art deep learning libraries.

The mini-batch size usually ranges between 16 and 256 depending on the application and the training hardware. Mini-batch gradient descent is typically the algorithm of choice from the three ones discussed above.

![Variations comparison](doc/variations_comparison.png)

*Source: [Understanding Optimization Algorithms](https://laptrinhx.com/understanding-optimization-algorithms-3818430905/)*

## Challenges

All three of the above-mentioned types of Gradient descent have a few challenges that need to be addressed:

- Choosing a proper learning rate can be difficult.
- The learning rate is the same throughout training.
- The same learning rate is applied to all parameter updates.
- Gradient Descent is a first-order optimization algorithm. Meaning that it only takes into account the first derivative of the objective function and not the second derivative, even though the curvature of the objective function also affects the size of each learning step.

## Momentum

Momentum is a method that helps accelerate SGD in the relevant direction and dampens oscillations by adding a fraction <img src="tex/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode" align=middle width=9.423880949999988pt height=14.15524440000002pt/> of the last update vector <img src="tex/bec0f956437138a98cb909f5dae6b77f.svg?invert_in_darkmode" align=middle width=29.76042629999999pt height=14.15524440000002pt/> to the current update vector.

<p align="center"><img src="tex/f6ba11db1e6b10797a9ebcc12aeda2dc.svg?invert_in_darkmode" align=middle width=246.86632079999998pt height=16.438356pt/></p>

The momentum term increases for dimensions where gradients continuously point in the same directions and reduces the updates for dimensions whose gradients change directions from one time step to another. As a result, convergence is faster, and oscillation is reduced. 

![Momentum](doc/momentum.png)

*Source: [SGD with Momentum](https://paperswithcode.com/method/sgd-with-momentum)*

## Nesterov accelerated gradient

Nesterov accelerated gradient (NAG), also called Nesterov Momentum is a variation of momentum that approximates the next values of the parameters by computing <img src="tex/666d1825fe38f52f9b0a01c2721dc4c8.svg?invert_in_darkmode" align=middle width=67.44900689999999pt height=22.831056599999986pt/> and then takes the gradient of the objective funtion not w.r.t the current parameters <img src="tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/> but w.r.t the approximate future parameters <img src="tex/666d1825fe38f52f9b0a01c2721dc4c8.svg?invert_in_darkmode" align=middle width=67.44900689999999pt height=22.831056599999986pt/>. 

<p align="center"><img src="tex/19f7986adf26d94218ca0cb10277f8e4.svg?invert_in_darkmode" align=middle width=306.9636669pt height=16.438356pt/></p>

Momentum first computes the current gradient (small blue vector) and then takes a big jump in the direction of the updated accumulated gradient (big blue vector). NAG, on the other hand, first makes a big jump in the direction of the previous accumulated gradient (brown vector), measures the gradient, and then makes a correction (red vector), which results in the green vector. The anticipation prevents the update from overshooting and results in increased responsiveness. 

![Nesterov accelerated gradient](doc/nesterov_accelerated_gradient.png)

*Source: [G. Hinton's lecture 6c](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)*

## Code

- [Gradient Descent with momentum](code/gradient_descent_with_momentum.py)
- [Gradient descent with nesterov momentum](code/gradient_descent_with_nesterov_momentum.py)

## Resources

- [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)
- [Gradient Descent ML Glossary](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)
- [Gradient Descent Algorithm and Its Variants](https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3)
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8)
- [Stochastic Gradient Descent, Clearly Explained!!!](https://www.youtube.com/watch?v=vMh0zPT0tLI)
- [Gradient descent, how neural networks learn | Deep learning, chapter 2](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- [Lecture 6a Overview of mini-batch gradient descent](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)