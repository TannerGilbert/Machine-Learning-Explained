# Adagrad 

![Adagrad example gif](doc/adagrad_example.gif)

Adagrad <a href="#citation1">[1]</a> is a gradient-based optimization algorithm that adaptively scales the learning rate to the parameters, performing smaller updates for parameters associated with frequently occurring features and larger updates for parameters associated with infrequent features eliminating the need to tune the learning rate manually. The above-mentioned behavior makes Adagrad well-suited for dealing with sparse data, and Dean et al. <a href="#citation2">[2]</a> have found out that Adagrad is much more robust than SGD.

Reminder: The SDG update for each parameter <img src="tex/f166369f3ef0a7ff052f1e9bbf57d2e2.svg?invert_in_darkmode" align=middle width=12.36779114999999pt height=22.831056599999986pt/> look as follows:

<p align="center"><img src="tex/45913b7ee3a34648c53cb1db66c97d75.svg?invert_in_darkmode" align=middle width=191.897541pt height=17.031940199999998pt/></p>

To scale the learning rate to each parameter Adagrad modifies the learning rate <img src="tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.57650494999999pt height=14.15524440000002pt/> at each time step <img src="tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/> for every parameter <img src="tex/f166369f3ef0a7ff052f1e9bbf57d2e2.svg?invert_in_darkmode" align=middle width=12.36779114999999pt height=22.831056599999986pt/> based on the past gradients of <img src="tex/f166369f3ef0a7ff052f1e9bbf57d2e2.svg?invert_in_darkmode" align=middle width=12.36779114999999pt height=22.831056599999986pt/>:

<p align="center"><img src="tex/ad3e2cec2e4e99bcb40a19ecda561e56.svg?invert_in_darkmode" align=middle width=260.38638615pt height=36.773649pt/></p>

Here <img src="tex/db14eb9fda4448bde6e9d57897df8aae.svg?invert_in_darkmode" align=middle width=74.63582609999999pt height=27.91243950000002pt/> is a diagonal matrix where each diagonal element <img src="tex/43af08929a34d369038ea5f29d4b9cad.svg?invert_in_darkmode" align=middle width=18.63233624999999pt height=21.68300969999999pt/> is the sum of the squares of the gradients w.r.t. <img src="tex/f166369f3ef0a7ff052f1e9bbf57d2e2.svg?invert_in_darkmode" align=middle width=12.36779114999999pt height=22.831056599999986pt/> up to time step <img src="tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/> and <img src="tex/7ccca27b5ccc533a2dd72dc6fa28ed84.svg?invert_in_darkmode" align=middle width=6.672392099999992pt height=14.15524440000002pt/> is a smoothing term used to avoid division by zero.

The above can be vectorized as follows:

<p align="center"><img src="tex/b1cc9c4f3f1d62306a8d45977e8f2946.svg?invert_in_darkmode" align=middle width=235.55004164999997pt height=33.4857765pt/></p>

Adagrads most significant benefit is that it eliminates the need to tune the learning rate manually, but it still isn't perfect. Its main weakness is that it accumulates the squared gradients in the denominator. Since all the squared terms are positive, the accumulated sum keeps on growing during training. Therefore the learning rate keeps shrinking as the training continues, and it eventually becomes infinitely small. Other algorithms like Adadelta, RMSprop, and Adam try to resolve this flaw. <a href="#citation3">[3]</a>

<p id="citation1">[1] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. Journal of Machine Learning Research, 12, 2121–2159. Retrieved from [http://jmlr.org/papers/v12/duchi11a.html](http://jml
<p id="citation2">[2] Dean, J., Corrado, G. S., Monga, R., Chen, K., Devin, M., Le, Q. V, … Ng, A. Y. (2012). Large Scale Distributed Deep Networks. NIPS 2012: Neural Information Processing Systems, 1–11. [http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf)</p>
<p id="citation3">[3] Sebastian Ruder (2016). An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747.</p>

## Code

- [Adagrad Numpy Implementation](code/adagrad.py)