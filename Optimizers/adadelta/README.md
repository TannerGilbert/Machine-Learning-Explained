# ADADELTA: An Adaptive Learning Rate Method

![Adadelta Example](doc/adadelta_example.png)

Adadelta is a stochastic gradient-based optimization algorithm that allows for per-dimension learning rates. Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to a fixed size <img src="tex/ae4fb5973f393577570881fc24fc2054.svg?invert_in_darkmode" align=middle width=10.82192594999999pt height=14.15524440000002pt/>. <a href="#citation1">[1]</a>

Instead of inefficiently storing <img src="tex/ae4fb5973f393577570881fc24fc2054.svg?invert_in_darkmode" align=middle width=10.82192594999999pt height=14.15524440000002pt/> previous squared gradients, the sum of gradients is recursively defined as a decaying average of all past squared gradients. The running average <img src="tex/16423efbb7c672354f022590a8f79ed2.svg?invert_in_darkmode" align=middle width=50.29113704999999pt height=27.94539330000001pt/> at time step <img src="tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/> therefore only depends on the previous average and the current gradient <a href="#citation2">[2]</a>:

<p align="center"><img src="tex/9d55fd72b8efdeca23093c2ed0ea5745.svg?invert_in_darkmode" align=middle width=238.72752749999998pt height=22.14809025pt/></p>

<img src="tex/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode" align=middle width=9.423880949999988pt height=14.15524440000002pt/> is usually set to around 0.9. Rewriting SGD updates in terms of the parameter update vector:

<p align="center"><img src="tex/3ca6889677ea09e526a816322160498f.svg?invert_in_darkmode" align=middle width=103.89164114999998pt height=16.1187015pt/></p>

<p align="center"><img src="tex/8fdfd1eb52433d071078828592da25cc.svg?invert_in_darkmode" align=middle width=117.5226921pt height=15.251136449999997pt/></p>

AdaDelta takes the form:

<p align="center"><img src="tex/407764bb35619057e9230a563546d02a.svg?invert_in_darkmode" align=middle width=176.39243985pt height=29.58934275pt/></p>

<p align="center"><img src="tex/20aafbd370a6b88bfacab3c7c49d8aa8.svg?invert_in_darkmode" align=middle width=143.38391265pt height=33.58376834999999pt/></p>

The authors that the units in the weight update don't match, i.e., the update should have the same hypothetical units as the parameters/weights. To realize this, they use the root mean squared error of parameter updates.

<p align="center"><img src="tex/8618e3e0464e1c4ae3ba41984874fa33.svg?invert_in_darkmode" align=middle width=261.70160775pt height=18.312383099999998pt/></p>

<p align="center"><img src="tex/4764741b6721dc727ba86e4c3ea5d106.svg?invert_in_darkmode" align=middle width=200.5364691pt height=19.726228499999998pt/></p>

Since <img src="tex/b999b985f5ccf08b3fce39e97a1c63b8.svg?invert_in_darkmode" align=middle width=77.34600719999999pt height=24.65753399999998pt/>  is unknown, it's approximated with the RMS of the parameter updates until the previous time step <img src="tex/e7d2063bdcfb3dfdb3f44724950543d1.svg?invert_in_darkmode" align=middle width=94.17257519999998pt height=24.65753399999998pt/>.

<p align="center"><img src="tex/15431539b7b73e500cc0fd3d7e0af147.svg?invert_in_darkmode" align=middle width=176.5975398pt height=63.05404875pt/></p>

For more information on how to derive this formula, take a look at '[An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/index.html#adadelta)' by [Sebastian Ruder](https://twitter.com/seb_ruder) and the [original Adadelta paper](https://arxiv.org/abs/1212.5701) by [Matthew D. Zeiler](https://arxiv.org/search/cs?searchtype=author&query=Zeiler%2C+M+D).

Adadelta's main advantages over Adagrad are that it doesn't need a default learning rate and that it doesn't decrease the learning rate as aggressively and monotonically as Adagrad. 

<p id="citation1">[1] Sebastian Ruder (2016). An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747.</p>

<p id="citation2">[2] <a href="https://paperswithcode.com/method/adadelta">https://paperswithcode.com/method/adadelta</a></p>

## Code

* [Adadelta Numpy Implementation](code/adadelta.py)