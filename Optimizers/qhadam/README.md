# QHAdam (Quasi-Hyperbolic Adam)

![QHAdam Example](doc/qhadam_example.png)

**Quasi-Hyperbolic Momentum Algorithm (QHM)** is a simple alteration of [SGD with momentum](https://paperswithcode.com/method/sgd-with-momentum), averaging a plain SGD step with a momentum step. **QHAdam (Quasi-Hyperbolic Adam)** is a QH augmented version of [Adam](https://ml-explained.com/blog/adam-explained) that replaces both of Adam's moment estimators with quasi-hyperbolic terms. Namely, QHAdam decouples the momentum term from the current gradient when updating the weights, and decouples the mean squared gradients term from the current squared gradient when updating the weights. [<a href="#citation1">1</a>, <a href="#citation2">2</a>, <a href="#citation3">3</a>]

Essentially, it's a weighted average of the momentum and plain SGD, weighting the current gradient with an immediate discount factor <img src="tex/41922e474070adc90e7c1379c28d22fe.svg?invert_in_darkmode" align=middle width=14.520613799999989pt height=14.15524440000002pt/> divided by a weighted average of the mean squared gradients and the current squared gradient, weighting the current squared gradient with an immediate discount factor <img src="tex/53292819177dbb29ba6d92fe3aa2880c.svg?invert_in_darkmode" align=middle width=14.520613799999989pt height=14.15524440000002pt/>. <a href="#citation2">[2]</a>

<p align="center"><img src="tex/bcf57c8141818aa66812cefcf9d1a886.svg?invert_in_darkmode" align=middle width=341.74707105pt height=49.315569599999996pt/></p>

<p id="citation1">[1] Ma, J. and Yarats, D. Quasi-hyperbolic momentum and Adam for deep learning. arXiv preprint arXiv:1810.06801, 2018</p>

<p id="citation2">[2] <a href="https://paperswithcode.com/method/qhadam">QHAdam Papers With Code</a></p>

<p id="citation3">[3] John Chen. <a href="https://johnchenresearch.github.io/demon/">An updated overview of recent gradient descent algorithms</a></p>

## Code

- [QHAdam Numpy Implementation](code/qhadam.py)