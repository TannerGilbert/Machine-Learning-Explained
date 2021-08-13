# QHAdam (Quasi-Hyperbolic Adam)

![QHAdam Example](doc/qhadam_example.png)

**Quasi-Hyperbolic Momentum Algorithm (QHM)** is a simple alteration of [SGD with momentum](https://paperswithcode.com/method/sgd-with-momentum), averaging a plain SGD step with a momentum step. **QHAdam (Quasi-Hyperbolic Adam)** is a QH augmented version of [Adam](https://ml-explained.com/blog/adam-explained) that replaces both of Adam's moment estimators with quasi-hyperbolic terms. Namely, QHAdam decouples the momentum term from the current gradient when updating the weights, and decouples the mean squared gradients term from the current squared gradient when updating the weights. [<a href="#citation1">1</a>, <a href="#citation2">2</a>, <a href="#citation3">3</a>]

Essentially, it's a weighted average of the momentum and plain SGD, weighting the current gradient with an immediate discount factor $v_1$ divided by a weighted average of the mean squared gradients and the current squared gradient, weighting the current squared gradient with an immediate discount factor $v_2$. <a href="#citation2">[2]</a>

$$ \theta_{t+1, i} = \theta_{t, i} - \eta\left[\frac{\left(1-v_{1}\right)\cdot{g_{t}} + v_{1}\cdot\hat{m}_{t}}{\sqrt{\left(1-v_{2}\right)g^{2}_{t} + v_{2}\cdot{\hat{v}_{t}}} + \epsilon}\right], \forall{t} $$

<p id="citation1">[1] Ma, J. and Yarats, D. Quasi-hyperbolic momentum and Adam for deep learning. arXiv preprint arXiv:1810.06801, 2018</p>

<p id="citation2">[2] <a href="https://paperswithcode.com/method/qhadam">QHAdam Papers With Code</a></p>

<p id="citation3">[3] John Chen. <a href="https://johnchenresearch.github.io/demon/">An updated overview of recent gradient descent algorithms</a></p>