# ADADELTA: An Adaptive Learning Rate Method

![Adadelta Example](doc/adadelta_example.png)

Adadelta is a stochastic gradient-based optimization algorithm that allows for per-dimension learning rates. Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to a fixed size $\omega$. <a href="#citation1">[1]</a>

Instead of inefficiently storing $\omega$ previous squared gradients, the sum of gradients is recursively defined as a decaying average of all past squared gradients. The running average $E\left[g^{2}\right]_{t}$ at time step $t$ therefore only depends on the previous average and the current gradient <a href="#citation2">[2]</a>:

$$E\left[g^{2}\right]_{t} = \gamma{E}\left[g^{2}\right]_{t-1} + \left(1-\gamma\right)g^{2}_{t}$$

$\gamma$ is usually set to around 0.9. Rewriting SGD updates in terms of the parameter update vector:

$$ \Delta\theta_{t} = -\eta\cdot{g_{t, i}}$$

$$\theta_{t+1}  = \theta_{t} + \Delta\theta_{t}$$

AdaDelta takes the form:

$$ \Delta\theta_{t} = -\frac{\eta}{\sqrt{E\left[g^{2}\right]_{t} + \epsilon}}g_{t} $$

For more information on how to derive this formula, take a look at '[An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/index.html#adadelta)' by [Sebastian Ruder](https://twitter.com/seb_ruder) and the [original Adadelta paper](https://arxiv.org/abs/1212.5701) by [Matthew D. Zeiler](https://arxiv.org/search/cs?searchtype=author&query=Zeiler%2C+M+D).

Adadelta's main advantages over Adagrad are that it doesn't need a default learning rate and that it doesn't decrease the learning rate as aggressively and monotonically as Adagrad. 

<p id="citation1">[1] Sebastian Ruder (2016). An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747.</p>

<p id="citation2">[2] <a href="https://paperswithcode.com/method/adadelta">https://paperswithcode.com/method/adadelta</a></p>