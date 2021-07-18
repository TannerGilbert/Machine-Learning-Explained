# Adaptive Moment Estimation (Adam)

Adaptive Moment Estimation better known as Adam is another adaptive learning rate method first published in 2014 by Kingma et. al. <a href="#citation1">[1]</a> In addition to storing an exponentially decaying average of past squared gradients <img src="tex/3e3c6ee78813607a4d976d92c19dd36e.svg?invert_in_darkmode" align=middle width=12.93385829999999pt height=14.15524440000002pt/> like Adadelta or RMSprop, Adam also keeps an exponentially decaying average of past gradients <img src="tex/ddb44cc6d9b5fa907d7e2d60daed1bca.svg?invert_in_darkmode" align=middle width=19.398893249999993pt height=14.15524440000002pt/>, similar to SGD with momentum. <a href="#citation2">[2]</a>

<p align="center"><img src="tex/b65d13242f56b3410177b1401dd8b7e8.svg?invert_in_darkmode" align=middle width=186.52399425pt height=16.438356pt/></p>

<p align="center"><img src="tex/824123b152beebd863c67856d33ed802.svg?invert_in_darkmode" align=middle width=175.77045585pt height=18.312383099999998pt/></p>

<img src="tex/ddb44cc6d9b5fa907d7e2d60daed1bca.svg?invert_in_darkmode" align=middle width=19.398893249999993pt height=14.15524440000002pt/> is an estimate of the first [moment](https://en.wikipedia.org/wiki/Moment_(mathematics)) (the mean) and <img src="tex/3e3c6ee78813607a4d976d92c19dd36e.svg?invert_in_darkmode" align=middle width=12.93385829999999pt height=14.15524440000002pt/> is the estimate of the second moment (the uncentered variance) of the gradients respectively. As <img src="tex/ddb44cc6d9b5fa907d7e2d60daed1bca.svg?invert_in_darkmode" align=middle width=19.398893249999993pt height=14.15524440000002pt/> and <img src="tex/3e3c6ee78813607a4d976d92c19dd36e.svg?invert_in_darkmode" align=middle width=12.93385829999999pt height=14.15524440000002pt/> are initialized as vectors of 0's, the authors of Adam observe that they are biased towards zero, especially during the initial time steps, and especially when the decay rates are small (i.e. <img src="tex/15ef3b23ef739e47090fa0825bf9d390.svg?invert_in_darkmode" align=middle width=15.85051049999999pt height=22.831056599999986pt/> and <img src="tex/2cae3bbfffb6ab2858054ba28bfcba80.svg?invert_in_darkmode" align=middle width=15.85051049999999pt height=22.831056599999986pt/> are close to 1). <a href="#citation2">[2]</a>

To counteract the biases by calculating bias-corrected first and second moment esimates:

<p align="center"><img src="tex/f4bee786ed43433221a48b27a5ed87ec.svg?invert_in_darkmode" align=middle width=89.0938092pt height=33.85762545pt/></p>

<p align="center"><img src="tex/4ea6f1054f33b2fe4ccc258e940fdce1.svg?invert_in_darkmode" align=middle width=82.62875774999999pt height=33.85762545pt/></p>

<img src="tex/285dbe2a851d6e35501b39511115cd05.svg?invert_in_darkmode" align=middle width=19.398893249999993pt height=22.831056599999986pt/> and <img src="tex/f24bd5b399fcd2f1620d8978d4c3d069.svg?invert_in_darkmode" align=middle width=12.93385829999999pt height=22.831056599999986pt/> are then used to update the parameters as follows:

<p align="center"><img src="tex/2feec3f6a85bfa367ca19d5e6d7002e6.svg?invert_in_darkmode" align=middle width=163.22396145pt height=33.4857765pt/></p>

As default values for <img src="tex/15ef3b23ef739e47090fa0825bf9d390.svg?invert_in_darkmode" align=middle width=15.85051049999999pt height=22.831056599999986pt/> and <img src="tex/2cae3bbfffb6ab2858054ba28bfcba80.svg?invert_in_darkmode" align=middle width=15.85051049999999pt height=22.831056599999986pt/> the authors propose <img src="tex/1c22e0ed21fd53f1f1d04d22d5d21677.svg?invert_in_darkmode" align=middle width=21.00464354999999pt height=21.18721440000001pt/> for <img src="tex/15ef3b23ef739e47090fa0825bf9d390.svg?invert_in_darkmode" align=middle width=15.85051049999999pt height=22.831056599999986pt/> and <img src="tex/a53a375441275f24641fc239deb138cb.svg?invert_in_darkmode" align=middle width=37.44306224999999pt height=21.18721440000001pt/> for <img src="tex/2cae3bbfffb6ab2858054ba28bfcba80.svg?invert_in_darkmode" align=middle width=15.85051049999999pt height=22.831056599999986pt/>.

<p id="citation1">[1] Diederik P. Kingma and Jimmy Ba (2014). Adam: A Method for Stochastic Optimization.</p>

<p id="citation2">[2] Sebastian Ruder (2016). An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747.</p>

## Code

- [Adam Numpy Implementation](code/adam.py)

## Resources

- [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
- [https://ruder.io/optimizing-gradient-descent/index.html#adam](https://ruder.io/optimizing-gradient-descent/index.html#adam)
- [https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c)