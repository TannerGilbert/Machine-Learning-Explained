# RMSprop

![RMSprop Example](doc/rmsprop_example.PNG)

RMSprop is an unpublished, adaptive learning rate optimization algorithm first proposed by [Geoff Hinton](https://en.wikipedia.org/wiki/Geoffrey_Hinton) in lecture 6 of his online class "[Neural Networks for Machine Learning](http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf)". RMSprop and Adadelta have been developed independently around the same time, and both try to resolve Adagrad's diminishing learning rate problem. <a href="#citation1">[1]</a>

<p align="center"><img src="tex/4d0c5cb8a4df5487f9457948069c0c86.svg?invert_in_darkmode" align=middle width=203.4418122pt height=18.312383099999998pt/></p>
<p align="center"><img src="tex/fffcaba90180362da033429f55895e5a.svg?invert_in_darkmode" align=middle width=189.42252285pt height=36.773649pt/></p>

The difference between Adadelta and RMSprop is that Adadelta removes the learning rate <img src="tex/1d0496971a2775f4887d1df25cea4f7e.svg?invert_in_darkmode" align=middle width=8.751954749999989pt height=14.15524440000002pt/> entirely and replaces it by the root mean squared error of parameter updates.

<p id="citation1">[1] Sebastian Ruder (2016). An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747.</p>

## Code

* [RMSprop Numpy Implementation](code/rmsprop.py)