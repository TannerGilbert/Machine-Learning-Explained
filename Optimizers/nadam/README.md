# Nadam (Nesterov-accelerated Adaptive Moment Estimation)

![Nadam Training Example](doc/nadam_example.png)

Nadam (Nesterov-accelerated Adaptive Moment Estimation) combines NAG (Nesterov accelerated gradient) and Adam. To do so, the momentum term <img src="tex/ddb44cc6d9b5fa907d7e2d60daed1bca.svg?invert_in_darkmode" align=middle width=19.398893249999993pt height=14.15524440000002pt/> needs to be updated. For more information, check out [the paper](http://cs229.stanford.edu/proj2015/054_report.pdf) or the [Nadam section](https://ruder.io/optimizing-gradient-descent/index.html#nadam) of ['An overview of gradient descent optimization algorithms'](https://ruder.io/optimizing-gradient-descent/index.html).

The final update rule looks as follows:

<p align="center"><img src="tex/b860e63de84df769d7d9d6ce9295ba65.svg?invert_in_darkmode" align=middle width=288.9365853pt height=39.10877025pt/></p>

## Code

- [Nadam Numpy Implementation](code/nadam.py)

## Resources

- [http://cs229.stanford.edu/proj2015/054_report.pdf](http://cs229.stanford.edu/proj2015/054_report.pdf)
- [https://paperswithcode.com/method/nadam](https://paperswithcode.com/method/nadam)
- [https://ruder.io/optimizing-gradient-descent/index.html#nadam](https://ruder.io/optimizing-gradient-descent/index.html#nadam)