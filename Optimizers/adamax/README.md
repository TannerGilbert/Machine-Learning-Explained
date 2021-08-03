# AdaMax

![AdaMax Example](doc/adamax_example.PNG)

In [Adam](https://ml-explained.com/blog/adam-explained), the update rule for individual weights is scaling their gradients inversely proportional to the <img src="tex/336fefe2418749fabf50594e52f7b776.svg?invert_in_darkmode" align=middle width=13.40191379999999pt height=22.831056599999986pt/> norm of the past and current gradients.

<p align="center"><img src="tex/6859140733d250349cb7e3623130b8d7.svg?invert_in_darkmode" align=middle width=190.10081639999999pt height=18.312383099999998pt/></p>

The L2 norm can be generalized to the <img src="tex/ca185a0f63add2baa6fe729fd1cfef60.svg?invert_in_darkmode" align=middle width=13.625845199999988pt height=22.831056599999986pt/> norm.

<p align="center"><img src="tex/34ec2fa234397799e854fa7109da32c2.svg?invert_in_darkmode" align=middle width=192.50771594999998pt height=17.2372761pt/></p>

Such variants generally become numerically unstable for large <img src="tex/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode" align=middle width=8.270567249999992pt height=14.15524440000002pt/>, which is why <img src="tex/839a0dc412c4f8670dd1064e0d6d412f.svg?invert_in_darkmode" align=middle width=13.40191379999999pt height=22.831056599999986pt/> and <img src="tex/336fefe2418749fabf50594e52f7b776.svg?invert_in_darkmode" align=middle width=13.40191379999999pt height=22.831056599999986pt/> norms are most common in practice. However, in the special case where we let <img src="tex/5f5bf3f4ba1dd968b4cf5449b4310370.svg?invert_in_darkmode" align=middle width=50.27957054999999pt height=14.15524440000002pt/>, a surprisingly simple and stable algorithm emerges.

To avoid confusion with Adam, we use <img src="tex/e6897b8647f3bd38144535d3f40078e2.svg?invert_in_darkmode" align=middle width=14.37606554999999pt height=14.15524440000002pt/> to denote the infinity norm-constrained <img src="tex/3e3c6ee78813607a4d976d92c19dd36e.svg?invert_in_darkmode" align=middle width=12.93385829999999pt height=14.15524440000002pt/>:

<p align="center"><img src="tex/485b078316d575b8a3edd55921040580.svg?invert_in_darkmode" align=middle width=368.2477029pt height=16.438356pt/></p>

We can now plug <img src="tex/e6897b8647f3bd38144535d3f40078e2.svg?invert_in_darkmode" align=middle width=14.37606554999999pt height=14.15524440000002pt/> into the Adam update equation replacing <img src="tex/c8a984d1a187544cc1d3132786b791b3.svg?invert_in_darkmode" align=middle width=54.21799019999999pt height=26.867530799999987pt/> to obtain the AdaMax update rule:

<p align="center"><img src="tex/c88595da993fcae459ef526daedd66d7.svg?invert_in_darkmode" align=middle width=124.20393479999998pt height=31.939908pt/></p>

## Code

- [AdaMax Numpy Implementation](code/adamax.py)

## Resources

- [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
- [https://ruder.io/optimizing-gradient-descent/index.html#adamax](https://ruder.io/optimizing-gradient-descent/index.html#adamax)
- [https://keras.io/api/optimizers/adamax/](https://keras.io/api/optimizers/adamax/)