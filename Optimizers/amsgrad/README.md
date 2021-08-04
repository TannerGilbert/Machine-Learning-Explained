# AMSGrad

![AMSGrad Example](doc/amsgrad_example.png)

The motivation for AMSGrad lies with the observation that [Adam](https://ml-explained.com/blog/adam-explained) fails to converge to an optimal solution for some data-sets and is outperformed by SDG with momentum.

Reddi et al. (2018) [1] show that one cause of the issue described above is the use of the exponential moving average of the past squared gradients.

To fix the above-described behavior, the authors propose a new algorithm called AMSGrad that keeps a running maximum of the squared gradients instead of an exponential moving average.

<p align="center"><img src="tex/824123b152beebd863c67856d33ed802.svg?invert_in_darkmode" align=middle width=175.77045585pt height=18.312383099999998pt/></p>

<p align="center"><img src="tex/44e392b0bc182e02eec7fbcb32745a0a.svg?invert_in_darkmode" align=middle width=130.69642574999997pt height=16.438356pt/></p>

For simplicity, the authors also removed the debiasing step, which leads to the following update rule:

<p align="center"><img src="tex/d3f0f052c885b9de14f9b3438d1ba9f0.svg?invert_in_darkmode" align=middle width=196.45169775pt height=109.66126875pt/></p>

For more information, check out the paper '[On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237v1)' and the [AMSGrad section](https://ruder.io/optimizing-gradient-descent/index.html#amsgrad) of the '[An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/index.html)' article.

[1] Reddi, Sashank J., Kale, Satyen, & Kumar, Sanjiv. [On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237v1).

## Code

## Resources

- [https://arxiv.org/abs/1904.09237v1](https://arxiv.org/abs/1904.09237v1)
- [https://paperswithcode.com/method/amsgrad](https://paperswithcode.com/method/amsgrad)
- [https://ruder.io/optimizing-gradient-descent/index.html#amsgrad](https://ruder.io/optimizing-gradient-descent/index.html#amsgrad)