# QHM (Quasi-Hyperbolic Momentum)

![QHM Update rule](doc/qhm_update_rule.PNG)

Quasi-Hyperbolic Momentum Algorithm (QHM) is a simple alteration of SGD with momentum, averaging a plain SGD step with a momentum step, thereby decoupling the momentum term $\beta$ from the current gradient $\nabla_t$ when updating the weights.

$$g_{t + 1} \leftarrow \beta \cdot g_t + (1 - \beta) \cdot \nabla_t$$

$$\theta_{t + 1} \leftarrow \theta_t + \alpha \left[ (1 - \nu) \cdot \nabla_t + \nu \cdot g_{t + 1} \right]$$

The authors recommend $\nu=0.7$ and $\beta=0.999$ as a good starting point. For more information about QHM, check out the resources below.

## Code

- [QHM Numpy Implementation](code/qhm.py)

## Resources

- [https://arxiv.org/pdf/1810.06801.pdf](https://arxiv.org/pdf/1810.06801.pdf)
- [https://paperswithcode.com/method/qhadam](https://paperswithcode.com/method/qhadam)
- [https://johnchenresearch.github.io/demon/](https://johnchenresearch.github.io/demon/)
- [https://facebookresearch.github.io/qhoptim/](https://facebookresearch.github.io/qhoptim/)
