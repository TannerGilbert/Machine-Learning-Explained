# Kernel PCA

![Kernel PCA Example](doc/kernel_pca.png)

Kernel PCA is an extension of [PCA](https://ml-explained.com/blog/principal-component-analysis-explained) that allows for the separability of nonlinear data by making use of kernels. The basic idea behind it is to project the linearly inseparable data onto a higher dimensional space where it becomes linearly separable.

Kernel PCA can be summarized as a 4 step process [<a href="#citation1">1</a>]:

1. Construct the kernel matrix $K$ from the training dataset

$$K_{i,j} = \kappa(\mathbf{x_i, x_j})$$

2. If the projected dataset $\left\lbrace \phi (\mathbf{x}_i) \right\rbrace$ doesn’t have zero mean use the Gram matrix $\stackrel{\sim}{K}$ to substitute the kernel matrix $K$.

$$\stackrel{\sim}{K} = K - \mathbf{1_N} K - K \mathbf{1_N} + \mathbf{1_N} K \mathbf{1_N}$$

3. Use $K_{a_k} = \lambda_k N_{a_{k}}$ to solve for the vector $a_i$.

4. Compute the kernel principal components $y_k\left(x\right)$

$$y_k(\mathbf{x})= \phi \left(\mathbf{x}\right)^T \mathbf{v}_k =  \sum_{i=1}^N a_{ki} \kappa(\mathbf{x_i, x_j})$$

<p id="citation1">[1] <a href="https://arxiv.org/pdf/1207.3538.pdf">Kernel Principal Component Analysis and its Applications in Face Recognition and Active Shape Models</a></p>

## Resources

- [Kernel Principal Component Analysis and its Applications in Face Recognition and Active Shape Models](https://arxiv.org/pdf/1207.3538.pdf)
- [Kernel tricks and nonlinear dimensionality reduction via RBF kernel PCA](https://sebastianraschka.com/Articles/2014_kernel_pca.html)
- [PCA and kernel PCA explained](https://nirpyresearch.com/pca-kernel-pca-explained/)
- [What are the advantages of kernel PCA over standard PCA?](https://stats.stackexchange.com/questions/94463/what-are-the-advantages-of-kernel-pca-over-standard-pca)
