# Kernel PCA

![Kernel PCA Example](doc/kernel_pca.png)

Kernel PCA is an extension of [PCA](https://ml-explained.com/blog/principal-component-analysis-explained) that allows for the separability of nonlinear data by making use of kernels. The basic idea behind it is to project the linearly inseparable data onto a higher dimensional space where it becomes linearly separable. 

Kernel PCA can be summarized as a 4 step process [<a href="#citation1">1</a>]:

1. Construct the kernel matrix <img src="tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode" align=middle width=15.13700594999999pt height=22.465723500000017pt/> from the training dataset

<p align="center"><img src="tex/88a947ead8011f566945b4f207fde1a8.svg?invert_in_darkmode" align=middle width=111.28551555pt height=17.031940199999998pt/></p>

2. If the projected dataset <img src="tex/1cef6a8d14b34297d97d3e1cf812ff5c.svg?invert_in_darkmode" align=middle width=54.46830179999999pt height=24.65753399999998pt/> doesn’t have zero mean use the Gram matrix <img src="tex/3821cc82b4d7dc6624ec03fd5a93dffc.svg?invert_in_darkmode" align=middle width=15.13700594999999pt height=34.8663909pt/> to substitute the kernel matrix <img src="tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode" align=middle width=15.13700594999999pt height=22.465723500000017pt/>.

<p align="center"><img src="tex/12e6d8a64abd9854079af8b0622eb86a.svg?invert_in_darkmode" align=middle width=239.70255374999996pt height=19.8989241pt/></p>

3. Use <img src="tex/87524c1390370d418a3be6af1b4136c5.svg?invert_in_darkmode" align=middle width=95.39667554999998pt height=22.831056599999986pt/> to solve for the vector <img src="tex/65ed4b231dcf18a70bae40e50d48c9c0.svg?invert_in_darkmode" align=middle width=13.340053649999989pt height=14.15524440000002pt/>.

4. Compute the kernel principal components <img src="tex/a6096ac2cee42d8fa76ec9110eb9c598.svg?invert_in_darkmode" align=middle width=41.06745224999999pt height=24.65753399999998pt/>

<p align="center"><img src="tex/28a3c6f9dc75c8bf1b3498bbcea108be.svg?invert_in_darkmode" align=middle width=257.03949975pt height=47.806078649999996pt/></p>

<p id="citation1">[1] <a href="https://arxiv.org/pdf/1207.3538.pdf">Kernel Principal Component Analysis and its Applications in Face Recognition and Active Shape Models</a></p>

## Resources

- [Kernel Principal Component Analysis and its Applications in Face Recognition and Active Shape Models](https://arxiv.org/pdf/1207.3538.pdf)
- [Kernel tricks and nonlinear dimensionality reduction via RBF kernel PCA](https://sebastianraschka.com/Articles/2014_kernel_pca.html)
- [PCA and kernel PCA explained](https://nirpyresearch.com/pca-kernel-pca-explained/)
- [What are the advantages of kernel PCA over standard PCA?](https://stats.stackexchange.com/questions/94463/what-are-the-advantages-of-kernel-pca-over-standard-pca)