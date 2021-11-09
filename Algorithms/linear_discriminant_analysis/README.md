# Linear Discriminant Analysis (LDA)

![LDA Example](doc/lda_example.png)

Linear Discriminant Analysis (LDA) is a dimensionality reduction technique commonly used for supervised classification problems. The goal of LDA is to project the dataset onto a lower-dimensional space while maximizing the class separability.

LDA is very similar to Principal Component Analysis (PCA), but there are some important differences. PCA is an unsupervised algorithm, meaning it doesn't need class labels <img src="tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.649225749999989pt height=14.15524440000002pt/>. PCA's goal is to find the principal components that maximize the variance in a dataset. LDA, on the other hand, is a supervised algorithm, which uses both the input data <img src="tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=22.465723500000017pt/> and the class labels <img src="tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.649225749999989pt height=14.15524440000002pt/> to find linear discriminants that maximize the separation between multiple classes.

LDA can be performed in 5 steps:
1. Compute the mean vectors for the different classes from the dataset.
2. Compute the scatter matrices (in-between-class and within-class scatter matrices).
3. Compute the eigenvectors and corresponding eigenvalues for the scatter matrices.
4. Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the largest eigenvalues.
5. Use this eigenvector matrix to transform the samples onto the new subspace.

## Computing the mean vectors

First, calculate the mean vectors for all classes inside the dataset.

<p align="center"><img src="tex/66a81133e5715952856e2a06741f4676.svg?invert_in_darkmode" align=middle width=104.3779407pt height=45.080067449999994pt/></p>

## Computing the scatter matrices

After calculating the mean vectors, the within-class and between-class scatter matrices can be calculated.

### Within-class scatter matrix <img src="tex/c7eee0782fa9ccb115b1518f68c8908f.svg?invert_in_darkmode" align=middle width=24.21242834999999pt height=22.465723500000017pt/>

<p align="center"><img src="tex/5a163b5cb124f209aed344b8f61b493f.svg?invert_in_darkmode" align=middle width=88.16675175pt height=44.89738935pt/></p>

where <img src="tex/d28140eda2d12e24b434e011b930fa23.svg?invert_in_darkmode" align=middle width=14.730823799999989pt height=22.465723500000017pt/> is the scatter matrix for a specific class

<p align="center"><img src="tex/6711c7bae84526c845527391cb33d2e5.svg?invert_in_darkmode" align=middle width=208.19363895pt height=46.790122499999995pt/></p>

and <img src="tex/47b592a798cd56ccf668b67abad36a61.svg?invert_in_darkmode" align=middle width=19.083998999999988pt height=14.15524440000002pt/> is the mean vector for that class

<p align="center"><img src="tex/66a81133e5715952856e2a06741f4676.svg?invert_in_darkmode" align=middle width=104.3779407pt height=45.080067449999994pt/></p>

Alternativeley the class-covariance matrices can be used by adding the scaling factor <img src="tex/fcda2be66b20dba76606c4f982b63b60.svg?invert_in_darkmode" align=middle width=28.4727069pt height=27.77565449999998pt/> to the within-class scatter matrix.

<p align="center"><img src="tex/874357dd0ff10af024f68c608dfc7a98.svg?invert_in_darkmode" align=middle width=263.66168235pt height=46.790122499999995pt/></p>

<p align="center"><img src="tex/021a2e6a7f973e9edb8dcb0bf5bda569.svg?invert_in_darkmode" align=middle width=146.99574614999997pt height=44.89738935pt/></p>

### Between-class scatter matrix <img src="tex/518542ce2a067b399803d0396d9c5aae.svg?invert_in_darkmode" align=middle width=20.572566299999988pt height=22.465723500000017pt/>

<p align="center"><img src="tex/61daf4e5401f3020b1b0bfefbbd0e59e.svg?invert_in_darkmode" align=middle width=232.4415951pt height=44.89738935pt/></p>

where <img src="tex/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode" align=middle width=14.433101099999991pt height=14.15524440000002pt/> is the overall mean, <img src="tex/47b592a798cd56ccf668b67abad36a61.svg?invert_in_darkmode" align=middle width=19.083998999999988pt height=14.15524440000002pt/> is the mean of the respective class, and <img src="tex/3bf9c1fe4273ed003fd49e744378a5ac.svg?invert_in_darkmode" align=middle width=17.85866609999999pt height=22.465723500000017pt/> is the sample size of that class.

## Calculate linear discriminants

Next, LDA solves the [generalized eigenvalue problem](https://arxiv.org/pdf/1903.11240.pdf) for the matrix <img src="tex/d8cf0d84a4e9973bace4607b359224f4.svg?invert_in_darkmode" align=middle width=49.24840634999998pt height=28.894955100000008pt/> to obtain the linear discriminants.

## Select linear discriminants for the new feature subspace

After calculating the eigenvectors and eigenvalues, we sort the eigenvectors from highest to lowest depending on their corresponding eigenvalue and then choose the top <img src="tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> eigenvectors, where <img src="tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> is the number of dimensions we want to keep. 

## Transform data onto the new subspace

After selecting the <img src="tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> eigenvectors, we can use the resulting <img src="tex/0aa7f58b7e561001f5301aa03507f552.svg?invert_in_darkmode" align=middle width=37.72252274999999pt height=22.831056599999986pt/>-dimensional eigenvector matrix <img src="tex/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.80826024999999pt height=22.465723500000017pt/> to transform data onto the new subspace via the following equation:

<p align="center"><img src="tex/a9ba65368f9892beab04bf21d7e17b4f.svg?invert_in_darkmode" align=middle width=87.92212934999999pt height=12.6027363pt/></p>

## Code

- [LDA Numpy Implementation](code/linear_discriminant_analysis.py)

## Credit / Resources

- [Linear Discriminant Analysis Bit by Bit](https://sebastianraschka.com/Articles/2014_python_lda.html)
- [StatQuest: Linear Discriminant Analysis (LDA) clearly explained.](https://www.youtube.com/watch?v=azXCzI57Yfc)
- [1.2. Linear and Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/lda_qda.html)
- [ML-From-Scratch MultiClassLDA](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/multi_class_lda.py)
- [mlxtend LDA](https://github.com/rasbt/mlxtend/blob/master/mlxtend/feature_extraction/linear_discriminant_analysis.py)