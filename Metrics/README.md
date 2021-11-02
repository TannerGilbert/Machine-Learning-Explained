# Metrics

![Confusion Matrix Example](doc/confusion_matrix.png)

## Classification

### Binary cross entropy

Binary cross entropy is a loss function used for binary classification tasks (tasks with only two outcomes/classes). It works by calculating the following average:

<p align="center"><img src="tex/894224f3dc1a64562c781eff86cad001.svg?invert_in_darkmode" align=middle width=578.6670021pt height=49.2398742pt/></p>

The above equation can be split into two parts to make it easier to understand:
<p align="center"><img src="tex/a92f489b7bf58458ad9a831191712560.svg?invert_in_darkmode" align=middle width=482.74484444999996pt height=100.03433549999998pt/></p>

![Binary Cross Entropy](doc/binary_cross_entropy.png)

The above graph shows that the further away the prediction is from the actual y value the bigger the loss gets.

That means that if the correct answer is 0, then the cost function will be 0 if the prediction is also 0. If the prediction approaches 1, then the cost function will approach infinity.

If the correct answer is 1, then the cost function will be 0 if the prediction is 1. If the prediction approaches 0, then the cost function will approach infinity.

Resources:

- [Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names](https://gombru.github.io/2018/05/23/cross_entropy_loss/#binary-cross-entropy-loss)
- [Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)
- [Understanding binary cross-entropy / log loss: a visual explanation](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)

Code:

- [Binary Cross Entropy Numpy Implementation](code/binary_cross_entropy.py)

### Categorical Crossentropy

Categorical crossentropy is a loss function used for multi-class classification tasks. The outputed loss is the negative average of the sum of the true values <img src="tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.649225749999989pt height=14.15524440000002pt/> multiplied by the log of the predicted values <img src="tex/0a5c2da8007e2edc6de9ca962be3f3ed.svg?invert_in_darkmode" align=middle width=33.32006039999999pt height=22.831056599999986pt/>.

<p align="center"><img src="tex/c0f72f6ec2f0d5623ef75e15d1a9f197.svg?invert_in_darkmode" align=middle width=301.37445855pt height=49.2398742pt/></p>

Resources:

- [Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names](https://gombru.github.io/2018/05/23/cross_entropy_loss/#losses)
- [Categorical crossentropy](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/categorical-crossentropy)

Code:

- [Categorical Cross Entropy Numpy Implementation](code/categorical_cross_entropy.py)

### Accuracy Score

The fraction of predictions the model classified correctly.

<p align="center"><img src="tex/e1a2df39f105072461870caf8fa0e344.svg?invert_in_darkmode" align=middle width=302.2927941pt height=37.0084374pt/></p>

or

<p align="center"><img src="tex/db850e0baa86d7832b5c75d7c4488d78.svg?invert_in_darkmode" align=middle width=320.50695269999994pt height=49.2398742pt/></p>

For binary classification, accuracy can also be calculated in terms of positives and negatives as follows:

<p align="center"><img src="tex/8a1f6bce1cca2d7cb34ee00ca6d18614.svg?invert_in_darkmode" align=middle width=239.00431994999997pt height=34.999293449999996pt/></p>

Where <img src="tex/86bbcafb36f7dfddde972e1b47296b4c.svg?invert_in_darkmode" align=middle width=146.80377524999997pt height=22.465723500000017pt/>, <img src="tex/202a192d4715ffd00cf289c10c107b43.svg?invert_in_darkmode" align=middle width=154.0184217pt height=22.465723500000017pt/>, <img src="tex/c821543a0ee6e81d1a637188ab98345e.svg?invert_in_darkmode" align=middle width=148.21934654999998pt height=22.831056599999986pt/>, and <img src="tex/d821640e564d2c34dbc9ee887fb60ca1.svg?invert_in_darkmode" align=middle width=155.433993pt height=22.831056599999986pt/>.

Resources:

- ['Classification: Accuracy' Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/classification/accuracy)
- [Accuracy Score Scikit Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score)
- [Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)

Code:

- [Accuracy Score Numpy Implementation](code/accuracy_score.py)

### Confusion matrix

A confusion matrix is a table that summarises the predictions of a classifier or classification model. By definition, entry <img src="tex/4fe48dde86ac2d37419f0b35d57ac460.svg?invert_in_darkmode" align=middle width=20.679527549999985pt height=21.68300969999999pt/> in a confusion matrix is the number of observations actually in group <img src="tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/>, but predicted to be in group <img src="tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.710416999999989pt height=21.68300969999999pt/>.

![Confusion matrix Example](https://cdn-images-1.medium.com/max/950/1*PPgItHcPSaskyjLMWFC-Kw.png)

Resources:

- [Confusion matrix Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix)
- [What is a Confusion Matrix in Machine Learning](https://machinelearningmastery.com/confusion-matrix-machine-learning/)
- [Simple guide to confusion matrix terminology](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)

### Hinge Loss

Hinge loss is a loss function usef for "maximum-margin" classification, most notably for Support Vector Machines (SVMs).

<p align="center"><img src="tex/e9999393d8d1b46365ba09586571c55d.svg?invert_in_darkmode" align=middle width=227.1765507pt height=17.031940199999998pt/></p>

Resources:

- [Hinge Loss Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#hinge-loss)
- [Hinge loss Wikipedia](https://en.wikipedia.org/wiki/Hinge_loss)
- [What is the definition of the hinge loss function?](https://ai.stackexchange.com/a/26336)

Code:

- [Hinge Loss Numpy Implementation](code/hinge.py)

### KL Divergence

The **Kullback-Leibler divergence**, <img src="tex/f1128d54a4a5ff0cc3a487dc3f920c62.svg?invert_in_darkmode" align=middle width=34.47958799999999pt height=22.465723500000017pt/>, often shortenend to just KL divergence, is a measure of how one probability distribution is different from a second, reference porbability distribution.

<p align="center"><img src="tex/15a86bf084c2654dfd8c0ab4ddda5bb3.svg?invert_in_darkmode" align=middle width=249.9011394pt height=49.2398742pt/></p>

Resources:

- [Kullback–Leibler divergence Wikipedia](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
- [Kullback-Leibler Divergence Explained](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)

Code:

- [KL Divergence Numpy Implementation](code/kl_divergence.py)

## Regression

### Mean Squared Error

The **mean squared error (MSE)** or **mean squared deviation (MSD)** measure the average of the squares of the errors - that is, the average squared differences between the estimated and actual values.

<p align="center"><img src="tex/735371fbbd0b21c453edc23b25d47a60.svg?invert_in_darkmode" align=middle width=292.4476896pt height=49.2398742pt/></p>

Resources:

- [Mean squared error Wikipedia](https://en.wikipedia.org/wiki/Mean_squared_error)
- [Mean squared error Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error)
- [Machine learning: an introduction to mean squared error and regression lines](https://www.freecodecamp.org/news/machine-learning-mean-squared-error-regression-line-c7dde9a26b93/)

Code:

- [Mean Squared Error Numpy Implementation](code/mean_squared_error.py)

### Mean Squared Logarithmic Error

**Mean Squared Logarithmic Error (MSLE)** is an extension of [**Mean Squared Error (MSE)**](#mean-squared-error) often used when the target <img src="tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.649225749999989pt height=14.15524440000002pt/> has an exponential growth.

> Note: This metrics penalizes under-predictions greater than over-predictions.

<p align="center"><img src="tex/61e1a35fbe056f586e6a9dbc645eabb7.svg?invert_in_darkmode" align=middle width=441.49680795pt height=49.2398742pt/></p>

Code:

- [Mean Squared Logarithmic Error Numpy Implementation](code/mean_squared_log_error.py)

Resources:

- [Mean squared logarithmic error (MSLE)](<https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-squared-logarithmic-error-(msle)>)
- [Mean squared logaritmic error Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-logarithmic-error)
- [Understanding the metric: RMSLE](https://www.kaggle.com/carlolepelaars/understanding-the-metric-rmsle)

### Mean Absolute Error

The **mean absolute error (MAE)** measure the average of the absolute values of the errors - that is, the average absolute differences between the estimated and actual values.

<p align="center"><img src="tex/5cd6e6c44dcdc5d9134e7ff6c5b812fc.svg?invert_in_darkmode" align=middle width=290.09589345pt height=49.2398742pt/></p>

Code:

- [Mean Absolute Error Numpy Implementation](code/mean_absolute_error.py)

Resources:

- [Mean absolute error Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_error)
- [Mean absolute error Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error)

### Mean Absolute Percentage Error

**Mean absolute percentage error** is an extension of the **mean absolute error (MAE)** that divides the difference between the predicted value <img src="tex/282f38ecf82d8d7b9d2813044262d5f3.svg?invert_in_darkmode" align=middle width=9.347490899999991pt height=22.831056599999986pt/> and the actual value <img src="tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.649225749999989pt height=14.15524440000002pt/> by the actual value. The main idea of MAPD is to be sensitive to relative errors. It's for example not changed by a global scaling of the target variable.

<p align="center"><img src="tex/d8bc4fe1fed0596068b06f14dc5b6186.svg?invert_in_darkmode" align=middle width=321.60739545pt height=49.2398742pt/></p>

Code:

- [Mean Absolute Percentage Error Numpy Implementation](code/mean_absolute_percentage_error.py)

Resources:

- [Mean absolute percentage error Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)
- [Mean absolute percentage error Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-percentage-error)

### Median Absolute Error

The **median absolute error** also often called **median absolute deviation (MAD)** is metric that is particularly robust to outliers. The loss is calculated by taking the median of all absolute differences between the target and the prediction.

<p align="center"><img src="tex/ce9e403e07bb796a5a4aea8e9aea8727.svg?invert_in_darkmode" align=middle width=361.860477pt height=16.438356pt/></p>

Code:

- [Median Absolute Error Numpy Implementation](code/median_absolute_error.py)

Resources:

- [Median absolute error Wikipedia](https://en.wikipedia.org/wiki/Median_absolute_deviation)
- [Median absolute error Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#median-absolute-error)

### Cosine Similarity

Cosine similarity is a measure of similarity between two vectors. The cosine similarity is the cosine of the angle between two vectors.

<p align="center"><img src="tex/0df67ef21a0ddee56433ca033cb933c1.svg?invert_in_darkmode" align=middle width=506.11591634999996pt height=91.2537549pt/></p>

Code:

- [Cosine Similarity Numpy Implementation](code/cosine_distance.py)

Resources:

- [Cosine Similarity Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Cosine Similarity – Understanding the math and how it works (with python codes)](https://www.machinelearningplus.com/nlp/cosine-similarity/)

### R2 Score

The **coefficient of determination**, denoted as <img src="tex/ee9dc84d168b211ff9f4b354e295af3c.svg?invert_in_darkmode" align=middle width=19.161017699999988pt height=26.76175259999998pt/> is the proportion of the variation in the dependent variable that has been explained by the independent variables in the model.

<p align="center"><img src="tex/a1b798ffc158c4ee0b440f4114c4f1c0.svg?invert_in_darkmode" align=middle width=216.35715585pt height=41.065845149999994pt/></p>

where <img src="tex/d5d6a7178f9ca2be9eab3bf855709944.svg?invert_in_darkmode" align=middle width=100.29605354999998pt height=27.77565449999998pt/>

Code:

- [R2 Score Numpy Implementation](code/r2_score.py)

Resources:

- [Coefficient of determination Wikipedia](https://en.wikipedia.org/wiki/Coefficient_of_determination)
- [R² score, the coefficient of determination Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score-the-coefficient-of-determination)

### Tweedie deviance

The Tweedie distributions are a family of probability distributions, which include he purely continuous normal, gamma and Inverse Gaussian distributions and more.

The unit [deviance](<https://en.wikipedia.org/wiki/Deviance_(statistics)>) of a reproductive Tweedie distribution is given by:

<p align="center"><img src="tex/bfcf5229cb3b2eb7b6472152c5538e88.svg?invert_in_darkmode" align=middle width=604.5553041pt height=100.10823074999999pt/></p>

Code:

- [Tweedie deviance Numpy Implementation](code/tweedie_deviance.py)

Resources:

- [Tweedie distribution Wikipedia](https://en.wikipedia.org/wiki/Tweedie_distribution#The_Tweedie_deviance)
- [Mean Poisson, Gamma, and Tweedie deviances](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-poisson-gamma-and-tweedie-deviances)

### Huber Loss

Huber loss is a loss function that is often used in [robust regression](https://en.wikipedia.org/wiki/Robust_regression). The function is quadratich for small values of <img src="tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode" align=middle width=8.68915409999999pt height=14.15524440000002pt/> and linear for large values.

<p align="center"><img src="tex/928194bd8bb89cb48374d0ab69a41c69.svg?invert_in_darkmode" align=middle width=265.48753109999996pt height=49.315569599999996pt/></p>

where <img src="tex/8cdee07f9c86dc6c56f28b9f8fb8ae6d.svg?invert_in_darkmode" align=middle width=68.69467275pt height=22.831056599999986pt/> and <img src="tex/cf644cbd499c18ed6f22cee5950c0d75.svg?invert_in_darkmode" align=middle width=7.928075099999989pt height=22.831056599999986pt/> is the point where the loss changes from a quadratic to linear.

Code:

- [Huber Numpy Implementation](code/huber.py)

Resources:

- [Huber loss Wikipedia](https://en.wikipedia.org/wiki/Huber_loss)
- [Huber loss Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/losses/Huber)

### Log Cosh Loss

Logarithm of the hyperbolic cosine of the prediction error.

<p align="center"><img src="tex/1ff5c2fb18f358c5a53d9f38bb1538b8.svg?invert_in_darkmode" align=middle width=300.97297725pt height=49.2398742pt/></p>

Code:

- [Log Cosh Loss Numpy Implementation](code/logcosh.py)

Resources:

- [Log Cosh loss Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/losses/log_cosh)
