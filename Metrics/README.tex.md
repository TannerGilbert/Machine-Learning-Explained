# Metrics

![Confusion Matrix Example](doc/confusion_matrix.png)

## Classification

### Binary cross entropy

Binary cross entropy is a loss function used for binary classification tasks (tasks with only two outcomes/classes). It works by calculating the following average:

$$\text{BinaryCrossentropy}(y, \hat{y}) = - \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} y_i * \log{\hat{y}_i} + \left(1-y_i\right) * \log{(1-\hat{y}_i)}$$

The above equation can be split into two parts to make it easier to understand:
$$\begin{align*}& \text{BinaryCrossentropy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} \mathrm{Cost}(y_i, \hat{y}_i) \\ & \mathrm{Cost}(y, \hat{y}) = -\log(\hat{y}) \; & \text{if y = 1} \\ & \mathrm{Cost}(y, \hat{y}) = -\log(1-\hat{y}) \; & \text{if y = 0}\end{align*}$$

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

Categorical crossentropy is a loss function used for multi-class classification tasks. The outputed loss is the negative average of the sum of the true values $y$ multiplied by the log of the predicted values $\log{\hat{y}}$.

$$\text{CCE}(y, \hat{y}) = - \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} y_i * \log{\hat{y}_i}$$

Resources:

- [Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names](https://gombru.github.io/2018/05/23/cross_entropy_loss/#losses)
- [Categorical crossentropy](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/categorical-crossentropy)

Code:

- [Categorical Cross Entropy Numpy Implementation](code/categorical_cross_entropy.py)

### Accuracy Score

The fraction of predictions the model classified correctly.

$$\text{accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}$$

or

$$\text{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)$$

For binary classification, accuracy can also be calculated in terms of positives and negatives as follows:

$$\text{accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

Where $\text{TP} = \text{True Positives}$, $\text{TN} = \text{True Negatives}$, $\text{FP} = \text{False Positives}$, and $\text{FN} = \text{False Negatives}$.

Resources:

- ['Classification: Accuracy' Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/classification/accuracy)
- [Accuracy Score Scikit Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score)
- [Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)

Code:

- [Accuracy Score Numpy Implementation](code/accuracy_score.py)

### Hinge Loss

Hinge loss is a loss function usef for "maximum-margin" classification, most notably for Support Vector Machines (SVMs).

$$L_\text{Hinge}(y, \hat{y}) = \max \left(0, 1 - y * \hat{y} \right)$$

Resources:

- [Hinge Loss Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#hinge-loss)
- [Hinge loss Wikipedia](https://en.wikipedia.org/wiki/Hinge_loss)
- [What is the definition of the hinge loss function?](https://ai.stackexchange.com/a/26336)

Code:

- [Hinge Loss Numpy Implementation](code/hinge.py)

### KL Divergence

The **Kullback-Leibler divergence**, $D_{KL}$, often shortenend to just KL divergence, is a measure of how one probability distribution is different from a second, reference porbability distribution.

$$D_{\text{KL}}(y \parallel \hat{y})=\sum_{i=0}^{n_{\text{samples}}-1}y \cdot \log{ \left({\frac{y}{\hat{y}}}\right)}$$

Resources:

- [Kullback–Leibler divergence Wikipedia](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
- [Kullback-Leibler Divergence Explained](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)

Code:

- [KL Divergence Numpy Implementation](code/kl_divergence.py)

## Regression

### Mean Squared Error

The **mean squared error (MSE)** or **mean squared deviation (MSD)** measure the average of the squares of the errors - that is, the average squared differences between the estimated and actual values.

$$\text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2.$$

Resources:

- [Mean squared error Wikipedia](https://en.wikipedia.org/wiki/Mean_squared_error)
- [Mean squared error Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error)
- [Machine learning: an introduction to mean squared error and regression lines](https://www.freecodecamp.org/news/machine-learning-mean-squared-error-regression-line-c7dde9a26b93/)

Code:

- [Mean Squared Error Numpy Implementation](code/mean_squared_error.py)

### Mean Squared Logarithmic Error

**Mean Squared Logarithmic Error (MSLE)** is an extension of [**Mean Squared Error (MSE)**](#mean-squared-error) often used when the target $y$ has an exponential growth.

> Note: This metrics penalizes under-predictions greater than over-predictions.

$$\text{MSLE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (\log_e (1 + y_i) - \log_e (1 + \hat{y}_i) )^2.$$

Code:

- [Mean Squared Logarithmic Error Numpy Implementation](code/mean_squared_log_error.py)

Resources:

- [Mean squared logarithmic error (MSLE)](<https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-squared-logarithmic-error-(msle)>)
- [Mean squared logaritmic error Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-logarithmic-error)
- [Understanding the metric: RMSLE](https://www.kaggle.com/carlolepelaars/understanding-the-metric-rmsle)

### Mean Absolute Error

The **mean absolute error (MAE)** measure the average of the absolute values of the errors - that is, the average absolute differences between the estimated and actual values.

$$\text{MAE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \left| y_i - \hat{y}_i \right|.$$

Code:

- [Mean Absolute Error Numpy Implementation](code/mean_absolute_error.py)

Resources:

- [Mean absolute error Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_error)
- [Mean absolute error Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error)

### Mean Absolute Percentage Error

**Mean absolute percentage error** is an extension of the **mean absolute error (MAE)** that divides the difference between the predicted value $\hat{y}$ and the actual value $y$ by the actual value. The main idea of MAPD is to be sensitive to relative errors. It's for example not changed by a global scaling of the target variable.

$$\text{MAPE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \frac{{}\left| y_i - \hat{y}_i \right|}{max(\epsilon, \left| y_i \right|)}$$

Code:

- [Mean Absolute Percentage Error Numpy Implementation](code/mean_absolute_percentage_error.py)

Resources:

- [Mean absolute percentage error Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)
- [Mean absolute percentage error Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-percentage-error)

### Median Absolute Error

The **median absolute error** also often called **median absolute deviation (MAD)** is metric that is particularly robust to outliers. The loss is calculated by taking the median of all absolute differences between the target and the prediction.

$$\text{MedAE}(y, \hat{y}) = \text{median}(\mid y_1 - \hat{y}_1 \mid, \ldots, \mid y_n - \hat{y}_n \mid).$$

Code:

- [Median Absolute Error Numpy Implementation](code/median_absolute_error.py)

Resources:

- [Median absolute error Wikipedia](https://en.wikipedia.org/wiki/Median_absolute_deviation)
- [Median absolute error Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#median-absolute-error)

### Cosine Similarity

Cosine similarity is a measure of similarity between two vectors. The cosine similarity is the cosine of the angle between two vectors.

$$\text{cosine similarity}=S_{C}(A,B):=\cos(\theta )={\mathbf {A} \cdot \mathbf {B}  \over \|\mathbf {A} \|\|\mathbf {B} \|}={\frac {\sum \limits _{i=1}^{n}{A_{i}B_{i}}}{{\sqrt {\sum \limits _{i=1}^{n}{A_{i}^{2}}}}{\sqrt {\sum \limits _{i=1}^{n}{B_{i}^{2}}}}}}$$

Code:

- [Cosine Similarity Numpy Implementation](code/cosine_distance.py)

Resources:

- [Cosine Similarity Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Cosine Similarity – Understanding the math and how it works (with python codes)](https://www.machinelearningplus.com/nlp/cosine-similarity/)

### R2 Score

The **coefficient of determination**, denoted as $R^2$ is the proportion of the variation in the dependent variable that has been explained by the independent variables in the model.

$$R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

where $\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i$

Code:

- [R2 Score Numpy Implementation](code/r2_score.py)

Resources:

- [Coefficient of determination Wikipedia](https://en.wikipedia.org/wiki/Coefficient_of_determination)
- [R² score, the coefficient of determination Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score-the-coefficient-of-determination)

### Tweedie deviance

The Tweedie distributions are a family of probability distributions, which include he purely continuous normal, gamma and Inverse Gaussian distributions and more.

The unit [deviance](<https://en.wikipedia.org/wiki/Deviance_(statistics)>) of a reproductive Tweedie distribution is given by:

$$\begin{align} \begin{split}\text{D}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} \begin{cases} (y_i-\hat{y}_i)^2, & \text{for }p=0\text{ (Normal)}\\ 2(y_i \log(y/\hat{y}_i) + \hat{y}_i - y_i),  & \text{for }p=1\text{ (Poisson)}\\ 2(\log(\hat{y}_i/y_i) + y_i/\hat{y}_i - 1),  & \text{for }p=2\text{ (Gamma)}\\ 2\left(\frac{\max(y_i,0)^{2-p}}{(1-p)(2-p)}-\frac{y\,\hat{y}^{1-p}_i}{1-p}+\frac{\hat{y}^{2-p}_i}{2-p}\right), & \text{otherwise} \end{cases}\end{split} \end{align}$$

Code:

- [Tweedie deviance Numpy Implementation](code/tweedie_deviance.py)

Resources:

- [Tweedie distribution Wikipedia](https://en.wikipedia.org/wiki/Tweedie_distribution#The_Tweedie_deviance)
- [Mean Poisson, Gamma, and Tweedie deviances](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-poisson-gamma-and-tweedie-deviances)

### Huber Loss

Huber loss is a loss function that is often used in [robust regression](https://en.wikipedia.org/wiki/Robust_regression). The function is quadratich for small values of $a$ and linear for large values.

$$L_{\delta }(y, \hat{y})={\begin{cases}{\frac {1}{2}}{a^{2}}&{\text{for }}|a|\leq \delta ,\\\delta (|a|-{\frac {1}{2}}\delta ),&{\text{otherwise.}}\end{cases}}$$

where $a = y - \hat{y}$ and $\delta $ is the point where the loss changes from a quadratic to linear.

Code:

- [Huber Numpy Implementation](code/huber.py)

Resources:

- [Huber loss Wikipedia](https://en.wikipedia.org/wiki/Huber_loss)
- [Huber loss Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/losses/Huber)

### Log Cosh Loss

Logarithm of the hyperbolic cosine of the prediction error.

$$\text{log cosh} = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \log{\left(\cosh{(x)}\right)} $$

Code:

- [Log Cosh Loss Numpy Implementation](code/logcosh.py)

Resources:

- [Log Cosh loss Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/losses/log_cosh)
