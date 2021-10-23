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

If the correct answer is 1, then the cost function will be 0 if the prediction is 1. If the prediction approaches  0, then the cost function will approach infinity.

Resources:
* [Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names](https://gombru.github.io/2018/05/23/cross_entropy_loss/#binary-cross-entropy-loss)
* [Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)
* [Understanding binary cross-entropy / log loss: a visual explanation](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)

### Accuracy Score

The fraction of predictions the model classified correctly.

$$\text{accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}$$

or

$$\text{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)$$

For binary classification, accuracy can also be calculated in terms of positives and negatives as follows:

$$\text{accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

Where $\text{TP} = \text{True Positives}$, $\text{TN} = \text{True Negatives}$, $\text{FP} = \text{False Positives}$, and $\text{FN} = \text{False Negatives}$.

Resources:
* ['Classification: Accuracy' Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/classification/accuracy)
* [Accuracy Score Scikit Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score)
* [Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)

### Hinge Loss

$$L_\text{Hinge}(y_w, y_t) = \max\left\{1 + y_t - y_w, 0\right\}$$

## Regression

### Mean Squared Error

$$\text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2.$$

### Mean Squared Logarithmic Error

$$\text{MSLE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (\log_e (1 + y_i) - \log_e (1 + \hat{y}_i) )^2.$$

### Mean Absolute Error

$$\text{MAE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \left| y_i - \hat{y}_i \right|.$$

### Mean Absolute Percentage Error

$$\text{MAPE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \frac{{}\left| y_i - \hat{y}_i \right|}{max(\epsilon, \left| y_i \right|)}$$

### Median Absolute Error

$$\text{MedAE}(y, \hat{y}) = \text{median}(\mid y_1 - \hat{y}_1 \mid, \ldots, \mid y_n - \hat{y}_n \mid).$$

### Cosine Similarity

$$\text{cosine similarity}=S_{C}(A,B):=\cos(\theta )={\mathbf {A} \cdot \mathbf {B}  \over \|\mathbf {A} \|\|\mathbf {B} \|}={\frac {\sum \limits _{i=1}^{n}{A_{i}B_{i}}}{{\sqrt {\sum \limits _{i=1}^{n}{A_{i}^{2}}}}{\sqrt {\sum \limits _{i=1}^{n}{B_{i}^{2}}}}}}$$

### R2 Score

$$R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

where $\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i$

### Tweedie deviance

$$\begin{align} \begin{split}\text{D}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} \begin{cases} (y_i-\hat{y}_i)^2, & \text{for }p=0\text{ (Normal)}\\ 2(y_i \log(y/\hat{y}_i) + \hat{y}_i - y_i),  & \text{for }p=1\text{ (Poisson)}\\ 2(\log(\hat{y}_i/y_i) + y_i/\hat{y}_i - 1),  & \text{for }p=2\text{ (Gamma)}\\ 2\left(\frac{\max(y_i,0)^{2-p}}{(1-p)(2-p)}-\frac{y\,\hat{y}^{1-p}_i}{1-p}+\frac{\hat{y}^{2-p}_i}{2-p}\right), & \text{otherwise} \end{cases}\end{split} \end{align}$$

### Huber Loss

$$L_{\delta }(y, \hat{y})={\begin{cases}{\frac {1}{2}}{(y - )^{2}}&{\text{for }}|a|\leq \delta ,\\\delta (|a|-{\frac {1}{2}}\delta ),&{\text{otherwise.}}\end{cases}}$$

### Log Cosh Loss

$$\text{log cosh} = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \log{\left(\cosh{(x)}\right)} $$

### KL Divergence

$$D_{\text{KL}}(y\parallel \hat{y})=\sum_{i=0}^{n_{\text{samples}}-1}y \log{ \left({\frac{y}{\hat{y}}}\right)}$$