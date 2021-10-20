# Metrics

## Classification

### Binary cross entropy

$$\text{BinaryCrossentropy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} y_i * \log{\hat{y}_i} + \left(1-y_i\right) * \log{(1-\hat{y}_i)}$$

### Accuracy Score

$$\text{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)$$

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

## Tweedie deviance

$$\begin{align} \begin{split}\text{D}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} \begin{cases} (y_i-\hat{y}_i)^2, & \text{for }p=0\text{ (Normal)}\\ 2(y_i \log(y/\hat{y}_i) + \hat{y}_i - y_i),  & \text{for }p=1\text{ (Poisson)}\\ 2(\log(\hat{y}_i/y_i) + y_i/\hat{y}_i - 1),  & \text{for }p=2\text{ (Gamma)}\\ 2\left(\frac{\max(y_i,0)^{2-p}}{(1-p)(2-p)}-\frac{y\,\hat{y}^{1-p}_i}{1-p}+\frac{\hat{y}^{2-p}_i}{2-p}\right), & \text{otherwise} \end{cases}\end{split} \end{align}$$

### Huber Loss

$$L_{\delta }(y, \hat{y})={\begin{cases}{\frac {1}{2}}{(y - )^{2}}&{\text{for }}|a|\leq \delta ,\\\delta (|a|-{\frac {1}{2}}\delta ),&{\text{otherwise.}}\end{cases}}$$

### Log Cosh Loss

$$\text{log cosh} = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \log{\left(\cosh{(x)}\right)} $$

## KL Divergence

$$D_{\text{KL}}(y\parallel \hat{y})=\sum_{i=0}^{n_{\text{samples}}-1}y \log{ \left({\frac{y}{\hat{y}}}\right)}$$