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

<p align="center"><img src="tex/c0f72f6ec2f0d5623ef75e15d1a9f197.svg?invert_in_darkmode" align=middle width=301.37445855pt height=49.2398742pt/></p>

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

### Hinge Loss

<p align="center"><img src="tex/a2f8c376f4edcf8033377b40424b287d.svg?invert_in_darkmode" align=middle width=265.89446730000003pt height=17.031940199999998pt/></p>

Code:

- [Hinge Loss Numpy Implementation](code/hinge.py)

### KL Divergence

<p align="center"><img src="tex/2d53cab7cfb446342d0f16408338cde0.svg?invert_in_darkmode" align=middle width=240.76881674999996pt height=49.2398742pt/></p>

Code:

- [KL Divergence Numpy Implementation](code/kl_divergence.py)

## Regression

### Mean Squared Error

<p align="center"><img src="tex/735371fbbd0b21c453edc23b25d47a60.svg?invert_in_darkmode" align=middle width=292.4476896pt height=49.2398742pt/></p>

Code:

- [Mean Squared Error Numpy Implementation](code/mean_squared_error.py)

### Mean Squared Logarithmic Error

<p align="center"><img src="tex/61e1a35fbe056f586e6a9dbc645eabb7.svg?invert_in_darkmode" align=middle width=441.49680795pt height=49.2398742pt/></p>

Code:

- [Mean Squared Logarithmic Error Numpy Implementation](code/mean_squared_log_error.py)

### Mean Absolute Error

<p align="center"><img src="tex/5cd6e6c44dcdc5d9134e7ff6c5b812fc.svg?invert_in_darkmode" align=middle width=290.09589345pt height=49.2398742pt/></p>

Code:

- [Mean Absolute Error Numpy Implementation](code/mean_absolute_error.py)

### Mean Absolute Percentage Error

<p align="center"><img src="tex/d8bc4fe1fed0596068b06f14dc5b6186.svg?invert_in_darkmode" align=middle width=321.60739545pt height=49.2398742pt/></p>

Code:

- [Mean Absolute Percentage Error Numpy Implementation](code/mean_absolute_percentage_error.py)

### Median Absolute Error

<p align="center"><img src="tex/ce9e403e07bb796a5a4aea8e9aea8727.svg?invert_in_darkmode" align=middle width=361.860477pt height=16.438356pt/></p>

Code:

- [Median Absolute Error Numpy Implementation](code/median_absolute_error.py)

### Cosine Similarity

<p align="center"><img src="tex/0df67ef21a0ddee56433ca033cb933c1.svg?invert_in_darkmode" align=middle width=506.11591634999996pt height=91.2537549pt/></p>

Code:

- [Cosine Similarity Numpy Implementation](code/cosine_distance.py)

### R2 Score

<p align="center"><img src="tex/a1b798ffc158c4ee0b440f4114c4f1c0.svg?invert_in_darkmode" align=middle width=216.35715585pt height=41.065845149999994pt/></p>

where <img src="tex/d5d6a7178f9ca2be9eab3bf855709944.svg?invert_in_darkmode" align=middle width=100.29605354999998pt height=27.77565449999998pt/>

Code:

- [R2 Score Numpy Implementation](code/r2_score.py)

### Tweedie deviance

<p align="center"><img src="tex/bfcf5229cb3b2eb7b6472152c5538e88.svg?invert_in_darkmode" align=middle width=604.5553041pt height=100.10823074999999pt/></p>

Code:

- [Tweedie deviance Numpy Implementation](code/tweedie_deviance.py)

### Huber Loss

<p align="center"><img src="tex/9152eaad31adde13360d6613b5dcc757.svg?invert_in_darkmode" align=middle width=265.48753109999996pt height=49.315569599999996pt/></p>

Code:

- [Huber Numpy Implementation](code/huber.py)

### Log Cosh Loss

<p align="center"><img src="tex/1ff5c2fb18f358c5a53d9f38bb1538b8.svg?invert_in_darkmode" align=middle width=300.97297725pt height=49.2398742pt/></p>

Code:

- [Log Cosh Loss Numpy Implementation](code/logcosh.py)
