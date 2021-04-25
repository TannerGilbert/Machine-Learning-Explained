# AdaBoost - Adaptive Boosting

![Adaboost Decision Boundary](doc/adaboost.png)

AdaBoost, short for **Ada**ptive [**Boost**ing](https://en.wikipedia.org/wiki/Boosting_(meta-algorithm)), of Freund and Schapire, was the first practical boosting algorithm and remains one of the most widely used and studied ones even today. Boosting is a general strategy for learning "strong models" by combining multiple simpler ones (weak models or weak learners).

A "weak learner" is a model that will do at least slightly better than chance. AdaBoost can be applied to any classification algorithm, but most often, it's used with **Decision Stumps** - Decision Trees with only one node and two leaves.

![Decision Stump](doc/decision_stump.PNG)

Decision Stumps alone are not an excellent way to make predictions. A full-grown decision tree combines the decisions from all features to predict the target value. A stump, on the other hand, can only use one feature to make predictions.

## How does the AdaBoost algorithm work?

1. Initialize sample weights uniformly as <img src="tex/dc56b266dfc19aea6656ef2dde1f1f14.svg?invert_in_darkmode" align=middle width=55.12169519999999pt height=27.77565449999998pt/>.
2. For each iteration <img src="tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/>:

**Step 1:** A weak learner (e.g. a decision stump) is trained on top of the weighted training data <img src="tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=22.465723500000017pt/>. The weight of each sample <img src="tex/c2a29561d89e139b3c7bffe51570c3ce.svg?invert_in_darkmode" align=middle width=16.41940739999999pt height=14.15524440000002pt/> indicates how important it is to classify the sample correctly.

**Step 2:** After training, the weak learner gets a weight based on its accuracy <img src="tex/3826eeb617fdc1a5c8840e859a7dafbb.svg?invert_in_darkmode" align=middle width=119.23682924999997pt height=37.80850590000001pt/>

![Alpha](doc/alpha.png)

**Step 3:** The weights of misclassified samples are updated <img src="tex/b4128148f8163b17d8269f72bf4e6d74.svg?invert_in_darkmode" align=middle width=186.86824199999998pt height=34.337843099999986pt/>

**Step 4:** Renormalize weights so they sum up to 1 <img src="tex/447a46ee6fdce8100ddf3d57c464612b.svg?invert_in_darkmode" align=middle width=117.2613255pt height=34.337843099999986pt/>

3. Make predicts using a linear combination of the weak learners <img src="tex/00cdc31549c67b60c6dff38106fea53a.svg?invert_in_darkmode" align=middle width=206.13978495pt height=37.80850590000001pt/>

![Adaboost Training](doc/adaboost_training.gif)

## Code

- [Adaboost Python](code/adaboost.py)

## Resources

- [https://scikit-learn.org/stable/modules/ensemble.html#adaboost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)
- [https://www.youtube.com/watch?v=LsK-xG1cLYA](https://www.youtube.com/watch?v=LsK-xG1cLYA)
- [https://blog.paperspace.com/adaboost-optimizer/](https://blog.paperspace.com/adaboost-optimizer/)
- [https://en.wikipedia.org/wiki/AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)
- [https://geoffruddock.com/adaboost-from-scratch-in-python/](https://geoffruddock.com/adaboost-from-scratch-in-python/)
- [https://www.cs.toronto.edu/~mbrubake/teaching/C11/Handouts/AdaBoost.pdf](https://www.cs.toronto.edu/~mbrubake/teaching/C11/Handouts/AdaBoost.pdf)
- [https://jeremykun.com/2015/05/18/boosting-census/](https://jeremykun.com/2015/05/18/boosting-census/)
- [https://ml-explained.com/blog/decision-tree-explained](https://ml-explained.com/blog/decision-tree-explained)