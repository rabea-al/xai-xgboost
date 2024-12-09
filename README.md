<p align="center">
  <a href="https://github.com/XpressAI/xircuits/tree/master/xai_components#xircuits-component-library-list">Component Libraries</a> •
  <a href="https://github.com/XpressAI/xircuits/tree/master/project-templates#xircuits-project-templates-list">Project Templates</a>
  <br>
  <a href="https://xircuits.io/">Docs</a> •
  <a href="https://xircuits.io/docs/Installation">Install</a> •
  <a href="https://xircuits.io/docs/category/tutorials">Tutorials</a> •
  <a href="https://xircuits.io/docs/category/developer-guide">Developer Guides</a> •
  <a href="https://github.com/XpressAI/xircuits/blob/master/CONTRIBUTING.md">Contribute</a> •
  <a href="https://www.xpress.ai/blog/">Blog</a> •
  <a href="https://discord.com/invite/vgEg2ZtxCw">Discord</a>
</p>


<p align="center"><i>Build efficient and scalable machine learning models with the Xircuits Component Library for XGBoost integration!</i></p>


---
## Xircuits Component Library for XGBoost

This library integrates XGBoost into Xircuits workflows, enabling seamless model training, evaluation, and optimization for scalable and efficient machine learning tasks.

## Table of Contents

- [Preview](#preview)
  
- [Prerequisites](#prerequisites)
- [Main Xircuits Components](#main-xircuits-components)
- [Try the Examples](#try-the-examples)
- [Installation](#installation)

## Preview
### The Example

![XGBoostClassifier](https://github.com/user-attachments/assets/2a2d5ffe-79e0-4f0b-b2f1-1660346f0e95)

### The Result

<img src="https://github.com/user-attachments/assets/51a6f8d4-5567-4dc0-b0da-5cfb7b23fd02" alt="XGBoostClassifier result" />

## Prerequisites

Before you begin, you will need the following:

1. Python3.9+.
2. Xircuits.

## Main Xircuits Components 

### XGBoostBinaryClassifier Component:  
Trains a binary classifier with XGBoost, allowing customization of parameters like tree depth, learning rate, and boosting objective.

<img src="https://github.com/user-attachments/assets/c13c82f5-c5a0-4191-996a-2e0fd686e4dc" alt="XGBoostBinaryClassifier" width="200" height="150" />


### XGBoostMultiClassClassifier Component:
Trains an XGBoost classifier for multi-class problems with configurable tree depth, learning rate, and objective functions. Suitable for tasks requiring efficient handling of multi-class datasets.

<img src="https://github.com/user-attachments/assets/787991f9-c316-408b-a84a-a29d7670f157" alt="XGBoostMultiClassClassifier" width="200" height="175" />


### XGBoostRegressor Component:
Trains an XGBoost regressor for regression tasks, supporting objectives like squared error and logistic regression. Offers high flexibility and accuracy in modeling numerical targets.

### XGBoostRanker Component:
Trains an XGBoost model for ranking tasks using objectives like pairwise and ndcg. Designed for learning-to-rank problems, it optimizes rank-based evaluation metrics.

### XGBoostBinaryPredict Component:
Generates predictions from a trained XGBoost binary classifier. Optionally evaluates the accuracy of predictions if the target variable is provided.

## Try the Examples
We have provided an example workflow to help you get started with the XGBoost component library. Give it a try and see how you can create custom XGBoost components for your applications.

### XGBoostClassifier

This example demonstrates a workflow for training and testing an XGBoost binary classifier using the Iris dataset. It splits the dataset into training and testing sets, trains the classifier, and evaluates its performance with predictions and accuracy.

## Installation
To use this component library, ensure that you have an existing [Xircuits setup](https://xircuits.io/docs/main/Installation). You can then install the XGBoost library using the [component library interface](https://xircuits.io/docs/component-library/installation#installation-using-the-xircuits-library-interface), or through the CLI using:
```
xircuits install xgboost
```
You can also do it manually by cloning and installing it:
```
# base Xircuits directory
git clone https://github.com/XpressAI/xai-xgboost xai_components/xai_xgboost
pip install -r xai_components/xai_xgboost/requirements.txt 
``` 
