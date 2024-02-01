from xai_components.base import InArg, OutArg, InCompArg, Component, xai_component
@xai_component
class XGBoostBinaryClassifier(Component):
    """
    Trains an XGBoost classifier for binary classification tasks.

    #### Reference:
    - [XGBoost Binary Classification](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)

    ##### inPorts:
    - X_train: The training data.
    - y_train: The target variable for the training data.
    - n_estimators: Number of gradient boosted trees. Default: 100
    - max_depth: Maximum tree depth for base learners. Default: 3
    - learning_rate: Boosting learning rate. Default: 0.1
    - objective: Specify 'logistic' or 'hinge', or full specification 'binary:logistic' or 'binary:hinge'. Default: 'logistic'

    ##### outPorts:
    - model: The trained XGBoost binary classification model.
    """

    X_train: InCompArg[any]
    y_train: InCompArg[any]
    n_estimators: InArg[int]
    max_depth: InArg[int]
    learning_rate: InArg[float]
    objective: InArg[str]
    model: OutArg[any]

    def __init__(self):
        super().__init__()
        self.n_estimators.value = 100  
        self.max_depth.value = 3       
        self.learning_rate.value = 0.1 
        self.objective.value = 'logistic'  

    def execute(self, ctx) -> None:
        from xgboost import XGBClassifier
        objective_full = self.objective.value if ':' in self.objective.value else 'binary:' + self.objective.value
        self.model.value = XGBClassifier(n_estimators=self.n_estimators.value, 
                                         max_depth=self.max_depth.value, 
                                         learning_rate=self.learning_rate.value, 
                                         objective=objective_full)
        self.model.value.fit(self.X_train.value, self.y_train.value)

@xai_component
class XGBoostMultiClassClassifier(Component):
    """
    Trains an XGBoost classifier for multi-class classification tasks.

    #### Reference:
    - [XGBoost Multi-Class Classification](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)

    ##### inPorts:
    - X_train: The training data.
    - y_train: The target variable for the training data.
    - n_estimators: Number of gradient boosted trees. Default: 100
    - max_depth: Maximum tree depth for base learners. Default: 3
    - learning_rate: Boosting learning rate. Default: 0.1
    - num_class: Number of classes in the target variable. Default: 3 (Assuming a common scenario)
    - objective: Specify 'softmax' or 'softprob', or full specification 'multi:softmax' or 'multi:softprob'. Default: 'softmax'

    ##### outPorts:
    - model: The trained XGBoost multi-class classification model.
    """

    X_train: InCompArg[any]
    y_train: InCompArg[any]
    n_estimators: InArg[int]
    max_depth: InArg[int]
    learning_rate: InArg[float]
    num_class: InCompArg[int]
    objective: InArg[str]
    model: OutArg[any]

    def __init__(self):
        super().__init__()
        self.n_estimators.value = 100  
        self.max_depth.value = 3       
        self.learning_rate.value = 0.1 
        self.objective.value = 'softmax'  

    def execute(self, ctx) -> None:
        from xgboost import XGBClassifier
        objective_full = self.objective.value if ':' in self.objective.value else 'multi:' + self.objective.value
        self.model.value = XGBClassifier(n_estimators=self.n_estimators.value, 
                                         max_depth=self.max_depth.value, 
                                         learning_rate=self.learning_rate.value, 
                                         objective=objective_full, 
                                         num_class=self.num_class.value)
        self.model.value.fit(self.X_train.value, self.y_train.value)

@xai_component
class XGBoostRegressor(Component):
    """
    Trains an XGBoost regressor for regression tasks.

    #### Reference:
    - [XGBoost Regression](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html)

    ##### inPorts:
    - X_train: The training data.
    - y_train: The target variable for the training data.
    - n_estimators: Number of gradient boosted trees. Default: 100
    - max_depth: Maximum tree depth for base learners. Default: 3
    - learning_rate: Boosting learning rate. Default: 0.1
    - objective: Specify regression type 'squarederror', 'logistic', 'squaredlogerror', 'pseudohubererror', or full specification like 'reg:logistic'. Default: 'squarederror'

    ##### outPorts:
    - model: The trained XGBoost regression model.
    """

    X_train: InCompArg[any]
    y_train: InCompArg[any]
    n_estimators: InArg[int]
    max_depth: InArg[int]
    learning_rate: InArg[float]
    objective: InArg[str]
    model: OutArg[any]

    def __init__(self):
        super().__init__()
        self.n_estimators.value = 100  
        self.max_depth.value = 3       
        self.learning_rate.value = 0.1 
        self.objective.value = 'squarederror'  

    def execute(self, ctx) -> None:
        from xgboost import XGBRegressor
        objective_full = self.objective.value if ':' in self.objective.value else 'reg:' + self.objective.value
        self.model.value = XGBRegressor(n_estimators=self.n_estimators.value, 
                                        max_depth=self.max_depth.value, 
                                        learning_rate=self.learning_rate.value, 
                                        objective=objective_full)
        self.model.value.fit(self.X_train.value, self.y_train.value)

@xai_component
class XGBoostRanker(Component):
    """
    Trains an XGBoost model for ranking tasks.

    #### Reference:
    - [XGBoost Learning to Rank](https://xgboost.readthedocs.io/en/latest/tutorials/ranking.html)

    ##### inPorts:
    - X_train: The training data.
    - y_train: The target variable for the training data.
    - n_estimators: Number of gradient boosted trees. Default: 100
    - max_depth: Maximum tree depth for base learners. Default: 3
    - learning_rate: Boosting learning rate. Default: 0.1
    - objective: Specify ranking type 'pairwise', 'ndcg', 'map', or full specification like 'rank:pairwise'. Default: 'rank:pairwise'

    ##### outPorts:
    - model: The trained XGBoost ranking model.
    """

    X_train: InCompArg[any]
    y_train: InCompArg[any]
    n_estimators: InArg[int]
    max_depth: InArg[int]
    learning_rate: InArg[float]
    objective: InArg[str]
    model: OutArg[any]

    def __init__(self):
        super().__init__()
        self.n_estimators.value = 100  
        self.max_depth.value = 3       
        self.learning_rate.value = 0.1 
        self.objective.value = 'rank:pairwise'  

    def execute(self, ctx) -> None:
        from xgboost import XGBRanker
        objective_full = self.objective.value if ':' in self.objective.value else 'rank:' + self.objective.value
        self.model.value = XGBRanker(n_estimators=self.n_estimators.value, 
                                     max_depth=self.max_depth.value, 
                                     learning_rate=self.learning_rate.value, 
                                     objective=objective_full)
        self.model.value.fit(self.X_train.value, self.y_train.value)


@xai_component
class XGBoostBinaryPredict(Component):
    """
    Makes predictions using a trained XGBoost classifier and optionally evaluates the accuracy of those predictions.

    #### Reference:
    - [XGBoost Prediction](https://xgboost.readthedocs.io/en/latest/python/python_intro.html#prediction)

    ##### inPorts:
    - bst: The trained XGBoost model.
    - X_test: The testing data.
    - y_test: The target variable for the testing data. If provided, the accuracy of the predictions is evaluated.

    ##### outPorts:
    - preds: The model's predictions.
    - accuracy: The accuracy of the model's predictions, if y_test was provided.
    """

    bst: InCompArg[any]
    X_test: InCompArg[any]
    y_test: InArg[any]
    preds: OutArg[any]
    accuracy: OutArg[float]

    def execute(self, ctx) -> None:
        from sklearn.metrics import accuracy_score

        self.preds.value = self.bst.value.predict(self.X_test.value)

        if self.y_test.value is not None:
            self.accuracy.value = accuracy_score(self.y_test.value, self.preds.value)
            print(f"Accuracy: {self.accuracy.value * 100:.2f}%")
        else:
            self.accuracy.value = None