import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt

data=pd.read_csv("F:\\DA_DS\\Data set\\csgo.csv")
data=data.drop(["day", "month", "year", "date"], axis=1)
# print ( data.corr())
y=data["result"]
x=data.drop("result", axis=1)
x_train, x_test, y_train, y_test= train_test_split( x, y, test_size=0.2, random_state=42, stratify=y)

num=["wait_time_s", "match_time_s", "team_a_rounds", "team_b_rounds", "ping", "kills", "assists", "deaths", "mvps", "hs_percent", "points"]
non=["map"]
prepricessor=ColumnTransformer(transformers=[
    ("num", StandardScaler(), num), 
    ("non", OneHotEncoder(handle_unknown='ignore'), non)
])

x_train=prepricessor.fit_transform(x_train)
x_test=prepricessor.transform( x_test)

# clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(x_train, x_test, y_train, y_test)
# print(models)

params={
    "n_estimators": [100, 200, 500], 
    "criterion":["gini", "entropy", "log_loss"], 
    "max_depth":[5, 10, 20]
}

grid=GridSearchCV( RandomForestClassifier(), param_grid=params, cv=5, n_jobs=1, scoring="accuracy")
model=grid.fit(x_train,y_train)
y_predict=model.predict(x_test)
print( model.best_params_)
print ( model.best_score_)
print (classification_report ( y_test, y_predict))
#            precision    recall  f1-score   support

#         Lost       0.78      0.89      0.83       112
#          Tie       1.00      1.00      1.00        17
#          Win       0.85      0.71      0.78        98

#     accuracy                           0.82       227
#    macro avg       0.88      0.87      0.87       227
# weighted avg       0.83      0.82      0.82       227


best_model = grid.best_estimator_
explainer = shap.TreeExplainer(best_model) # treeexplainer is used for tree based models like random forest, xgboost, etc
shap_values=explainer( x_test)
shap.plots.bar(shap_values[:, :, 0])
plt.show()


