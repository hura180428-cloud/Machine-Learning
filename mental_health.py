import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import GridSearchCV
from  sklearn.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt


data=pd.read_csv("D:\\DA_DS\\Data set\\Teen_Mental_Health_Dataset.csv")
# print (data)
data=data.drop(["anxiety_level","addiction_level","depression_label"], axis=1)
target= "stress_level"
data [target]= data [target].astype( str)
# data.loc[row, column]: change data dirextly from data frame
#  enumerate require two variables: the first is the index of the data, the other one is the value of this data
for index, i in enumerate(data [target]):
    if int (i)<5:
        data.loc[index, target]="normal"

    else: 
        data.loc [index, target]= "risk"

x= data.drop("stress_level", axis=1)
y=data["stress_level"]
x_train , x_test, y_train, y_test= train_test_split(x, y, train_size=0.75, random_state=42, stratify=y)

num=["age", "daily_social_media_hours", "sleep_hours","screen_time_before_sleep","academic_performance","physical_activity"]
ord=["gender", "social_interaction_level"]
gender_values=data["gender"].unique()
cs_interaction_values=["low", "medium", "high"]
# print(data["social_interaction_level"]).unique()
preprocessor=ColumnTransformer([
    ("num", StandardScaler(), num),
    ("ord", OrdinalEncoder(categories=[gender_values, cs_interaction_values]), ord)
])
x_train=preprocessor.fit_transform(x_train)
x_test=preprocessor.transform (x_test)

# reg = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = reg.fit(x_train, x_test, y_train, y_test)
# print ( models)
# print ( data["stress_level"].value_counts())  
params= {
    "penalty": ["l2"], 
    "l1_ratio": [0.1,  0.3, 0.5, 0.7, 0.9], 
    "C": [0.01, 0.1, 1, 10, 50, 100], 
    "solver": ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
}
grid=GridSearchCV( LogisticRegression(), param_grid=params, cv=5, n_jobs=3, scoring= "accuracy")
model=grid.fit ( x_train, y_train)
y_predict=model.predict( x_test)
print ( model.best_params_)
print ( model.best_estimator_)
print ( model.best_score_)
print(classification_report(y_test, y_predict))

best_model = grid.best_estimator_

explainer = shap.Explainer(best_model, x_train)
shap_values = explainer.shap_values(x_test)

shap.summary_plot(shap_values, x_test)


#  RESULTS
#              precision    recall  f1-score   support

#       normal       0.00      0.00      0.00       124
#         risk       0.59      1.00      0.74       176

#     accuracy                           0.59       300
#    macro avg       0.29      0.50      0.37       300
# weighted avg       0.34      0.59      0.43       300