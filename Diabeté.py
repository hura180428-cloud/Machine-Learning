import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

data=pd.read_csv("D:\\DA_DS\\Data set\\diabetes.csv")
# sns.heatmap( data.corr(), annot=True)
# plt.show()

x=data.drop("Outcome", axis=1)
y=data["Outcome"]
x_train, x_test, y_train, y_test= train_test_split(x,y, train_size=0.8, random_state=42)
# smote with numerical features to balance dataset

rbs=SMOTE(random_state=42, k_neighbors=5)
x_train, y_train=rbs.fit_resample(x_train, y_train)

# preprocessing
num=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI","DiabetesPedigreeFunction","Age" ]
# can consider to use transformer if data have multiple type of data including two types of categorical data, and numerical data
# but in this dataset, we just have one type of input, fit directly to standard scaller
std=StandardScaler()
x_train=std.fit_transform(x_train[num])
x_test=std.transform ( x_test[num])

# clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(x_train, x_test, y_train, y_test)
# print ( models)

# using grid search to find the best parametters for Linear SCV
params={
    "n_estimators": [100, 200, 500],
    "criterion":["gini", "entropy", "log_loss"],
    "max_depth":[5,10,20],
    "max_features":["sqrt", "log2"]
}
grid=GridSearchCV(RandomForestClassifier(), param_grid=params, cv=5, n_jobs=5, verbose=0, scoring="precision")
model=grid.fit(x_train, y_train)
print( model.best_params_)
print ( model.best_estimator_)
print ( model.best_score_)
# print ( data["Outcome"].value_counts())
y_predict=model.predict(x_test)
# print (f"Actual value: {y_test}, predicted_value: {y_predict}")









           
