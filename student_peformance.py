import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

data=pd.read_csv("D:\\DA_DS\\Data set\\student_performance_dataset.csv")
# print (data.isna().sum())
data['parental_education']=data['parental_education'].fillna('Unknown', inplace=True)
print (data.isna().sum())
data=data.drop (['student_id'], axis=1)
x=data.drop(['final_grade'], axis=1)
y=data['final_grade']
x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=42, stratify=y)
num=['study_time_hours', 'attendance_percent', 'sleep_hours', 'previous_grade', 'final_exam_score']
ord_cat=['parental_education', 'internet_access', 'extracurricular_activities', 'part_time_job']
parental_values=['Unknown', 'High School', 'Bachelors', 'Masters', 'PhD']
internet_access_values=data['internet_access'].unique()
extracurricular_values=data['extracurricular_activities'].unique()
part_time_job_values=data['part_time_job'].unique()
preprocessor=ColumnTransformer([
    ("num", StandardScaler(), num),
    ("ord", OrdinalEncoder(categories=[parental_values, internet_access_values, extracurricular_values, part_time_job_values]), ord_cat)
])
x_train=preprocessor.fit_transform ( x_train)
x_test=preprocessor.transform ( x_test)

# lazy=LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# model, prediction=lazy.fit( x_train, x_test, y_train, y_test)
# print ( model)
params={
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}



grid=GridSearchCV( DecisionTreeClassifier(), param_grid=params, cv=5, n_jobs=3, scoring='accuracy')
model=grid.fit ( x_train, y_train)
y_predict=model.predict( x_test)
print ( model.best_params_)
print ( model.best_score_)
print(classification_report(y_test, y_predict))


# result
# best_param ={'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}


#               precision    recall  f1-score   support

#            A       1.00      1.00      1.00        57
#            B       1.00      1.00      1.00        71
#            C       1.00      1.00      1.00        52
#            D       1.00      0.94      0.97        18
#            F       0.67      1.00      0.80         2

#     accuracy                           0.99       200
#    macro avg       0.93      0.99      0.95       200
# weighted avg       1.00      0.99      1.00       200

