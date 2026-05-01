import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from lazypredict.Supervised import LazyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

data=pd.read_csv("D:\\DA_DS\\Data set\\StudentScore.xls")
# plt.hist(data)
# plt.show()
x=data.drop("writing score", axis=1)
y=data["writing score"]

x_train, x_test, y_train, y_test= train_test_split( x, y, test_size=0.2, random_state=42)
# print ( data["writing score"].value_counts())
num=["reading score", "math score"]
non=["race/ethnicity"]
ord=["gender", "parental level of education","lunch", "test preparation course"]
gender_values=data["gender"].unique()
parental_values= ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
lunch_values=data["lunch"].unique()
test_preparation_values=data["test preparation course"].unique()
# print ( data["parental level of education"].unique())

preprocessor=ColumnTransformer(transformers=[
    ("num", StandardScaler(), num), 
    ("non", OneHotEncoder(), non),
    ("ord", OrdinalEncoder(categories=[gender_values, parental_values, lunch_values, test_preparation_values]), ord)
])
x_train=preprocessor.fit_transform( x_train)
x_test=preprocessor.transform(x_test)

# reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = reg.fit(x_train, x_test, y_train, y_test)
# print ( models)

# params={
#     "epsilon": [0, 0.1, 0.2, 0.5], 
#     "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
#     "intercept_scaling": [0.1, 0.2, 0.5]
# }
model= LinearRegression()
model_=model.fit( x_train, y_train)
y_predict=model_.predict( x_test)
# # print ( model.best_params_)
# # print ( model.best_score_)
# print ( model_.score)
print("R2 Score:", r2_score(y_test, y_predict))
print("Mean Squared Error:", mean_squared_error(y_test, y_predict))




