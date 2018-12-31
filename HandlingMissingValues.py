import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Fill in the line below to read the file into a variable home_data
iowa_data = pd.read_csv(iowa_file_path)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iowa_target = iowa_data.SalePrice
iowa_predictors = iowa_data.drop(['SalePrice'], axis =1)

iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude=['object'])

X_train, X_test, y_train, y_test = train_test_split(iowa_numeric_predictors,
                                                    iowa_target,
                                                    train_size = 0.7,
                                                    test_size = 0.3,
                                                    random_state = 0)
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
#get model score from dropping columns with missing values

cols_with_missing = [col for col in X_train.columns
                    if X_train[col].isnull().any()]
print(cols_with_missing)
reduced_X_train = X_train.drop(cols_with_missing, axis= 1)
reduced_X_test = X_test.drop(cols_with_missing, axis = 1)
print("MAE from dropping columns with missing values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))

#get model score from imputation
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("MAE for imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))

#get score from imputation with extra columns showing what was imputed

imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()
cols_with_missing = (col for col in X_train.columns
                    if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
    #Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("MAE from imputation while tracking what was imputed")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train,y_test))
