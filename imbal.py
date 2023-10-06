# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

from collections import Counter

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.datasets import make_imbalance
from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NearMiss

import dataset
X,Y=dataset.datasetprepare()

from sklearn.preprocessing import OrdinalEncoder

# num_pipe = SimpleImputer(strategy="mean", add_indicator=True)
# cat_pipe = make_pipeline(
#     SimpleImputer(strategy="constant", fill_value="missing"),
#     OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
# )
# cat_pipe = make_pipeline(
#     SimpleImputer(strategy="constant", fill_value="missing"),
#     OrdinalEncoder(handle_unknown="ignore"),
# )

df_res=X
y_res=Y
print(y_res.value_counts())
from sklearn.dummy import DummyClassifier

# %%
from sklearn.model_selection import cross_validate

dummy_clf = DummyClassifier(strategy="most_frequent")
scoring = ["accuracy", "balanced_accuracy"]
cv_result = cross_validate(dummy_clf, df_res, y_res, scoring=scoring)
print(f"Accuracy score of a dummy classifier: {cv_result['test_accuracy'].mean():.3f}")

print(
    f"Balanced accuracy score of a dummy classifier: "
    f"{cv_result['test_balanced_accuracy'].mean():.3f}"
)

index = []
scores = {"Accuracy": [], "Balanced accuracy": []}




import pandas as pd

index += ["Dummy classifier"]
cv_result = cross_validate(dummy_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
print(df_scores)




from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_pipe = make_pipeline(
    StandardScaler(), SimpleImputer(strategy="mean", add_indicator=True)
)
cat_pipe = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OneHotEncoder(handle_unknown="ignore"),
)
from sklearn.compose import make_column_selector as selector
from sklearn.compose import make_column_transformer

preprocessor_linear = make_column_transformer(
    (num_pipe, selector(dtype_include="number")),
    (cat_pipe, selector(dtype_include="category")),
    n_jobs=2,
)

from sklearn.linear_model import LogisticRegression

lr_clf = make_pipeline(preprocessor_linear, LogisticRegression(max_iter=1000))

# %%
index += ["Logistic regression"]
cv_result = cross_validate(lr_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)




from sklearn.ensemble import RandomForestClassifier

# %%

preprocessor_tree = make_column_transformer(
    (num_pipe, selector(dtype_include="number")),
    (cat_pipe, selector(dtype_include="category")),
    n_jobs=2,
)

rf_clf = make_pipeline(
    preprocessor_tree, RandomForestClassifier(random_state=42, n_jobs=2)
)

# %%
index += ["Random forest"]
cv_result = cross_validate(rf_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
# print(df_scores)


lr_clf.set_params(logisticregression__class_weight="balanced")

index += ["Logistic regression with balanced class weights"]
cv_result = cross_validate(lr_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
# print(df_scores)


rf_clf.set_params(randomforestclassifier__class_weight="balanced")

index += ["Random forest with balanced class weights"]
cv_result = cross_validate(rf_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
# print(df_scores)





from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler
from imblearn.under_sampling import RandomUnderSampler

lr_clf = make_pipeline_with_sampler(
    preprocessor_linear,
    RandomUnderSampler(random_state=42),
    LogisticRegression(max_iter=1000),
)

index += ["Under-sampling + Logistic regression"]
cv_result = cross_validate(lr_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
# print(df_scores)

rf_clf = make_pipeline_with_sampler(
    preprocessor_tree,
    RandomUnderSampler(random_state=42),
    RandomForestClassifier(random_state=42, n_jobs=2),
)
index += ["Under-sampling + Random forest"]
cv_result = cross_validate(rf_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)




from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler
from imblearn.under_sampling import AllKNN

lr_clf = make_pipeline_with_sampler(
    preprocessor_linear,
    AllKNN(),
    LogisticRegression(max_iter=1000),
)

index += ["Under-sampling +KNN+ Logistic regression"]
cv_result = cross_validate(lr_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
# print(df_scores)

rf_clf = make_pipeline_with_sampler(
    preprocessor_tree,
    AllKNN(),
    RandomForestClassifier(random_state=42, n_jobs=2),
)
index += ["Under-sampling +KNN+ Random forest"]
cv_result = cross_validate(rf_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)


from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler
from imblearn.over_sampling import SMOTE

lr_clf = make_pipeline_with_sampler(
    preprocessor_linear,
    SMOTE(),
    LogisticRegression(max_iter=1000),
)

index += ["Over-sampling +SMOTE+ Logistic regression"]
cv_result = cross_validate(lr_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
# print(df_scores)

rf_clf = make_pipeline_with_sampler(
    preprocessor_tree,
    SMOTE(),
    RandomForestClassifier(random_state=42, n_jobs=2),
)
index += ["Over-sampling +SMOTE+ Random forest"]
cv_result = cross_validate(rf_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)


# print(df_scores)



# from imblearn.ensemble import BalancedRandomForestClassifier
#
# rf_clf = make_pipeline(
#     preprocessor_tree,
#     BalancedRandomForestClassifier(
#         sampling_strategy="all", replacement=True, random_state=42, n_jobs=2
#     ),
# )
# index += ["Balanced random forest"]
# cv_result = cross_validate(rf_clf, df_res, y_res, scoring=scoring)
# scores["Accuracy"].append(cv_result["test_accuracy"].mean())
# scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())
#
# df_scores = pd.DataFrame(scores, index=index)
# # print(df_scores)
#
# from sklearn.ensemble import HistGradientBoostingClassifier
#
# from imblearn.ensemble import BalancedBaggingClassifier
#
# bag_clf = make_pipeline(
#     preprocessor_tree,
#     BalancedBaggingClassifier(
#         estimator=HistGradientBoostingClassifier(random_state=42),
#         n_estimators=10,
#         random_state=42,
#         n_jobs=2,
#     ),
# )
#
# index += ["Balanced bag of histogram gradient boosting"]
# cv_result = cross_validate(bag_clf, df_res, y_res, scoring=scoring)
# scores["Accuracy"].append(cv_result["test_accuracy"].mean())
# scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())
#
# df_scores = pd.DataFrame(scores, index=index)
print(df_scores)