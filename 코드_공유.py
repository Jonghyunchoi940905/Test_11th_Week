# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

wine = load_wine()
df=pd.DataFrame(data=wine.data, columns=wine.feature_names)
df['target'] = wine.target

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

X = df.drop('target',axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

param_grid = {
    "criterion" : ['gini', 'entropy'],
    "max_depth" : [2,5],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 2, 4]
}

clf_grid = DecisionTreeClassifier( random_state= 42 )
grid_search = GridSearchCV(clf_grid, param_grid, cv = 5)
grid_search.fit(X_train, y_train)

print("Best Hyper-parameter", grid_search.best_params_)
print("Best Score", grid_search.best_score_)

best_model = grid_search.best_estimator_

y_pred_grid = best_model.predict(X_test)
accuracy_grid_DT = accuracy_score(y_test, y_pred_grid)
print('Accuracy Grid :', accuracy_grid_DT)
print("\n Classification Report", classification_report(y_test, y_pred_grid))
report_DT = classification_report(y_test, y_pred_grid)

# Feature Importance를 계산
importances = best_model.feature_importances_

# Best model의 Feature Importance를  시각화
plt.figure(figsize = (20,6))
# 막대 그래프 생성
plt.bar(range(len(importances)), importances, width=0.3)
plt.xlabel('Feature')
plt.ylabel('importances')
plt.title('Feature Importance')
plt.xticks(range(len(importances)), X.columns, rotation = 45)
plt.show()