"""
PassengerId : 탑승객의 고유 아이디
Survival : 생존유무(0: 사망, 1: 생존)
Pclass : 등실의 등급
Name : 이름
Sex : 성별
Age : 나이
SibSp : 함께 탑승한 형제자매, 아내, 남편의 수
Parch : 함께 탑승한 부모, 자식의 수
Ticket : 티켓번호
Fare : 티켓의 요금
Cabin : 객실번호
Embarked : 배에 탑승한 위치(C = Cherbourg, Q = Queenstown, S = Southampton)
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("train.csv")
test = pd.read_csv('test.csv')

sample_submission = pd.read_csv('sample_submission.csv')
train = train.drop(['Ticket', 'Cabin'], axis = 1)
test = test.drop(['Ticket', 'Cabin'], axis = 1)
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1}).astype(int)
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1}).astype(int)
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

Pclass3 = test[test['Pclass'] == 3]
test['Fare'] = test['Fare'].fillna(Pclass3['Fare'].median())

train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train['Name'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Name'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

train['Name'] = train['Name'].replace(['Master', 'Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train['Name'] = train['Name'].replace('Mlle', 'Miss')
train['Name'] = train['Name'].replace('Ms', 'Miss')
train['Name'] = train['Name'].replace('Mme', 'Mrs')

test['Name'] = test['Name'].replace(['Master', 'Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test['Name'] = test['Name'].replace('Mlle', 'Miss')
test['Name'] = test['Name'].replace('Ms', 'Miss')
test['Name'] = test['Name'].replace('Mme', 'Mrs')

name_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Rare": 4}
train['Name'] = train['Name'].map(name_mapping)
train['Name'] = train['Name'].fillna(0)

name_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Rare": 4}
test['Name'] = test['Name'].map(name_mapping)
test['Name'] = test['Name'].fillna(0)

train.loc[train['FamilySize'] <= 1, 'FamilySize'] = 0
train.loc[(train['FamilySize'] > 1) & (train['FamilySize'] <= 3), 'FamilySize'] = 1
train.loc[(train['FamilySize'] > 3), 'FamilySize'] = 2

test.loc[test['FamilySize'] <= 1, 'FamilySize'] = 0
test.loc[(test['FamilySize'] > 1) & (test['FamilySize'] <= 3), 'FamilySize'] = 1
test.loc[(test['FamilySize'] > 3), 'FamilySize'] = 2

#Age 전처리학습(호칭과 Familysize로?) 호칭은 성별, 나이, 결혼유무 알려줌
train_notnull = train.dropna(axis = 0)
test_notnull = test.dropna(axis = 0)
train_age_X = train_notnull[['Name','FamilySize', 'SibSp','Parch']]
test_age_X = test_notnull[['Name','FamilySize', 'SibSp','Parch']]
final_train_age_X = pd.concat([train_age_X,test_age_X], ignore_index=False)

train_age_Y = train_notnull['Age']
test_age_Y = test_notnull['Age']
final_train_age_Y = pd.concat([train_age_Y,test_age_Y], ignore_index=False)
final_train_age_Y = final_train_age_Y.astype(int)

train_age_null = train[train['Age'].isnull()]
predictinput = train_age_null[['Name','FamilySize', 'SibSp','Parch']]
logreg = LogisticRegression()
logreg.fit(final_train_age_X, final_train_age_Y)

Y_pred = logreg.predict(predictinput)
Y_pred_series = pd.Series(Y_pred)
Y_pred_series.index = train_age_null.index
for i in Y_pred_series.index.tolist():
    train.loc[i,'Age'] = Y_pred_series.loc[i]

test_age_null = test[test['Age'].isnull()]
predictinput2 = test_age_null[['Name','FamilySize', 'SibSp','Parch']]
Y_pred2 = logreg.predict(predictinput2)
Y_pred2_series = pd.Series(Y_pred2)
Y_pred2_series.index = test_age_null.index
for i in Y_pred2_series.index.tolist():
    test.loc[i,'Age'] = Y_pred2_series.loc[i]

X_train = train.drop(["PassengerId", "Survived"], axis=1)
Y_train = train["Survived"]
X_test = test.drop(["PassengerId"], axis=1).copy()

sample_submission = pd.read_csv('sample_submission.csv')

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
sample_submission['Survived'] = Y_pred
sample_submission.to_csv('submission.csv', index=False)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 결정트리, Random Forest, 로지스틱 회귀를 위한 사이킷런 Classifier 클래스 생성
dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression()

# DecisionTreeClassifier 학습/예측/평가
dt_clf.fit(X_train , Y_train)
dt_pred = dt_clf.predict(X_test)
sample_submission['Survived'] = dt_pred
sample_submission.to_csv('submission2.csv', index=False)

# RandomForestClassifier 학습/예측/평가
rf_clf.fit(X_train , Y_train)
rf_pred = rf_clf.predict(X_test)
sample_submission['Survived'] = rf_pred
sample_submission.to_csv('submission3.csv', index=False)

# LogisticRegression 학습/예측/평가
lr_clf.fit(X_train , Y_train)
lr_pred = lr_clf.predict(X_test)
sample_submission['Survived'] = lr_pred
sample_submission.to_csv('submission4.csv', index=False)

print(dt_clf.score(X_train, Y_train))
print(rf_clf.score(X_train, Y_train))
print(lr_clf.score(X_train, Y_train))
