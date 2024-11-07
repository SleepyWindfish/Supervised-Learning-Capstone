import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("Telco-Customer-Churn.csv")
print(df.describe())
print(df.info())

#chaes weather it is a number or not
df.isna().sum()
#sns.countplot(data=df,x="Churn")

#sns.violinplot(data=df,x='Churn',y='TotalCharges')

#plt.figure(figsize=(10,4),dpi=200)
#sns.boxplot(data=df,y='TotalCharges',x='Contract',hue='Churn')
#plt.legend(loc=(1.1,0.5))
df.columns
corr_df=pd.get_dummies(df[['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','Churn']]).corr()

corr_df['Churn_Yes'].sort_values().iloc[1:-1]
"""
plt.figure(figsize=(10,4),dpi=200)
sns.barplot(x=corr_df['Churn_Yes'].sort_values().iloc[1:-1].index,y=corr_df['Churn_Yes'].sort_values().iloc[1:-1].values)
plt.title("Feature Correlation to Yes Churn")
plt.xticks(rotation=90)"""

df['Contract'].unique()

#plt.figure(figsize=(10,3),dpi=200)
#sns.displot(data=df,x='tenure',bins=70,col='Contract',row='Churn')

print("COLUMNS:",df.columns)
#plt.figure(figsize=(10,4),dpi=200)
#sns.scatterplot(data=df,x='MonthlyCharges',y='TotalCharges',hue='Churn',linewidth=0.5,alpha=0.5,palette='Dark2')

no_churn=df.groupby(['Churn','tenure']).count().transpose()['No']
yes_churn=df.groupby(['Churn','tenure']).count().transpose()['Yes']

churn_rate=100*yes_churn/(no_churn+yes_churn)
churn_rate.transpose()['customerID']
"""
plt.figure(figsize=(10,4),dpi=200)
churn_rate.iloc[0].plot()
plt.ylabel('Churn Percentage')
"""

def cohort(tenure):
  if tenure<13:
    return '0-12 Months'
  elif tenure<25:
    return "12-24 Months"
  elif tenure<49:
    return '24-48 months'
  else:
    return "Over 48 Months"

df['Tenure Cohort']=df['tenure'].apply(cohort)
df.head(10)[['tenure','Tenure Cohort']]
"""
plt.figure(figsize=(10,4),dpi=200)    sns.scatterplot(data=df,x='MonthlyCharges',y='TotalCharges',hue='Tenure Cohort', linewidth=0.5,alpha=0.5,palette='Dark2')
"""
"""
plt.figure(figsize=(10,4),dpi=200)
sns.catplot(data=df,x='Tenure Cohort',hue='Churn',col='Contract',kind='count')
"""

#PREDICTIVE MODELING
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X = df.drop(['Churn','customerID'],axis=1)
X = pd.get_dummies(X,drop_first=True)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

t = DecisionTreeClassifier(max_depth=6)
dt=DecisionTreeClassifier(max_depth=6)
dt.fit(X_train,y_train)
preds = dt.predict(X_test)
print(preds)
"""
from sklearn.metrics import accuracy_score,plot_confusion_matrix,classification_report
print(classification_report(y_test,preds))
plot_confusion_matrix(dt,X_test,y_test)
"""
imp_feats = pd.DataFrame(data=dt.feature_importances_,index=X.columns,columns=['Feature Importance']).sort_values("Feature Importance")

plt.figure(figsize=(14,6),dpi=200)
sns.barplot(data=imp_feats.sort_values('Feature Importance'),x=imp_feats.sort_values('Feature Importance').index,y='Feature Importance')
plt.xticks(rotation=90)
plt.title("Feature Importance for Decision Tree");

from sklearn.tree import plot_tree
plt.figure(figsize=(12,8),dpi=150)
plot_tree(dt,filled=True,feature_names=X.columns);
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
ada_model = AdaBoostClassifier()
ada_model.fit(X_train,y_train)
preds = ada_model.predict(X_test)

plt.show()