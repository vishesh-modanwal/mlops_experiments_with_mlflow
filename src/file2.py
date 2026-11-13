import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# remote server created on dagshub
import dagshub
dagshub.init(repo_owner='vishesh-modanwal', repo_name='mlops_experiments_with_mlflow', mlflow=True)


mlflow.set_tracking_uri("https://dagshub.com/vishesh-modanwal/mlops_experiments_with_mlflow.mlflow")


wine = load_wine()


X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


max_depth = 20
n_estimators = 50


# mention experiment name that create in mlflow ui
mlflow.set_experiment('sakalaka_2') 


with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(X_train,y_train)
    
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test,y_pred)
    
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n estimators',n_estimators)
    
    print(accuracy)
    
    
    
    
    # cm = confusion_matrix(y_test,y_pred)
    # plt.figure(figsize=(10,7))
    # sns.heatmap(cm, annot=True,fmt='d', cmap='red', xticklabels=wine.target_names, yticklabels=wine.target_names)
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('Confusion Matrix')
    # plt.savefig('confusion_matrix.png')
    
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Reds',  
        xticklabels=wine.target_names,
        yticklabels=wine.target_names,
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    
    
     
    
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)
    plt.show()
    
    
    # tags
    mlflow.set_tags(
        {
            "author": 'vishesh',
            "project": 'mlops_experiments_with_mlflow',
        }
    )
    
    # log the model
    mlflow.sklearn.log_model(
        rf,
        name="random_forest_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    ) 