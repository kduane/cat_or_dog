import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix


def plot_loss(res, title, figsize = (12,8)):
    train_loss = res.history['loss']
    test_loss = res.history['val_loss']
  
    plt.figure(figsize = figsize)
    plt.plot(train_loss, label='Training loss', color='navy')
    plt.plot(test_loss, label='Testing loss', color='skyblue')
    
    plt.title(title)
    plt.legend();

    
def plot_accuracy(res, title, figsize = (12,8)):
    train_accuracy = res.history['accuracy']
    test_accuracy = res.history['val_accuracy']

    plt.figure(figsize = figsize)
    plt.plot(train_accuracy, label = 'Training Accuracy', color = 'lightgreen')
    plt.plot(test_accuracy, label = 'Testing Accuracy', color = 'darkgreen')
    
    plt.title(title)
    plt.legend();

    
def plot_confusion_matrix(model, X_test, y_test, title):
    y_pred_pct = model.predict(X_test)
    label_dict = dict(enumerate(y_test.columns))
    y_pred = np.vectorize(label_dict.get)(np.argmax(y_pred_pct, axis = 1))
    y_test_flat = pd.Series(y_test.columns[np.where(y_test!=0)[1]])
    cm = confusion_matrix(y_test_flat, y_pred)
    
    sns.heatmap(cm, annot = True, cmap = 'Greens', xticklabels = y_test.columns, yticklabels= y_test.columns)
    plt.title(title);