
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

def kernel_pca_dimensionality_reduction(indep_X, dep_Y, n,kernel='rbf'):
    # Scaling the data
    scaler = StandardScaler()
    indep_X_scaled = scaler.fit_transform(indep_X)

    # Kernel PCA apply panni top 'n' components edukkum
    kpca = KernelPCA(n_components=n, kernel=kernel)
    selected_features = kpca.fit_transform(indep_X_scaled)


    # Explained Variance  to Return 
    explained_variance = None
    return selected_features, explained_variance

def split_scalar(indep_X, dep_Y):
    X_train, X_test, y_train, y_test = train_test_split(indep_X, dep_Y, test_size=0.25, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test

def cm_prediction(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    Accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return classifier, Accuracy, report, cm

def logistic(X_train, y_train, X_test, y_test):
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    return cm_prediction(classifier, X_test, y_test)

def svm_linear(X_train, y_train, X_test, y_test):
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, y_train)
    return cm_prediction(classifier, X_test, y_test)

def svm_NL(X_train, y_train, X_test, y_test):
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)
    return cm_prediction(classifier, X_test, y_test)

def Navie(X_train, y_train, X_test, y_test):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return cm_prediction(classifier, X_test, y_test)

def knn(X_train, y_train, X_test, y_test):
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)
    return cm_prediction(classifier, X_test, y_test)

def Decision(X_train, y_train, X_test, y_test):
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    return cm_prediction(classifier, X_test, y_test)

def random(X_train, y_train, X_test, y_test):
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    return cm_prediction(classifier, X_test, y_test)

def compare_model_accuracies(acclog, accsvml, accsvmnl, accknn, accnav, accdes, accrf): 
    # DataFrame-ku 1 row dhan irukku, so list-oda 1st value eduthuko
    dataframe = pd.DataFrame(
        {
            'Logistic': [acclog[0]], 
            'SVMl': [accsvml[0]], 
            'SVMnl': [accsvmnl[0]], 
            'KNN': [accknn[0]], 
            'Navie': [accnav[0]], 
            'Decision': [accdes[0]], 
            'Random': [accrf[0]]
        },
        index=['KPCA Results']  # Index name fix panni
    )
    return dataframe


# Dataset preparation
dataset1 = pd.read_csv("prep.csv", index_col=None)
df2 = dataset1

# Preprocessing
df2 = pd.get_dummies(df2, drop_first=True)
df2.replace({True: 1, False: 0}, inplace=True)

indep_X = df2.drop('classification_yes', axis=1)
dep_Y = df2['classification_yes']


selected_features, explained_variance = kernel_pca_dimensionality_reduction(indep_X, dep_Y, 7)

# Model Training and Evaluation
acclog = []
accsvml = []
accsvmnl = []
accknn = []
accnav = []
accdes = []
accrf = []


X_train, X_test, y_train, y_test = split_scalar(selected_features, dep_Y)

classifier, Accuracy, report, cm = logistic(X_train, y_train, X_test, y_test)
acclog.append(Accuracy)

classifier, Accuracy, report, cm = svm_linear(X_train, y_train, X_test, y_test)
accsvml.append(Accuracy)

classifier, Accuracy, report, cm = svm_NL(X_train, y_train, X_test, y_test)
accsvmnl.append(Accuracy)

classifier, Accuracy, report, cm = knn(X_train, y_train, X_test, y_test)
accknn.append(Accuracy)

classifier, Accuracy, report, cm = Navie(X_train, y_train, X_test, y_test)
accnav.append(Accuracy)

classifier, Accuracy, report, cm = Decision(X_train, y_train, X_test, y_test)
accdes.append(Accuracy)

classifier, Accuracy, report, cm = random(X_train, y_train, X_test, y_test)
accrf.append(Accuracy)

# Classification Results
result = compare_model_accuracies(acclog, accsvml, accsvmnl, accknn, accnav, accdes, accrf)
