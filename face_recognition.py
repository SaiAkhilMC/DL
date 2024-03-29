import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import classification_report

data = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = data.data, data.target

n_components = 256
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
pca.fit(X)
X_pca = pca.transform(X)

X_pca.shape

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.25, random_state=42)

clf = SVC(kernel='linear', C=1000, gamma=0.001)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("y_test \t y_pred \n")
print(np.column_stack((y_test, y_pred)))

print("Classification Report")

print(classification_report(y_test, y_pred))

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=16)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print("y_test \t y_pred \n")
print(np.column_stack((y_test, y_pred_logreg)))

print("Classification Report")
print(classification_report(y_test, y_pred_logreg))

