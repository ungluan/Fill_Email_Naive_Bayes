from sklearn.naive_bayes import MultinomialNB
import numpy as np

e1 = [1, 2, 1, 0, 1, 0, 0]
e2 = [0, 2, 0, 0, 1, 1, 1]
e3 = [1, 0, 1, 1, 0, 2, 0]
train_data = np.array([e1, e2, e3])
label = np.array(['N', 'N', 'S'])

e4 = np.array([[1, 0, 0, 0, 0, 0, 1]])

clf1 = MultinomialNB(alpha=1)

clf1.fit(train_data, label)

print(clf1.predict_proba(e4))
print(str(clf1.predict(e4)[0]))
