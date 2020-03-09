import pandas as pd
import fasttext
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import string, re

sample_submission = pd.read_csv("sample_submission.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
train.replace(np.nan, '', regex=True, inplace=True)
test.replace(np.nan, '', regex=True, inplace=True)

train['textagain'] = train.text + ' ' + train.location + ' ' + train.keyword
test['text'] = test.text + ' ' + test.location + ' ' + test.keyword

train.head()

train.textagain[93]

train.textagain = train.textagain.str.lower()
train.textagain = train.textagain.str.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(' '*5, ' ').replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ')

train.textagain = train.textagain.str.translate(str.maketrans(string.digits, ' ' * len(string.digits))).replace(' '*5, ' ').replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ')

train.textagain = train.textagain.replace(regex=True,to_replace=r'\s+',value=' ')
train.textagain = train.textagain.str.strip()


train.textagain[93]
train.text = train.textagain

train.text[93]

model = fasttext.load_model('wiki.en/wiki.en.bin')

X, y = np.zeros((len(train), 300)), np.zeros((len(train)))

for i in range(len(train)):
    X[i] = model.get_sentence_vector(" ".join(train.loc[i].text.split()))
    y[i] = train.loc[i].target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape
y_train.shape

clf = RandomForestClassifier(n_jobs=-1, n_estimators=10000, max_depth=100)

clf.fit(X_train, y_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)



test.text = test.text.str.lower()
test.textagain = test.text.str.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(' '*5, ' ').replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ')

test.text = test.text.str.translate(str.maketrans(string.digits, ' ' * len(string.digits))).replace(' '*5, ' ').replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ')

test.text = test.text.replace(regex=True,to_replace=r'\s+',value=' ')
test.text = test.text.str.strip()




X_pred, y_pred = np.zeros((len(test), 300)), np.zeros((len(test)))

for i in range(len(test)):
    X_pred[i] = model.get_sentence_vector(" ".join(test.loc[i].text.split()))

y_pred = clf.predict(X_pred)

X_pred.shape
y_pred.shape

submission = pd.DataFrame()

submission['id'] = test['id']
submission['target'] = y_pred.astype('int64')

submission.to_csv('submission.csv', index=False)


test.head()












#TF-idf method check
train.head()



tfidf = TfidfVectorizer(stop_words='english', encoding='utf-8', ngram_range=(1, 3))
features = tfidf.fit_transform(train.text)
labels = train.target
features.shape
X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state = 0)

clf = RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=1000, random_state=0)
clf = clf.fit(X_train, y_train)
clf.score(X_test, y_test)
clf.score(X_train, y_train)

y_pred = clf.predict(tfidf.transform(test.text))
