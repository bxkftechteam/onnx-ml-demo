#!/usr/bin/env python3

"""Train several models for phone-play detector using synthetic data"""

from os import path
import random
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from hmmlearn.hmm import MultinomialHMM

from map_pred import map_pred
from infer import infer

# Load dataset
project_dir = path.dirname(path.dirname(path.abspath(__file__)))
with open(path.join(project_dir, "data/synthetic.npy"), "rb") as f:
    dataset = pickle.load(f)

# Split dataset to training set and test set
random.shuffle(dataset)
split_pos = int(len(dataset) * 0.8)
feature_train = np.vstack([d["X"] for d in dataset[:split_pos]])
feature_test = np.vstack([d["X"] for d in dataset[split_pos:]])
target_train = np.hstack([d["label"] for d in dataset[:split_pos]])
target_test = np.hstack([d["label"] for d in dataset[split_pos:]])

# Train scaler
scaler = StandardScaler()
scaler.fit(feature_train)
feature_train = scaler.transform(feature_train)

# Train random forest classifier
clf = RandomForestClassifier()
clf.fit(feature_train, target_train)

# Train HMM
pred_probs = clf.predict_proba(feature_train)[:, 1]
pred_labels = np.array([map_pred(x) for x in pred_probs], dtype=np.int64)
hmm = MultinomialHMM(n_components=2,
                     startprob_prior=np.array([0.5, 0.5]),
                     transmat_prior=np.array([
                         [0.8, 0.2],
                         [0.2, 0.8],
                     ]))
hmm.fit(pred_labels.reshape(-1, 1))

# Evaluation of the entire procedure
predict_results = infer(feature_test, scaler, clf, hmm)
print(classification_report(target_test, predict_results))

# Save models
pickle.dump(scaler, open(path.join(project_dir, "models/scaler.pkl"), "wb"))
pickle.dump(clf, open(path.join(project_dir, "models/clf.pkl"), "wb"))
pickle.dump(hmm, open(path.join(project_dir, "models/hmm.pkl"), "wb"))
