#!/usr/bin/env python3

"""Run inference using phone-play models"""

from os import path
import pickle
import numpy as np

from map_pred import map_pred


def infer(X, scaler, clf, hmm):
    X = scaler.transform(X)
    pred_probs = clf.predict_proba(X)[:, 1]
    pred_labels = np.array([map_pred(x) for x in pred_probs], dtype=np.int64)
    return hmm.predict(pred_labels.reshape(-1, 1))


if __name__ == '__main__':
    project_dir = path.dirname(path.dirname(path.abspath(__file__)))
    scaler = pickle.load(open(path.join(project_dir, "models/scaler.pkl"), "rb"))
    clf = pickle.load(open(path.join(project_dir, "models/clf.pkl"), "rb"))
    hmm = pickle.load(open(path.join(project_dir, "models/hmm.pkl"), "rb"))
    dataset = pickle.load(open(path.join(project_dir, "data/synthetic.npy"), "rb"))
    features = np.vstack([d["X"] for d in dataset])
    print(infer(features, scaler, clf, hmm))
