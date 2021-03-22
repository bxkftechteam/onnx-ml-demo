#!/usr/bin/env python3

import sys
import pickle
import numpy as np


def log_mask_zero(a):
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)


def _viterbi(log_startprob,     # n_components items
             log_transmat,      # n_components * n_components matrix
             framelogprob):     # n_samples * n_components array

    n_samples, n_components = framelogprob.shape
    state_sequence = np.empty(n_samples, dtype=np.int32)
    viterbi_lattice = np.zeros((n_samples, n_components), dtype=np.float32)
    work_buffer = np.empty(n_components, dtype=np.float32)

    viterbi_lattice[0, :] = log_startprob + framelogprob[0, :]

    # Induction
    for t in range(1, n_samples):
        for i in range(n_components):
            work_buffer = log_transmat[:, i] + viterbi_lattice[t - 1, :]
            viterbi_lattice[t, i] = np.max(work_buffer) + framelogprob[t, i]

    # Observation traceback
    state_sequence[n_samples - 1] = where_from = \
        np.argmax(viterbi_lattice[n_samples - 1])
    logprob = viterbi_lattice[n_samples - 1, where_from]

    for t in range(n_samples - 2, -1, -1):
        work_buffer = viterbi_lattice[t, :] + log_transmat[:, where_from]
        state_sequence[t] = where_from = np.argmax(work_buffer)

    return np.asarray(state_sequence), logprob


def _decode_viterbi(log_emissionprob, log_startprob, log_transmat, X):
    framelogprob = log_emissionprob[:, np.concatenate(X)].T.astype(np.float32)
    return _viterbi(log_startprob, log_transmat, framelogprob)


def predict(hmm_model, X):
    log_emissionprob = log_mask_zero(hmm_model.emissionprob_).astype(np.float32)
    log_startprob = log_mask_zero(hmm_model.startprob_).astype(np.float32)
    log_transmat = log_mask_zero(hmm_model.transmat_).astype(np.float32)
    pred, _ = _decode_viterbi(log_emissionprob, log_startprob, log_transmat, X)
    return pred


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: {} <model_path> <N>".format(sys.argv[0]))
        exit(1)
    hmm_model_path = sys.argv[1]
    N = int(sys.argv[2])
    features = np.random.randint(0, 6, (N, 1))
    hmm_model = pickle.load(open(hmm_model_path, "rb"))
    res_hmmlearn = hmm_model.predict(features)
    res_predict = predict(hmm_model, features)
    print("hmmlearn:", res_hmmlearn)
    print("predict:", res_predict)
    print("is equal:", np.array_equal(res_hmmlearn, res_predict))
