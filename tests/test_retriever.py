import numpy as np
from tools import retriever_example


def test_cosine_sim_identical_vectors():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    score = retriever_example.cosine_sim(a, b)
    assert abs(score - 1.0) < 1e-6


def test_cosine_sim_orthogonal_vectors():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    score = retriever_example.cosine_sim(a, b)
    assert abs(score - 0.0) < 1e-6
