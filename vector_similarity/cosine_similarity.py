from numpy import dot
from numpy.linalg import norm


def cosine_similarity(A, B):
  return dot(A, B)/(norm(A)*norm(B))