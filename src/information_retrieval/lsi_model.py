"""
Latent Semantic Indexing Model

1. term-document matrix
Terms | doc1 | ... | docN
2. A = U * S * V^T (decompose A)
3. Rank 2 Approximation: first two columns of U & V and first two columns and rows of S
4. Document vector coordinates in reduced 2D space
5. Query vector coordinates q=q^T * Uk * Sk^(-1)
6. Rank documents decreasing by cosine similarity sim(q, d) = (q*d) / (|q| * |d|)

This model was created and debugged with help of ChatGPT.
"""
import heapq
import numpy as np
from numpy.linalg import svd

from typing import List, Tuple
from src.information_retrieval.vector_space_model import VectorSpaceModel

class LSIModel(VectorSpaceModel):
    def __init__(self, k_numbers: int = 10):
        """
        LSIModel based on the VectorSpaceModel
        """
        self.k_dim = 150
        self.U_matrix = None
        self.S_matrix = None
        self.Vt_matrix = None
        self.term_index = {}  # term -> index mapping
        self.doc_index = {}  # doc -> index mapping
        self.reduced_doc_vectors = None
        super().__init__()
        self.export_path = "output\\lsi_model"
        self.k_numbers = k_numbers  # limited number of docs being returned

    def reduce_dimensionality(self):
        """
        Apply SVD on the TF-IDF weight matrix and reduce dimensions
        """
        terms = list(self.matrix.keys())
        docs = sorted({doc for doc_weights in self.matrix.values() for doc in doc_weights})

        self.term_index = {term: i for i, term in enumerate(terms)}
        self.doc_index = {doc: i for i, doc in enumerate(docs)}

        # Create full TF-IDF matrix (terms x docs)
        A = np.zeros((len(terms), len(docs)))

        for term, doc_weights in self.matrix.items():
            for doc, weight in doc_weights.items():
                A[self.term_index[term], self.doc_index[doc]] = weight

        # Apply SVD
        self.U_matrix, S, self.Vt_matrix = svd(A, full_matrices=False)
        self.S_matrix = np.diag(S[:self.k_dim])
        self.U_matrix = self.U_matrix[:, :self.k_dim]
        self.Vt_matrix = self.Vt_matrix[:self.k_dim, :]

        # Precompute reduced document vectors (docs x k)
        self.reduced_doc_vectors = self.S_matrix @ self.Vt_matrix

    def transform_query(self, query: List[str]) -> np.ndarray:
        """
        Project the query vector into the latent semantic space
        """
        query_weights = self.calculate_query_weights(query)
        q_vec = np.zeros((len(self.term_index),))

        for term, weight in query_weights.items():
            if term in self.term_index:
                q_vec[self.term_index[term]] = weight

        # Project into reduced space: q' = (q^T * U) * S^-1
        q_reduced = (q_vec @ self.U_matrix) @ np.linalg.inv(self.S_matrix)
        return q_reduced

    def cosine_score(self, query: List[str]) -> List[Tuple[int, float]]:
        """
        Overridden cosine_score method to use reduced latent vectors
        """
        if self.U_matrix is None:
            self.reduce_dimensionality()

        q_vec = self.transform_query(query)

        scores = {}
        for doc, index in self.doc_index.items():
            doc_vec = self.reduced_doc_vectors[:, index]  # reduced vector of doc
            numerator = np.dot(q_vec, doc_vec)
            denominator = np.linalg.norm(q_vec) * np.linalg.norm(doc_vec)  # cosine similarity: |q| * |d|
            if denominator != 0:
                scores[doc] = numerator / denominator
            else:
                scores[doc] = 0

        return heapq.nlargest(self.k_numbers, scores.items(), key=lambda x: x[1])
