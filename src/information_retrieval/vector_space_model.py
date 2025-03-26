"""
Vector Space Model
-> providing a ranked result

1. term frequency
2. inverse document frequency

-> Combined weight by formulary:
    w = tf * idf
    => w = word frequency in document/ max word frequency in document * log2(Nr of docs/dfi)

Matrix of weights of all words
        doc1  doc2  doc3  . . . 
term1   3     6.1   2.3
term2   5     2     0
term3   7.1   8     1.2
.
.
.

To consider:
1.
    fij = frequency of i in j
    ftij = normalization of term (for example fij/max(fij) or 1+log(fij))
    => relative view, the more often a word in one document the more important
    
2.
    dfi = number of documents containing term
    idfi = inverse document frequency (for example log2(Nr of docs/dfi))  (the more unique the more information)

-> typical combination:
    tfij * idfi (in other words relative high frequency * uniqueness)
    
query processing:
    1. tokenize query
    2. calculate weight of query terms
    3. similarity ranking
    4. return top k documents

similarity ranking:
    cosinus similarity
    cos(weights) = (A*B)/(||A||*||B||)
    => the more similar the higher the value

"""

import csv
import heapq
import math
import os
import shutil
from typing import Dict, List, Set, Tuple
from src.information_retrieval.query import Query
from src.information_retrieval.boolean_model import BooleanModel


class VectorSpaceModel(BooleanModel):
    """Model based on the boolean model to avoid code duplication"""

    def __init__(self):
        self.idf = {}  # Inverse document frequency calculated during matrix creation
        self.matrix = {}  # in this case weights of words in documents
        self.export_path = "output/vector_space_model"
        super().__init__()  # super called last to fill matrix and idf after initialization
        
    def print_query_info(self):
        print("Please enter query (Example: \"i love my life\"): ")

    def create_matrix(self):
        """Load matrices if possible else create the matrix of weights and idf"""
        self.get_matrix_if_stored()  # loads matrix and idf if stored

        if self.matrix and self.idf:
            return self.matrix

        matrix_unorganized: List[Tuple[str, int]] = self.create_unoptimized_matrix()
        matrix_organized = self.order_matrix(matrix_unorganized)
        self.idf = self.compute_inverse_document_frequency(matrix_organized)
        self.store_idf(self.idf)

        self.matrix = self.compute_weight_matrix(matrix_organized)
        self.store_matrix(self.matrix)

        return self.matrix  # just in case the result is needed somewhere else

    def order_matrix(self, matrix: List[Tuple[str, int]]):
        """Orders matrix and adds weight calculation"""
        weight_matrix: Dict[str, Dict[int]] = {}

        matrix.sort(key=lambda x: x[0])

        # inverting and counting how often words appear
        for word, index in matrix:
            if word not in weight_matrix:
                weight_matrix[word] = {index: 1}
            else:
                weight_matrix[word][index] = weight_matrix[word].get(index, 0) + 1

        return weight_matrix  # word: {doc1: number of appearances, doc2: number of appearances, ...}

    def compute_term_frequency(self, ordered_matrix: Dict[str, Dict[int, int]]):
        """Calculate highest term frequency of each document"""
        max_tf: Dict[str, int] = {}

        for _, doc_dict in ordered_matrix.items():
            for doc, frequency in doc_dict.items():
                if doc not in max_tf:
                    max_tf[doc] = 0
                else:
                    max_tf[doc] = max(max_tf[doc], frequency)

        return max_tf  # {doc1: max_tf, doc2: max_tf, ...}

    def compute_inverse_document_frequency(self, ordered_matrix: Dict[str, Dict[int, int]]):
        """Calculate inverse document frequency of each word"""
        idf: Dict[str, float] = {}

        for word, doc_dict in ordered_matrix.items():
            df = len(doc_dict)
            if df == 0:  # shouldn't happen, here to avoid division by zero if it still happened somehow
                continue
            idf[word] = math.log(self.nr_of_docs / df)

        return idf

    def compute_weight_matrix(self, ordered_matrix: Dict[str, Dict[int, int]]):
        """Compute the weight matrix of terms and docs"""
        max_tf: Dict[str, int] = self.compute_term_frequency(ordered_matrix)

        weight_matrix: Dict[str, Dict[int, float]] = {}

        for word, doc_dict in ordered_matrix.items():
            for doc, frequency in doc_dict.items():
                if word not in weight_matrix:
                    weight_matrix[word] = {doc: frequency / max_tf[doc] * self.idf[word]}
                else:
                    weight_matrix[word][doc] = frequency / max_tf[doc] * self.idf[word]

        return weight_matrix  # word: {doc1: weight, doc2: weight, ...}

    def export_results(self, query: Query, result_set: List[Tuple[int, float]]):
        """Copy top found documents to output directory and create a csv with the results for clarity of order"""
        filenames: List[Tuple[int, Tuple[str, int]]] = []
        counter = 0

        print("Finding files by index...\n\nResults:")

        source_directory = "data/lyrics/"
        dest_directory = os.path.join(self.export_path, query.export())
        os.makedirs(dest_directory, exist_ok=True)

        for filename in os.listdir(source_directory):
            if filename.endswith(".txt") and (counter in [doc_id for doc_id, _ in result_set]):
                score = [score for doc_id, score in result_set if doc_id == counter][0]
                rank = [rank for rank, (doc_id, _) in enumerate(result_set) if doc_id == counter][0] + 1
                filenames.append((rank, (filename, score)))
            counter += 1

        # sort by rank
        filenames.sort(key=lambda x: x[0])

        with open(os.path.join(dest_directory, "results_overview.csv"), "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["Rank", "Document", "Score"])
            writer.writeheader()

            for rank, file_with_score in filenames:
                filename, score = file_with_score
                print(str(rank) + ": " + filename)
                writer.writerow({"Rank": rank, "Document": filename, "Score": score})

                try:
                    source_path = os.path.join(source_directory, filename)
                    dest_path = os.path.join(dest_directory, filename)

                    shutil.copy2(source_path, dest_path)

                except Exception as e:
                    print(f"Error copying {source_path}: {e}")

        print("Results can be found in: " + dest_directory + "\n\n")

    def store_matrix(self, matrix: Dict[str, Dict[int, float]]):
        """Store the weight matrix in a csv file to reduce time complexity at later use"""
        with open("data/ir/vector_space_model.csv", mode='w', newline='', encoding='utf-8') as vector_file:  
            writer = csv.writer(vector_file)
            writer.writerow(["Word", "Document", "Weight"])

            for word, doc_dict in matrix.items():
                for doc, weight in doc_dict.items():
                    writer.writerow([word, doc, weight])

        # print("Weight matrix exported to data/ir/vector_space_model.csv")

    def store_idf(self, idf: Dict[str, float]):
        """Store the IDF values in a csv file"""
        try:
            with open("data/ir/idf_vector_space_model.csv", mode='w', newline='', encoding='utf-8') as idf_file:
                writer = csv.writer(idf_file)
                writer.writerow(["Word", "IDF"])

                for word, idf_value in idf.items():
                    writer.writerow([word, idf_value])

            # print("IDF exported to data/ir/idf_vector_space_model.csv")

        except FileNotFoundError:
            print("Something went wrong when storing the idf values")

    def get_matrix_if_stored(self):
        """Import weight matrix and IDF values if stored, else return current weight matrix (should be empty)"""
        try:
            with open("data/ir/idf_vector_space_model.csv", mode='r', newline='', encoding='utf-8') as idf_file:
                reader = csv.reader(idf_file)

                if not any(reader):
                    return self.matrix

                next(reader)

                for row in reader:
                    word = row[0]
                    idf_value = float(row[1])
                    self.idf[word] = idf_value
                
                # print("IDF matrix imported from data/ir/idf_vector_space_model.csv")

            try:
                with open("data/ir/vector_space_model.csv", mode='r', newline='', encoding='utf-8') as idf_file:
                    reader = csv.reader(idf_file)

                    if not any(reader):
                        return self.matrix

                    next(reader)

                    for row in reader:
                        word, doc, weight = row
                        doc = int(doc)
                        weight = float(weight)
                        
                        if word not in self.matrix:
                            self.matrix[word] = {doc: weight}
                        else:
                            self.matrix[word][doc] = weight
                        
                    # update nr_of_docs (max fileindex)
                    self.nr_of_docs = max(max(self.matrix.values(), key=lambda x: max(x.keys())).keys())
                    # print(f"Matrix imported from data/ir/vector_space_model.csv with {self.nr_of_docs} documents")

            except FileNotFoundError:
                # print("Weight matrix not existing... Creating new matrix")
                return self.matrix
                
        except FileNotFoundError:
            # print("IDF matrix not existing... Creating new matrix")
            return self.matrix

        return self.matrix
    
    def compute_doc_lengths(self) -> Dict[int, float]:
        """Compute the Euclidean length of each document vector for cosine normalization."""
        doc_lengths = {}

        for _, doc_weights in self.matrix.items():
            for doc, weight in doc_weights.items():
                if doc not in doc_lengths:
                    doc_lengths[doc] = 0
                doc_lengths[doc] += weight ** 2

        # Square root to get vector norm
        for doc in doc_lengths:
            doc_lengths[doc] = math.sqrt(doc_lengths[doc])

        return doc_lengths
    
    def calculate_query_weights(self, query: List[str]) -> Dict[str, float]:
        """Calculate weights of the query terms"""
        query_tf = {}
        for term in query:
            query_tf[term] = query_tf.get(term, 0) + 1  # Count term frequency

        # Compute querys TF-IDF weights
        for term, tf in query_tf.items():
            query_tf[term] = 1 + math.log(tf)  # log normalization fot term frequency (because setting tf in relation to max tf makes not much sense here, but depending on another query I might change that)
            query_tf[term] *= self.idf.get(term, 0)  # Multiplying with IDF (remember its log2(Nr of docs/dfi))
        return query_tf

    def cosine_score(self, query: List[str], k: int = 10) -> List[Tuple[int, float]]:
        """Compute cosine similarity scores for a query and returns the top K documents."""
        scores = {}  # {doc_id: similarity_score}

        query_weights = self.calculate_query_weights(query)

        for term, w_tq in query_weights.items():
            if term not in self.matrix:
                continue

            for doc_id, w_td in self.matrix[term].items():
                scores[doc_id] = scores.get(doc_id, 0) + w_td * w_tq  # Compute dot product

        # Normalize by document vector lengths
        doc_lengths: Dict[int, float] = self.compute_doc_lengths()
        for doc_id in scores:
            scores[doc_id] /= doc_lengths.get(doc_id, 1)  # Avoid division by zero

        # Get the top K documents
        return heapq.nlargest(k, scores.items(), key=lambda x: x[1])

    def process_query(self, query: Query) -> Set[int]:
        """Process a query with vector space model and return the set of relevant documents."""
        if not query.tokens:
            print("Tokenization failed\n")
            return set()

        # print("Starting vector space model ...")
        result_set = self.cosine_score(query.tokens)

        if not result_set:
            print("No results found\n")

        else:
            self.export_results(query, result_set)

        return result_set