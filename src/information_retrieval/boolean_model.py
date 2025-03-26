"""
File for boolean information retrieval
Creates inverse index matrix to find lyrics fullfilling the users query
No query optimization, just logical execution.
"""

import csv
import os
import re
import shutil
from typing import Dict, List, Set, Tuple
from src.information_retrieval.query import Query
from src.information_retrieval.basic_model import BasicModel


class BooleanModel(BasicModel):
    """Class for the boolean model of the information retrieval system"""

    def __init__(self):
        super().__init__()
        self.matrix: Dict[str, Tuple[int, List[int]]] = self.create_matrix()
        self.export_path = "output/boolean_model"

    def create_matrix(self):
        """Create the inverted index matrix"""
        matrix = self.get_matrix_if_stored()
        if matrix != {}:
            return matrix
        
        matrix_unorganized: List[Tuple[str, int]] = self.create_unoptimized_matrix()

        final_matrix = self.order_matrix(matrix_unorganized)

        self.store_matrix(final_matrix)
        self.matrix = final_matrix

        return final_matrix  # just in case the result is needed somewhere else

    def create_unoptimized_matrix(self):
        """Create the unoptimized matrix"""
        matrix_unorganized: List[Tuple[str, int]] = [] 
        filenames: List[str] = []

        print("Creating dictionary... Finding lyric files...\nResults:\n")

        directory = "data/lyrics/"
        for lyric_file in os.listdir(directory):
            filename = lyric_file
            if filename.endswith(".txt"):
                filenames.append(os.path.join(directory, filename))
                self.nr_of_docs += 1

        for fileindex in range(0, len(filenames)):  # one number for each entry
            # print(fileindex)

            with open(filenames[fileindex], "r", encoding="utf-8") as lyric_file:
                for line in lyric_file:
                    words = re.findall(r"\b\w+\b", line.lower())  # Extract words
                    for word in words:
                        # print(word)
                        # print(fileindex)
                        matrix_unorganized.append((word, fileindex))

        return matrix_unorganized

    def order_matrix(self, matrix: List[Tuple[str, int]]):
        """Create the inverted index matrix from the unoptimized matrix"""
        optimized_inverted_matrix: Dict[str, Tuple[int, List[int]]] = {}

        matrix.sort(key=lambda x: x[0])  # Sort by the first element (word)

        for word, index in matrix:
            if word not in optimized_inverted_matrix:
                optimized_inverted_matrix[word] = (0, [])

            freq, filelist = optimized_inverted_matrix[word]

            # Avoid duplicate document indices
            if index not in filelist:
                filelist.append(index)
                freq += 1  

            # Update the dictionary entry
            optimized_inverted_matrix[word] = (freq, filelist)

        return optimized_inverted_matrix

    def store_matrix(self, matrix):
        """Store the inverted index matrix in a CSV file"""
        with open("data/ir/boolean_model.csv", mode='w', newline='', encoding='utf-8') as bool_file:
            csv_writer = csv.writer(bool_file)

            csv_writer.writerow(["Word", "Frequency", "Document IDs"])

            for word, (freq, doc_ids) in matrix.items():
                doc_ids_str = ', '.join(map(str, doc_ids))  # string of fileindexes
                csv_writer.writerow([word, freq, doc_ids_str])

        # print("Matrix exported to data/ir/boolean_model.csv")

    def get_matrix_if_stored(self):
        """Import the inverted index matrix from a CSV file"""
        imported_matrix: Dict[str, Tuple[int, List[int]]] = {}

        try:
            with open("data/ir/boolean_model.csv", mode='r', newline='', encoding='utf-8') as bool_file:
                reader = csv.reader(bool_file)

                if not any(reader):
                    return imported_matrix

                next(reader)

                for row in reader:
                    word = row[0]  # First column: the term
                    frequency = row[1]  # Second column: frequency of word in files
                    fileindexes = row[2]  # Third column: document IDs as a string!
                    fileindex_list = list(map(int, fileindexes.split(', ')))
                    imported_matrix[word] = [frequency, fileindex_list]
                    
                self.nr_of_docs = max(max(imported_matrix.values())[1])
                # print(f"Matrix imported from data/ir/boolean_model.csv with {self.nr_of_docs} documents")

        except FileNotFoundError:
            print("Matrix not existing... Creating new matrix")

        return imported_matrix

    def get_files(self, word: str) -> Set[int]:
        """A helper to extract the file index list of a word"""	
        return set(self.matrix.get(word, (0, []))[1])

    def evaluate_expression(self, tokens: List[str], start: int = 0) -> Tuple[Set[int], int]:
        """Recursively evaluate the expression"""
        result = set()
        operation = None  # last logical operation
        counter = start

        # Recursively processing of the subexpression
        while counter < len(tokens):
            token = tokens[counter]

            # if token == "(":  # leave out parenthesis for now
            #     sub_result, counter = self.evaluate_expression(tokens, counter + 1)
            # elif token == ")":
            #     return result, counter  # End recursion at closing parenthesis
            if token.upper() == "NOT":
                # Apply NOT to the next term
                next_set, counter = self.evaluate_expression(tokens, counter + 1)
                result = result - next_set if operation else next_set
            elif token.upper() == "AND":
                operation = "AND"
            elif token.upper() == "OR":
                operation = "OR"
            else:
                # Normal word!!
                sub_result = self.get_files(token)

                if operation == "AND":
                    result &= sub_result
                elif operation == "OR":
                    result |= sub_result
                else:
                    result = sub_result

            counter += 1

        return result, counter

    def process_query(self, query: Query) -> Set[int]:
        """Process the user query"""
        if not query.tokens:
            print("Tokenization failed\n")
            return set()

        # print("Starting recursive search...")
        result_set, _ = self.evaluate_expression(query.tokens)

        if not result_set:
            print("No results found\n")

        else:
            self.export_results(query, result_set)

        return result_set
    
    def export_results(self, query: Query, result_set: Set[int]):
        """Export the results to output directory"""
        filenames: List[str] = []
        counter = 0

        print("Finding files by index... Result:")

        source_directory = "data/lyrics/"
        dest_directory = os.path.join(self.export_path, query.export())
        os.makedirs(dest_directory, exist_ok=True)

        for filename in os.listdir(source_directory):
            if filename.endswith(".txt") and (counter in result_set):
                filenames.append(filename)
            counter += 1

        for filename in filenames:
            print(filename)

            try:
                source_path = os.path.join(source_directory, filename)
                dest_path = os.path.join(dest_directory, filename)

                shutil.copy2(source_path, dest_path)
                # print(f"Copied: {source_path} → {dest_path}")
                # print(filename)

            except Exception as e:
                print(f"Error copying {source_path}: {e}")

        print("\n")
