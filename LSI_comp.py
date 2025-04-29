"""File that runns the entire IR"""

import csv
from src.information_retrieval.query import Query
from src.information_retrieval.vector_space_model import VectorSpaceModel
from src.information_retrieval.lsi_model import LSIModel


queries = ["love", "summer", "sound of the end of the world", "i love you and i miss you", "remind me of memories"]
nr_of_docs = [124, 11, 43, 41, 32]
counter = 0
results = []

for query in queries:

    vec_mod = VectorSpaceModel(k_numbers=nr_of_docs[counter])
    ver_res = vec_mod.process_query(query = Query(query))
    print("k number VSM: ", vec_mod.k_numbers)

    lsi_mod = LSIModel(k_numbers=nr_of_docs[counter])
    lsi_res = lsi_mod.process_query(query = Query(query))

    results.append((query, ver_res, lsi_res))
    counter += 1

headers = ["artist_title.txt",
        "love - expected", "love - VSM", "love - LSI",
        "summer - expected", "summer - VSM", "summer - LSI",
        "sound of the end of the world - expected", "sound of the end of the world - VSM", "sound of the end of the world - LSI",
        "i love you and i miss you - expected", "i love you and i miss you - VSM", "i love you and i miss you - LSI",
        "remind me of memories - expected", "remind me of memories - VSM", "remind me of memories - LSI"]

results_sum = {"artist_title.txt": "error.txt",
                "love - expected": "error.txt", "love - VSM": "error.txt", "love - LSI": "error.txt",
                "summer - expected": "error.txt", "summer - VSM": "error.txt", "summer - LSI": "error.txt",
                "sound of the end of the world - expected": "error.txt", 
                "sound of the end of the world - VSM": "error.txt", 
                "sound of the end of the world - LSI": "error.txt",
                'i love you and i miss you - expected': 'error.txt', 
                'i love you and i miss you - VSM': 'error.txt', 
                'i love you and i miss you - LSI': 'error.txt',
                'remind me of memories - expected': 'error.txt', 
                'remind me of memories - VSM': 'error.txt', 
                'remind me of memories - LSI': 'error.txt'}
    
with open("evaluation/lsi_eval_blank.csv", "r") as prev_file:
    dict_reader = csv.DictReader(prev_file, delimiter=";")

    with open("evaluation/lsi_eval.csv", "w", newline="") as new_file:
        dict_writer = csv.DictWriter(new_file, fieldnames=headers, delimiter=";")

        dict_writer.writeheader()
        row_id_nr = 0

        for read_row in dict_reader:
            artist_title = read_row["artist_title.txt"]

            results_sum["artist_title.txt"] = artist_title
            results_sum["love - expected"] = read_row["love - expected"]
            results_sum["summer - expected"] = read_row["summer - expected"]
            results_sum["sound of the end of the world - expected"] = read_row["sound of the end of the world - expected"]
            results_sum["i love you and i miss you - expected"] = read_row["i love you and i miss you - expected"]
            results_sum["remind me of memories - expected"] = read_row["remind me of memories - expected"]

            for result in results:
                query, lsi_res, ver_res = result

                results_sum[f"{query} - VSM"] = 0
                results_sum[f"{query} - LSI"] = 0
            
                for doc_id, score in ver_res:
                    if row_id_nr == doc_id:
                        results_sum[f"{query} - VSM"] = 1  # alternative: score

                for doc_id, score in lsi_res:
                    if row_id_nr == doc_id:
                        results_sum[f"{query} - LSI"] = 1  # score

            dict_writer.writerow(results_sum)
            row_id_nr += 1
