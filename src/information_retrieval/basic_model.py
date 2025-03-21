"""File for the basic Model taking user requests to start any IR System"""

from src.information_retrieval.query import Query


class BasicModel():
    
    def __init__(self):
        pass

    def get_user_query(self):
        print("Please enter query (Example: love AND fire): ")
        query: Query = Query(input())
        if not self.check_query(query):
            print("Please enter query. No special characters allowed: ")
            query = Query(input())
        return query

    def check_query(self, query: Query):
        print(f"Checking query: {query} (Type: {type(query)})")
        
        for one_character in query:
            # print(f"Character: {one_character}")  # Debugging print
            if not (one_character.isalpha() or
                    one_character.isdigit() or
                    one_character in [' ', '(', ')']):
                print("Invalid character found: " + one_character)
                return False
        return True
    
    def check_tokens_query(self, query: Query):
        """Stores query parameters, thought i needed it first but now not in use, maybe later"""
        tokens = query.tokens

        # not supposed to start with AND or OR
        if tokens[0] in ["AND", "OR"]:
            return False

        for i in len(tokens):
            if tokens[i] == "AND":
                query.ands.append(i)
            elif tokens[i] == "OR":
                query.ors.append(i)
            elif tokens[i] == "NOT":
                query.nots.append(i)
            else:  # everything else is a searchterm
                query.terms.append(i)
