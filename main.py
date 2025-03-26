"""File that runns the entire IR"""

from src.information_retrieval.vector_space_model import VectorSpaceModel
from src.information_retrieval.boolean_model import BooleanModel

menu_index = 0

while (menu_index != 3):
    print("Welcome to the Information Retrieval System!\n")
    print("Choose your option:")
    print("1. Boolean Model")
    print("2. Vector Space Model")
    print("3. Exit\n")

    try:
        menu_index = int(input())
    except ValueError:
        print("INVALID INPUT, choose any number between 1-3!\n")
        menu_index = 0

    match menu_index:
        case 1:
            bool_mod = BooleanModel()
            bool_mod.process_query(bool_mod.get_user_query())
        case 2:
            vec_mod = VectorSpaceModel()
            vec_mod.process_query(vec_mod.get_user_query())
