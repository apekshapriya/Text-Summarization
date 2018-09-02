import pickle


def get_train_data():

    with open("../processed_data/X_train.pkl", "r") as inp:
        train_input = pickle.load(inp)
    with open("../processed_data/y_train.pkl", "r") as inp:
        train_labels = pickle.load(inp)

    return train_input, train_labels


def get_val_data():

    with open("../processed_data/X_val.pkl", "r") as inp:
        val_input = pickle.load(inp)
    with open("../processed_data/y_val.pkl", "r") as inp:
        val_labels = pickle.load(inp)

    return val_input, val_labels


def get_test_data():

    with open("../processed_data/X_test.pkl", "r") as inp:
        test_input = pickle.load(inp)
    with open("../processed_data/y_test.pkl", "r") as inp:
        test_labels = pickle.load(inp)

    return test_input, test_labels


def get_embeddings():

    with open("../processed_data/embedding_vector.pkl", "r") as inp:
        embedding = pickle.load(inp)

    return embedding
