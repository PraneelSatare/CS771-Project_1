# --------------------------------
# File Paths for Datasets
# --------------------------------

def get_dataset_paths():
    train_path_emoticon = input("Please provide the path to the training dataset for emoticon dataset: ")
    test_path_emoticon = input("Please provide the path to the testing dataset for emoticon dataset: ")
    test_path_deep = input("Please provide the path to the testing dataset for deep features dataset: ")
    test_path_text = input("Please provide the path to the testing dataset for text sequences dataset: ")
    
    return train_path_emoticon, test_path_emoticon, test_path_deep, test_path_text