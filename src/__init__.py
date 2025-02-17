import os
import pandas as pd
actual_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

os.chdir(actual_directory)
#os.system('cls' if os.name == 'nt' else 'clear')


from data_utils import get_datasets, get_feature_target, get_train_val_sets

if __name__ == "__main__":
    app_train, app_test, columns_description = get_datasets()
    X_train, y_train, X_test, y_test = get_feature_target(app_train, app_test)
    
    print(X_train)
    get_train_val_sets(X_train=X_train, y_train=y_train)
    