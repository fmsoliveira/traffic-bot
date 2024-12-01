import joblib
import glob

import regression as sl
import temporal_series_supervised as tss
import classification



def get_best_model_regression():
    best_name, best_model = sl.train_model_regression()
    print("Main: ", best_name)

    # save supervised learning best model as pkl file
    supervised_learning_filename = f"{best_name}_regression.pkl"

    # serialize the model for later use
    joblib.dump(best_model, supervised_learning_filename)

def get_best_model_classification():
    print("Starting classification algorithms")
    classification.train_model_classification()

def get_best_model_spatial_clustering():
    print("Starting spatial clustering")
    #spatial_clustering.train_model_spatial_clustering()

def get_best_model_k_means():
    #kmeans.train_model()
    pass

if __name__ == "__main__":

    # Match any file containing "part_of_name" in its name
    matching_files_regression = glob.glob("*_regression.pkl")
    matching_files_classification = glob.glob("*_classification.pkl")
    #matching_files_complexity = glob.glob("*_spatial_clustering.pkl")
    #matching_files_k_means = glob.glob("*_k_means.pkl")

    if not matching_files_regression:
        print("Supervised learning file does not exist...")
        get_best_model_regression()

    if not matching_files_classification:
        print("Classification file does not exist...")
        get_best_model_classification()

    #if not matching_files_complexity:
    #    print("Spatial Clustering file does not exist...")
    #    get_best_model_spatial_clustering()

    #if not matching_files_k_means():
    #    get_best_model_k_means()
    else:
        # do something
        print("Supervided AI models already trained")
        print("Check the web app to use them")