import joblib
import os
import glob

import regression as sl
import temporal_series_supervised as tss
import spatial_clustering
import kmeans



def get_best_model_supervised_learning():
    best_name, best_model = sl.train_model_supervised_learning()
    print("Main: ", best_name)

    # save supervised learning best model as pkl file
    supervised_learning_filename = f"{best_name}_supervised.pkl"

    # serialize the model for later use
    joblib.dump(best_model, supervised_learning_filename)

def get_best_model_time_series():
    #tss.train_model_time_series()
    pass

def get_best_model_spatial_clustering():
    print("Starting spatial clustering")
    spatial_clustering.train_model_spatial_clustering()

def get_best_model_k_means():
    kmeans.train_model()

if __name__ == "__main__":

    # Match any file containing "part_of_name" in its name
    matching_files_supervised = glob.glob("*_supervised.pkl")
    matching_files_temporal_series = glob.glob("*_temp_series.pkl")
    matching_files_complexity = glob.glob("*_spatial_clustering.pkl")
    matching_files_k_means = glob.glob("*_k_means.pkl")

    if not matching_files_supervised:
        print("Supervised learning file does not exist...")
        get_best_model_supervised_learning()

    # if not matching_files_temporal_series:
    #    print("Temporal Series file does not exist...")
    #    get_best_model_time_series()

    #if not matching_files_complexity:
    #    print("Spatial Clustering file does not exist...")
    #    get_best_model_spatial_clustering()

    #if not matching_files_k_means():
    #    get_best_model_k_means()
    else:
        # do something
        print("Supervided AI models already trained")