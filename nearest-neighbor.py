import numpy as np
import time

global_best_accuracy_so_far = 0

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    global global_best_accuracy_so_far
    features = current_set.copy()
    features.append(feature_to_add)
    number_correct = 0
    
    min_correct = int(np.ceil(global_best_accuracy_so_far * len(data)))

    for i in range(len(data)):
        object_to_classify = data[i,features]
        label_object_to_classify = data[i,0]

        nearest_neighbor_distance = np.inf
        nearest_neighbor_location = np.inf

        for k in range(len(data)):
            if k != i:
                distance = np.sqrt(np.sum((object_to_classify - data[k,features]) ** 2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighor_label = data[nearest_neighbor_location,0]
        
        if label_object_to_classify == nearest_neighor_label:
            number_correct += 1
        
        max_possible = number_correct + (len(data) - i - 1)
        if max_possible < min_correct:
            return -1
        
    return number_correct / len(data)


def forward_selection(data):
    global global_best_accuracy_so_far
    data_content = np.loadtxt(data)
    current_set_of_features = []
    best_accuracy_overall = 0
    best_features_overall = []

    for i in range(1, data_content.shape[1]):
        print(f"On the {i}th level of the search tree.")
        feature_to_add = None
        best_so_far_accuracy = 0

        for k in range(1, data_content.shape[1]):
            if k not in current_set_of_features:
                accuracy = leave_one_out_cross_validation(data_content, current_set_of_features, k)

                if accuracy == -1:
                    print(f"Skipping feature {k}.")
                    continue

                features = current_set_of_features.copy()
                features.append(k)
                print(f"Using feature(s) {set(features)} accuracy is {accuracy * 100:.1f}%")

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add = k
        
        if feature_to_add is not None:
            current_set_of_features.append(feature_to_add)
            print(f"On level {i}, I added feature '{feature_to_add}' to current set.")

            if best_so_far_accuracy > global_best_accuracy_so_far:
                global_best_accuracy_so_far = best_so_far_accuracy
            if best_so_far_accuracy > best_accuracy_overall:
                best_accuracy_overall = best_so_far_accuracy
                best_features_overall = current_set_of_features.copy()
            else:
                print("Accuracy has decreased! Continuing search in case of local maxima.")
        else:
            print(f"No feature improved accuracy at this level.\n")
            break

    print(f"Finished search, best feature subset is {set(best_features_overall)}, which has the accuracy of {best_accuracy_overall * 100:.1f}%")

def backward_elimination(data):
    data_content = np.loadtxt(data)
    num_features = data_content.shape[1] - 1
    current_set_of_features = list(range(1, num_features + 1))
    best_accuracy_overall = 0
    best_features_overall = []

if __name__ == "__main__":
    print("Welcome to Shirley's Feature Selection Algorithm.")
    file = input("Type in the name of the file to test: ")
    print("\nType the number of the algorithm you want to run.\n")
    algorithm = input("1. Forward Selection or 2. Backward Elimination\n")

    if algorithm == '1':
        start_time = time.time()
        forward_selection(file)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total Time: {total_time}")
    # else:
    #     start_time = time.time()
    #     backward_elimination(file)
    #     end_time = time.time()
    #     total_time = end_time - start_time
    #     print(f"Total Time: {total_time}")
