import numpy as np
import time

# all commented code is for early abandonment
global_best_accuracy_so_far = 0

# referenced Dr. Eamonn Keogh's pseudocode
def leave_one_out_cross_validation(data, current_set, features):
    global global_best_accuracy_so_far
    number_correct = 0
    
    # min_correct = int(np.ceil(global_best_accuracy_so_far * len(data)))

    for i in range(len(data)):
        object_to_classify = data[i,features]
        label_object_to_classify = data[i,0]

        nearest_neighbor_distance = np.inf
        nearest_neighbor_location = np.inf

        for k in range(len(data)):
            # ensure we are not comparing distance to yourself
            if k != i:
                distance = np.sqrt(np.sum((object_to_classify - data[k,features]) ** 2))
                # update nearest_neighbor distance
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location,0]
        
        if label_object_to_classify == nearest_neighbor_label:
            number_correct += 1
        
        # max_possible = number_correct + (len(data) - i - 1)
        # if max_possible < min_correct:
        #     return -1
    
    #calculate accuracy
    return number_correct / len(data)


def forward_selection(data):
    global global_best_accuracy_so_far
    data_content = np.loadtxt(data)
    output_file = open("forward_results.txt", "w") # output file to organize data to plot graph
    current_set_of_features = []
    best_accuracy_overall = 0
    best_features_overall = []

    for i in range(1, data_content.shape[1]):
        print(f"On the {i}th level of the search tree.")
        feature_to_add = None
        best_so_far_accuracy = 0

        for k in range(1, data_content.shape[1]):
            # ensure the feature is not in the current set of features, so it can be appended
            if k not in current_set_of_features:
                features = current_set_of_features.copy()
                features.append(k)
                accuracy = leave_one_out_cross_validation(data_content, current_set_of_features, features)

                # if accuracy == -1:
                #     print(f"Skipping feature {k}.")
                #     continue

                print(f"Using feature(s) {features} accuracy is {accuracy * 100:.1f}%")
                output_file.write(f"{features},{accuracy * 100:.1f}\n") # write subsets and accuracies to file

                # update best so far accuracy
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add = k
        
        # append feature if it isn't None
        if feature_to_add is not None:
            current_set_of_features.append(feature_to_add)
            print(f"On level {i}, I added feature '{feature_to_add}' to current set.")

            # update best overall features and accuracy
            if best_so_far_accuracy > global_best_accuracy_so_far:
                global_best_accuracy_so_far = best_so_far_accuracy
                best_features_overall = current_set_of_features.copy()
                best_accuracy_overall = best_so_far_accuracy
            else:
                print("Accuracy has decreased! Continuing search in case of local maxima.")
        # else:
        #     print(f"No feature improved accuracy at this level.\n")
        #     break

    print(f"Finished search, best feature subset is {best_features_overall}, which has the accuracy of {best_accuracy_overall * 100:.1f}%")
    output_file.close()

def backward_elimination(data):
    # global global_best_accuracy_so_far
    data_content = np.loadtxt(data)
    output_file = open("backward_results.txt", "w") #output file to organize data to plot graph
    current_set_of_features = list(range(1, data_content.shape[1]))
    best_accuracy_overall = 0
    best_features_overall = []

    for i in range(1, data_content.shape[1]):
        print(f"On the {i}th level of the search tree.")
        feature_to_remove = None
        best_so_far_accuracy = 0

        for k in range(1, data_content.shape[1]):
            # ensure the feature is in the current set of features, so it can be removed
            if k in current_set_of_features:
                features = current_set_of_features.copy()
                features.remove(k)

                if not features:
                    continue

                accuracy = leave_one_out_cross_validation(data_content, current_set_of_features, features)

                # if accuracy == -1:
                #     print(f"Skipping feature {k}.")
                #     continue

                print(f"Using feature(s) {features} accuracy is {accuracy * 100:.1f}%")
                output_file.write(f"{features},{accuracy * 100:.1f}\n")  # write subsets and accuracies to file

                # update best so far accuracy
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_remove = k
        
        # remove feature if it isn't None
        if feature_to_remove is not None:
            current_set_of_features.remove(feature_to_remove)
            print(f"On level {i}, I removed feature '{feature_to_remove}' in current set.")

            # update best overall features and accuracy
            if best_so_far_accuracy > global_best_accuracy_so_far:
                global_best_accuracy_so_far = best_so_far_accuracy
                best_features_overall = current_set_of_features.copy()
                best_accuracy_overall = best_so_far_accuracy
            else:
                print("Accuracy has decreased! Continuing search in case of local maxima.")
        # else:
        #     print(f"No feature improved accuracy at this level.\n")
        #     break

    print(f"Finished search, best feature subset is {best_features_overall}, which has the accuracy of {best_accuracy_overall * 100:.1f}%")
    output_file.close()

if __name__ == "__main__":
    print("Welcome to Shirley's Feature Selection Algorithm.")
    file = input("Type in the name of the file to test: ")
    print("\nType the number of the algorithm you want to run.")
    algorithm = input("1. Forward Selection or 2. Backward Elimination\n")

    if algorithm == '1':
        start_time = time.time()
        forward_selection(file)
        end_time = time.time()
        total_time = end_time - start_time # calculate total time
        print(f"Total Time: {total_time}")
    if algorithm == '2':
        start_time = time.time()
        backward_elimination(file)
        end_time = time.time()
        total_time = end_time - start_time # calculate total time
        print(f"Total Time: {total_time}")
