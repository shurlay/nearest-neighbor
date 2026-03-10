import numpy as np

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    number_correct = 0
    
    for i in range(len(data)):
        object_to_classify = data[i,1:]
        label_object_to_classify = data[i,0]

        nearest_neighbor_distance = np.inf
        nearest_neighbor_location = np.inf

        for k in range(len(data)):
            if k != i:
                distance = np.sqrt(np.sum((object_to_classify - data[k,1:]) ** 2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighor_label = data[nearest_neighbor_location,0]
        
        if label_object_to_classify == nearest_neighor_label:
            number_correct += 1
        
    return number_correct / len(data)


def forward_selection(data):
    data_content = np.loadtxt(data)
    current_set_of_features = []

    for i in range(len(data_content[0]) - 1):
        print(f"On the {i}th level of the search tree.")
        feature_to_add = None
        best_so_far_accuracy = 0

        for k in range(len(data_content[0]) - 1):
            if k not in current_set_of_features:
                accuracy = leave_one_out_cross_validation(data_content, current_set_of_features, k+1)
                print(f"Considering adding the '{k}' feature with {accuracy * 100}% accuracy.")
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add = k
        
        if (feature_to_add is not None):
            current_set_of_features.append(feature_to_add)
            print(f"On level {i}, I added feature '{feature_to_add}' to current set.")

if __name__ == "__main__":
    print("Welcome to Shirley's Feature Selection Algorithm.")
    file = input("Type in the name of the file to test: ")
    print("\nType the number of the algorithm you want to run.\n\n")
    algorithm = input("1. Forward Selection or 2. Backward Elimination")

    if algorithm == '1':
        forward_selection(file)
    # else:
    #     backward_elimination(file)
