import numpy as np

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    data_content = np.loadtxt(data)
    number_correct = 0
    
    for i in range(len(data_content)):
        object_to_classify = data_content[i,2:]
        label_object_to_classify = data_content[i,0]

        nearest_neighbor_distance = inf
        nearest_neighbor_location = inf

        for k in range(len(data_content)):
            if k is not i:
                distance = np.sqrt(np.sum((object_to_classify - data_content[k,2:]) ** 2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighor_label = data_content[nearest_neighbor_location,0]
        
        if label_object_to_classify is nearest_neighor_label:
            number_correct += 1
        
    return number_correct / len(data_content)


def forward_selection(data):
    data_content = np.loadtxt(data)
    current_set_of_features = []

    for i in range(len(data_content[0]) - 1):
        print(f"On the {i}th level of the search tree.")
        feature_to_add = []
        best_so_far_accuracy = 0

        for k in range(len(data_content[0]) - 1):
            if k not in current_set_of_features:
                print(f"Considering adding the '{k}' feature.")
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k+1)

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add = k
        
        current_set_of_features(i) = feature_to_add
        print(f"On level {i}, I added feature '{feature_to_add}' to current set.")
