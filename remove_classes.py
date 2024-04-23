import os
import shutil
import random
import json
import argparse

def remove_classes(input_dir, output_dir, percentage):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the list of class directories in the input directory
    train_input_dir = os.path.join(input_dir, "train")
    test_input_dir = os.path.join(input_dir, "val")
    train_output_dir = os.path.join(output_dir, "train")
    test_output_dir = os.path.join(output_dir, "val")
    
    # Create output directories for train and test
    os.makedirs(train_output_dir)
    os.makedirs(test_output_dir)

    # Get the list of class directories in the input training directory
    train_class_dirs = [d for d in os.listdir(train_input_dir) if os.path.isdir(os.path.join(train_input_dir, d))]

    # Determine the number of classes to remove
    num_classes_to_remove = int(len(train_class_dirs) * percentage / 100)
    
    # Randomly select classes to remove
    classes_to_remove = random.sample(train_class_dirs, num_classes_to_remove)
    #classes_to_remove = ["tench", "English springer", "chain saw"]
    known_classes = [d for d in train_class_dirs if d not in classes_to_remove]
    # Save the list of removed classes to a file
    with open(os.path.join(output_dir, "known_unknown_classes.json"), "w") as f:
        json.dump({"known": known_classes, "unknown": classes_to_remove}, f)


    # Create dir for removed classes
    os.makedirs(os.path.join(output_dir, "unknown_classes"))

    # Copy the remaining training classes to the output directory
    for class_dir in train_class_dirs:
        if class_dir not in classes_to_remove:
            shutil.copytree(os.path.join(train_input_dir, class_dir), os.path.join(train_output_dir, class_dir))
        else:
            shutil.copytree(os.path.join(train_input_dir, class_dir), os.path.join(output_dir, "unknown_classes", class_dir))


    # Copy the remaining test classes to the output directory
    test_class_dirs = [d for d in os.listdir(test_input_dir) if os.path.isdir(os.path.join(test_input_dir, d))]
    for class_dir in test_class_dirs:
        if class_dir not in classes_to_remove:
            destination_dir = os.path.join(test_output_dir, class_dir)
            if os.path.exists(destination_dir):
                # Append images to the existing destination folder
                for file in os.listdir(os.path.join(test_input_dir, class_dir)):
                    shutil.copy(os.path.join(test_input_dir, class_dir, file), destination_dir)
            else:
                shutil.copytree(os.path.join(test_input_dir, class_dir), destination_dir)
        else:
            # Check if destination directory exists
            destination_dir = os.path.join(output_dir, "unknown_classes", class_dir)
            if os.path.exists(destination_dir):
                # Append images to the existing destination folder
                for file in os.listdir(os.path.join(test_input_dir, class_dir)):
                    shutil.copy(os.path.join(test_input_dir, class_dir, file), destination_dir)
            else:
                shutil.copytree(os.path.join(test_input_dir, class_dir), destination_dir)

if __name__ == "__main__":
    random.seed(1234)

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory")
    args = parser.parse_args()
    #input_dir = "datasets/tiny-imagenet-200"
    percentage_to_remove = 10  # Change this value to the desired percentage

    output_dir = f"{args.input_dir}_{100-percentage_to_remove}"

    remove_classes(args.input_dir, output_dir, percentage_to_remove)
