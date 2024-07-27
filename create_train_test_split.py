import os
import random
import argparse

def split_images(input_folder, test_percentage=15):
    # Get a list of all files in the folder
    all_files = os.listdir(os.path.join(input_folder, "images"))
    # Filter out non-image files (you can customize this if needed)
    image_files = [f for f in all_files if f.lower().endswith('.jpg')]
    image_files.sort()

    # Calculate the number of images for testing
    num_test_images = int(len(image_files) * test_percentage / 100)

    # Randomly select test images
    test_images = random.sample(image_files, num_test_images)

    # Create lists for training and testing images
    train_images = list(set(image_files) - set(test_images))

    # Write selected test images to test.txt
    with open(os.path.join(input_folder, 'test.txt'), 'w') as test_file:
        for image in test_images:
            test_file.write(image.split(".")[0] + '\n')

    # Write remaining images to train.txt
    with open(os.path.join(input_folder, 'train.txt'), 'w') as train_file:
        for image in train_images:
            train_file.write(image.split(".")[0] + '\n')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Split images into training and testing sets.')
    parser.add_argument('base_folder', type=str, help='Path to the base folder containing image datasets.')
    parser.add_argument('--test_percentage', type=int, default=15, help='Percentage of images to be used for testing.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Iterate over each item in the base folder
    for item in os.listdir(args.base_folder):
        # Create the full path to the item
        item_path = os.path.join(args.base_folder, item)
        # Check if the item is a folder
        if os.path.isdir(item_path):
            print(f"Folder: {item_path}")
            # Split images within the folder
            split_images(item_path, args.test_percentage)

if __name__ == "__main__":
    main()
