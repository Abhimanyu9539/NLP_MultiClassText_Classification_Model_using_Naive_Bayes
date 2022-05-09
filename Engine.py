import os
import config
import argparse
from Source.model import train_model
from Source.peprocessing import process_text

def main(args_):
    # Create file path
    file_path = os.path.join(args_.input_path, args_.file_name)
    # Process the data
    X_train, X_test, y_train, y_test = process_text(file_path)
    # Train the model
    test_accuracy = train_model(X_train, X_test, y_train, y_test)
    print(f"Test accuracy: {test_accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default=config.file_name,
                        help="Input file name")
    parser.add_argument("--input_path", type=str, default=config.input_folder,
                        help="Input folder name")
    args = parser.parse_args()
    main(args)
