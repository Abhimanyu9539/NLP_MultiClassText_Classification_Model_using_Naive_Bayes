import os
import config
import argparse
from Source.utils import load_file

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer


sw = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')


def main(args):
    # Load the vectorizer and the model
    vect = load_file(os.path.join(config.output_folder, config.vect_file))
    model = load_file(os.path.join(config.output_folder, config.model_file))
    test_complaints = [args.test_complaint]
    # Convert text to lower case
    test_complaints = [r.lower() for r in test_complaints]
    # Tokenize the text
    test_tokens = [word_tokenize(r) for r in test_complaints]
    # Remove stop words
    test_tokens = [[word for word in t if word not in sw] for t in test_tokens]
    # Remove punctuations
    test_tokens = [["".join(tokenizer.tokenize(word)) for word in t
                    if len(tokenizer.tokenize(word)) > 0] for t in test_tokens]
    # Remove 'xxxx' and '000'
    test_tokens = [[t for t in token if t not in ['xxxx', '000']] for token in test_tokens]
    # Join tokens
    clean_test_complaints = [" ".join(complaint) for complaint in test_tokens]
    # Vectorize the tokens
    X_test = vect.transform(clean_test_complaints)
    # Make predictions
    test_prediction = model.predict(X_test)[0]
    print(f"Prediction: {test_prediction}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_complaint", type=str, help="Input file name")
    args = parser.parse_args()
    main(args)