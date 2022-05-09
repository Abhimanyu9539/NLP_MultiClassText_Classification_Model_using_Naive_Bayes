import os
import config
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer

from Source.utils import save_file


sw = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')


def process_text(file_path):
    print("Reading input data...")
    data = pd.read_csv(file_path)
    # Select text column and label column
    data = data[[config.label_col, config.comp_col]]
    # Drop rows with null values
    data.dropna(inplace=True)
    # Rename text column name
    data.rename({config.comp_col: "Complaint"}, axis=1, inplace=True)
    # Map products to common name
    data.replace({"Product": config.product_map}, inplace=True)
    # Select complaints as list
    data=data[1:10000]
    complaints = list(data["Complaint"])
    # Convert to lower case
    print("Converting text to lower case...")
    complaints = [c.lower() for c in tqdm(complaints)]
    print("Tokenizing the text...")
    tokens = [word_tokenize(r) for r in tqdm(complaints)]
    print("Removing stop words...")
    tokens = [[word for word in t if word not in sw] for t in tqdm(tokens)]
    print("Removing punctuations...")
    tokens = [["".join(tokenizer.tokenize(word)) for word in t
               if len(tokenizer.tokenize(word)) > 0] for t in tqdm(tokens)]
    print("Removing 'xxxx' and '000' tokens...")
    tokens = [[t for t in token if t not in ['xxxx', '000']] for token in tqdm(tokens)]
    print("Joining tokens...")
    clean_complaints = [" ".join(complaint) for complaint in tqdm(tokens)]
    print("vectorizing the data...")
    vect = CountVectorizer(min_df=config.min_df)
    X = vect.fit_transform(clean_complaints)
    y = data[config.label_col]
    print("Split the data into train and test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y,
                                                        random_state=config.random_state)
    print("Saving the files...")
    save_file(os.path.join(config.output_folder, config.vect_file), vect)
    return X_train, X_test, y_train, y_test
