import os
import config
from Source.utils import save_file
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


def train_model(X_train, X_test, y_train, y_test):
    model = MultinomialNB()
    print("Training the model...")
    model.fit(X_train, y_train)
    # Save model object
    save_file(os.path.join(config.output_folder, config.model_file), model)
    # Predict on test set
    test_pred = model.predict(X_test)
    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test, test_pred)
    return test_accuracy
