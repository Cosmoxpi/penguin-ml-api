import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.models.data_loader import load_data
from src.models.preprocess import preprocess

def evaluate(model_path: str="artifacts/model.pkl"):
    """
    Evaluate the saved model on the test dataset.
    Prints accuracy and classification report.
    """

    # 1.Load full dataset
    df=load_data()

    # 2. Preprocess to get features and target
    X,y=preprocess(df)

    # 3. Same split as train.py for consistency
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Load trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 5. Predict & evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Model Evaluation Complete")
    print(f"Test Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    evaluate()


