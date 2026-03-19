import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.models.data_loader import load_data
from src.models.preprocess import preprocess

def train():
    # 1. Load the dataset
    df=load_data()

    # 2. Preprocess -> get X(features) and y(target)
    X,y=preprocess(df)

    # 3. Train-Test split
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    # 4. Define model
    model=RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    # 5. Train
    model.fit(X_train,y_train)

    # 6. Evaluate
    preds=model.predict(X_test)
    acc=accuracy_score(y_test,preds)
    print(f"Model trained successfully.Accuracy={acc:.4f}")

    # 7. Save model to artifacts/
    model_path="artifacts/model.pkl"
    with open(model_path,"wb") as f:
        pickle.dump(model,f)

    print(f"Model saved to {model_path}")

if __name__=="__main__":
    train()
