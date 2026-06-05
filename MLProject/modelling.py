import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model_ci():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "loan_clean.csv")
    
    print("Memuat dataset untuk Re-training...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Melatih model di server GitHub Actions...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    output_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    model_file_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(model, model_file_path)
    
    print(f"Model berhasil dilatih dan disimpan di: {model_file_path}")

if __name__ == "__main__":
    train_model_ci()