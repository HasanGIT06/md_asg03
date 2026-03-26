import pandas as pd
from feature_engineering.feature_engineering import feature_engineering
from data.data_ingestion import ingest_data
from preprocessing.preprocessing import preprocess_data
from models.train import train
from models.evaluation import evaluate
from pipeline.sklearn_pipeline import build_churn_pipeline

def run_pipeline():
    print("=" * 50)
    print("Step 1: Data Ingestion")
    ingest_data()
    df = pd.read_csv("D:/University/Semester 4/Model Deployment/ASG Session 5/src/data/ingested/train.csv")
    
    print("Step 2: Feature Engineering")
    df = feature_engineering(df)
    print(df.columns)
    
    print("\nStep 3: Preprocessing")
    train_scaled, test_scaled, cat_features, num_features = preprocess_data(df, is_train=True)
    print(num_features)
    print(train_scaled.columns)
    pipeline = build_churn_pipeline(num_features, cat_features)

    print("\nStep 4: Training")
    run_id = train(pipeline, train_scaled)

    print("\nStep 5: Evaluation")
    accuracy, precision, recall = evaluate(test_scaled, run_id)

if __name__ == "__main__":
    run_pipeline()