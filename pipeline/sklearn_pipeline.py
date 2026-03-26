from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocessing.preprocessing_pipeline import build_preprocessor

def build_churn_pipeline(num_features: list, cat_features: list) -> Pipeline:
    preprocessor = build_preprocessor(num_features, cat_features)
    lr = LogisticRegression(
            C=0.03,
            solver="liblinear",
            penalty="l2",
            max_iter=491
        )
    return Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", lr),
    ])