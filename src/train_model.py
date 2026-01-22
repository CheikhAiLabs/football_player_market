import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import joblib
import os
import sys
from data_pipeline import FootballDataPipeline

def train_model(model_type='random_forest', n_estimators=100, max_depth=20, random_state=42):
    print("="*80)
    print("FOOTBALL PLAYER MARKET VALUE PREDICTION - MODEL TRAINING")
    print("="*80)
    
    pipeline = FootballDataPipeline(data_dir='../data')
    X_train, X_test, y_train, y_test, feature_cols = pipeline.run_pipeline(
        test_size=0.2, 
        random_state=random_state
    )
    
    print(f"\nTraining {model_type} model...")
    print(f"Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}")
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            verbose=1
        )
    elif model_type == 'ridge':
        model = Ridge(random_state=random_state)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print("\nPerforming 3-fold cross-validation on training data...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, 
                                 scoring='r2', n_jobs=-1, verbose=1)
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print("\nTraining final model on full training set...")
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nTraining R²: {train_score:.4f}")
    print(f"Test R²: {test_score:.4f}")
    
    os.makedirs('../models', exist_ok=True)
    model_path = '../models/football_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    metadata = {
        'model_type': model_type,
        'n_estimators': n_estimators if model_type in ['random_forest', 'gradient_boosting'] else None,
        'max_depth': max_depth if model_type in ['random_forest', 'gradient_boosting'] else None,
        'feature_columns': feature_cols,
        'train_score': train_score,
        'test_score': test_score,
        'cv_scores': cv_scores.tolist(),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    metadata_path = '../models/model_metadata.pkl'
    joblib.dump(metadata, metadata_path)
    print(f"Model metadata saved to {metadata_path}")
    
    test_data_path = '../models/test_data.pkl'
    joblib.dump({'X_test': X_test, 'y_test': y_test}, test_data_path)
    print(f"Test data saved to {test_data_path}")
    
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    return model, metadata

if __name__ == "__main__":
    model, metadata = train_model(
        model_type='random_forest',
        n_estimators=100,
        max_depth=20,
        random_state=42
    )