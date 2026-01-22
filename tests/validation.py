import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from data_pipeline import FootballDataPipeline
import joblib
import numpy as np

def test_pipeline_initialization():
    pipeline = FootballDataPipeline(data_dir='../data')
    assert pipeline.data_dir == '../data'
    print("✓ Pipeline initialization test passed")

def test_data_loading():
    pipeline = FootballDataPipeline(data_dir='../data')
    pipeline.load_data()
    assert len(pipeline.player_profiles) > 0
    assert len(pipeline.player_market_value) > 0
    print(f"✓ Data loading test passed - {len(pipeline.player_profiles)} players loaded")

def test_feature_engineering():
    pipeline = FootballDataPipeline(data_dir='../data')
    pipeline.load_data()
    pipeline.engineer_features()
    assert 'age' in pipeline.engineered_df.columns
    assert 'market_value_eur' in pipeline.engineered_df.columns
    print(f"✓ Feature engineering test passed - {len(pipeline.engineered_df)} records")

def test_preprocessing():
    pipeline = FootballDataPipeline(data_dir='../data')
    pipeline.load_data()
    pipeline.engineer_features()
    X, y = pipeline.preprocess()
    assert X.shape[0] == y.shape[0]
    assert not X.isnull().any().any()
    assert (y > 0).all()
    print(f"✓ Preprocessing test passed - Shape: {X.shape}")

def test_model_loading():
    model = joblib.load('../models/football_model.pkl')
    metadata = joblib.load('../models/model_metadata.pkl')
    test_data = joblib.load('../models/test_data.pkl')
    
    assert hasattr(model, 'predict')
    assert 'feature_columns' in metadata
    assert metadata['test_score'] > 0.5
    
    X_test = test_data['X_test']
    predictions = model.predict(X_test[:10])
    assert len(predictions) == 10
    assert all(predictions > 0)
    
    print(f"✓ Model loading and prediction test passed - R²: {metadata['test_score']:.4f}")

if __name__ == "__main__":
    print("="*80)
    print("RUNNING VALIDATION TESTS")
    print("="*80)
    
    try:
        test_pipeline_initialization()
        test_data_loading()
        test_feature_engineering()
        test_preprocessing()
        test_model_loading()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        raise