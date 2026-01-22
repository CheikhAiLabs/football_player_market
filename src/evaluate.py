import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import os

def evaluate_model():
    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    print("\nLoading model and test data...")
    model = joblib.load('../models/football_model.pkl')
    metadata = joblib.load('../models/model_metadata.pkl')
    test_data = joblib.load('../models/test_data.pkl')
    
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    print(f"Model type: {metadata['model_type']}")
    print(f"Test samples: {len(X_test)}")
    
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    print("\n" + "="*80)
    print("REGRESSION METRICS")
    print("="*80)
    print(f"Mean Absolute Error (MAE):        €{mae:,.0f}")
    print(f"Root Mean Squared Error (RMSE):   €{rmse:,.0f}")
    print(f"R² Score:                         {r2:.4f}")
    print(f"Mean Absolute Percentage Error:   {mape:.2f}%")
    
    baseline_mae = mean_absolute_error(y_test, [y_test.mean()] * len(y_test))
    baseline_r2 = 0.0
    
    print("\n" + "="*80)
    print("BASELINE COMPARISON (Mean Predictor)")
    print("="*80)
    print(f"Baseline MAE:  €{baseline_mae:,.0f}")
    print(f"Baseline R²:   {baseline_r2:.4f}")
    print(f"Improvement:   {((baseline_mae - mae) / baseline_mae * 100):.1f}% better MAE")
    
    if r2 > 0.5:
        print(f"\n✓ Model PASSES baseline: R² = {r2:.4f} > 0.5")
    else:
        print(f"\n✗ Model below target: R² = {r2:.4f} < 0.5")
    
    os.makedirs('../analysis', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=10)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Market Value (€)', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Market Value (€)', fontsize=12)
    axes[0, 0].set_title(f'Actual vs Predicted (R² = {r2:.4f})', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Market Value (€)', fontsize=12)
    axes[0, 1].set_ylabel('Residuals (€)', fontsize=12)
    axes[0, 1].set_title('Residual Plot', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': metadata['feature_columns'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        axes[1, 0].barh(range(len(feature_importance)), feature_importance['importance'])
        axes[1, 0].set_yticks(range(len(feature_importance)))
        axes[1, 0].set_yticklabels(feature_importance['feature'])
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_xlabel('Importance', fontsize=12)
        axes[1, 0].set_title('Top 15 Feature Importances', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    error_pct = np.abs(residuals) / y_test * 100
    error_pct = error_pct[error_pct < 200]
    axes[1, 1].hist(error_pct, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Absolute Percentage Error (%)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Distribution of Prediction Errors', fontsize=14)
    axes[1, 1].axvline(x=mape, color='r', linestyle='--', lw=2, label=f'Mean: {mape:.1f}%')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = '../analysis/evaluation_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {plot_path}")
    plt.close()
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("FOOTBALL PLAYER MARKET VALUE PREDICTION - EVALUATION REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"Model Type: {metadata['model_type'].replace('_', ' ').title()}")
    report_lines.append(f"Training Samples: {metadata['train_samples']:,}")
    report_lines.append(f"Test Samples: {metadata['test_samples']:,}")
    report_lines.append(f"Number of Features: {len(metadata['feature_columns'])}")
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("PERFORMANCE METRICS")
    report_lines.append("="*80)
    report_lines.append(f"Mean Absolute Error (MAE):        €{mae:,.0f}")
    report_lines.append(f"Root Mean Squared Error (RMSE):   €{rmse:,.0f}")
    report_lines.append(f"R² Score:                         {r2:.4f}")
    report_lines.append(f"Mean Absolute Percentage Error:   {mape:.2f}%")
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("BASELINE COMPARISON")
    report_lines.append("="*80)
    report_lines.append(f"Baseline MAE (Mean Predictor):    €{baseline_mae:,.0f}")
    report_lines.append(f"Model MAE Improvement:            {((baseline_mae - mae) / baseline_mae * 100):.1f}%")
    report_lines.append("")
    if r2 > 0.5:
        report_lines.append(f"✓ SUCCESS: Model achieves R² = {r2:.4f} > 0.5 target")
    else:
        report_lines.append(f"⚠ Model R² = {r2:.4f} below 0.5 target")
    report_lines.append("")
    
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': metadata['feature_columns'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        report_lines.append("="*80)
        report_lines.append("TOP 15 FEATURE IMPORTANCES")
        report_lines.append("="*80)
        for idx, row in feature_importance.head(15).iterrows():
            report_lines.append(f"{row['feature']:30s} {row['importance']:8.4f}")
        report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("PREDICTION EXAMPLES (First 10 Test Samples)")
    report_lines.append("="*80)
    report_lines.append(f"{'Actual (€)':>15s} {'Predicted (€)':>15s} {'Error (€)':>15s} {'Error %':>10s}")
    report_lines.append("-"*80)
    
    for i in range(min(10, len(y_test))):
        actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        predicted = y_pred[i]
        error = actual - predicted
        error_pct = abs(error) / actual * 100
        report_lines.append(f"{actual:15,.0f} {predicted:15,.0f} {error:15,.0f} {error_pct:9.1f}%")
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)
    
    report_text = "\n".join(report_lines)
    
    report_path = '../analysis/evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"Evaluation report saved to {report_path}")
    
    print("\n" + report_text)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

if __name__ == "__main__":
    metrics = evaluate_model()