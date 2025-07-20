import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Load and analyze the Parkinson's dataset
def load_and_analyze_data():
    # Read the data
    df = pd.read_csv('parkinsons.data')
    
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())
    
    print("\nClass Distribution:")
    print(df['status'].value_counts())
    print(f"Parkinson's patients: {df['status'].sum()} ({df['status'].mean()*100:.1f}%)")
    print(f"Healthy individuals: {(df['status']==0).sum()} ({(1-df['status'].mean())*100:.1f}%)")
    
    return df

def create_visualizations(df):
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Class distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Class distribution pie chart
    df['status'].value_counts().plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%', 
                                   labels=['Healthy', "Parkinson's"])
    axes[0,0].set_title('Distribution of Health Status')
    axes[0,0].set_ylabel('')
    
    # Feature correlation heatmap
    numeric_features = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_features].corr()
    
    # Plot correlation heatmap for key features
    key_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
                   'MDVP:Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA', 'status']
    
    sns.heatmap(df[key_features].corr(), annot=True, cmap='coolwarm', center=0,
                ax=axes[0,1], fmt='.2f')
    axes[0,1].set_title('Feature Correlation Matrix')
    
    # Box plots for key features
    key_vocal_features = ['MDVP:Fo(Hz)', 'HNR', 'MDVP:Shimmer', 'MDVP:Jitter(%)']
    
    for i, feature in enumerate(key_vocal_features[:2]):
        df.boxplot(column=feature, by='status', ax=axes[1,i])
        axes[1,i].set_title(f'{feature} by Health Status')
        axes[1,i].set_xlabel('Health Status (0=Healthy, 1=Parkinson\'s)')
    
    plt.tight_layout()
    plt.savefig('parkinson_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Detailed feature analysis
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    important_features = [
        'MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 
        'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'PPE'
    ]
    
    for i, feature in enumerate(important_features):
        row, col = i // 3, i % 3
        
        # Create violin plot
        healthy = df[df['status'] == 0][feature]
        parkinson = df[df['status'] == 1][feature]
        
        axes[row, col].violinplot([healthy, parkinson], positions=[0, 1])
        axes[row, col].set_xticks([0, 1])
        axes[row, col].set_xticklabels(['Healthy', "Parkinson's"])
        axes[row, col].set_title(f'{feature}')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_matrix

def statistical_analysis(df):
    # Statistical comparison between healthy and Parkinson's patients
    features = df.columns[1:-1]  # Exclude 'name' and 'status'
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: HEALTHY vs PARKINSON'S PATIENTS")
    print("="*80)
    
    healthy_stats = df[df['status'] == 0][features].describe()
    parkinson_stats = df[df['status'] == 1][features].describe()
    
    print("\nKey Differences (Mean values):")
    print("-" * 60)
    
    important_features = [
        'MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 
        'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'PPE'
    ]
    
    comparison_df = pd.DataFrame({
        'Healthy_Mean': healthy_stats.loc['mean', important_features],
        'Parkinsons_Mean': parkinson_stats.loc['mean', important_features],
    })
    comparison_df['Difference_%'] = ((comparison_df['Parkinsons_Mean'] - comparison_df['Healthy_Mean']) / 
                                   comparison_df['Healthy_Mean'] * 100)
    
    print(comparison_df.round(4))
    
    return healthy_stats, parkinson_stats, comparison_df

def build_models(df):
    # Prepare data
    X = df.drop(['name', 'status'], axis=1)
    y = df['status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    model_results = {}
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    for name, model in models.items():
        if name in ['SVM', 'Neural Network']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        model_results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred,
            'scaler': scaler if name in ['SVM', 'Neural Network'] else None
        }
        
        print(f"\n{name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
    
    # PCA Analysis
    pca = PCA()
    X_pca = pca.fit_transform(X_train_scaled)
    
    # Plot PCA explained variance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             pca.explained_variance_ratio_, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance Ratio')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum) + 1), cumsum, 'ro-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.grid(True)
    plt.axhline(y=0.95, color='k', linestyle='--', label='95% variance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model_results, scaler, pca

def feature_importance_analysis(model_results, feature_names):
    # Feature importance from Random Forest
    rf_model = model_results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
    plt.title('Top 15 Most Important Features (Random Forest)')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance

if __name__ == "__main__":
    # Run complete analysis
    df = load_and_analyze_data()
    correlation_matrix = create_visualizations(df)
    healthy_stats, parkinson_stats, comparison_df = statistical_analysis(df)
    model_results, scaler, pca = build_models(df)
    
    feature_names = df.drop(['name', 'status'], axis=1).columns.tolist()
    feature_importance = feature_importance_analysis(model_results, feature_names)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("Generated files:")
    print("- parkinson_analysis.png")
    print("- feature_distributions.png") 
    print("- pca_analysis.png")
    print("- feature_importance.png") 