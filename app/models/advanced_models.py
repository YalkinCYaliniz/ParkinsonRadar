import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import joblib
import warnings
warnings.filterwarnings('ignore')

class ParkinsonModelTrainer:
    """
    Advanced model trainer for Parkinson's disease detection
    Includes ensemble methods and deep learning
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.best_model = None
        self.feature_names = None
        
    def load_data(self, data_path='data/raw/parkinsons.data'):
        """Load and prepare the Parkinson's dataset"""
        df = pd.read_csv(data_path)
        
        # Features and target
        X = df.drop(['name', 'status'], axis=1)
        y = df['status']
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2, random_state=42):
        """Advanced preprocessing with feature selection and scaling"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Feature scaling
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        X_train_std = self.scalers['standard'].fit_transform(X_train)
        X_test_std = self.scalers['standard'].transform(X_test)
        
        X_train_robust = self.scalers['robust'].fit_transform(X_train)
        X_test_robust = self.scalers['robust'].transform(X_test)
        
        # Feature selection
        # SelectKBest
        self.feature_selectors['kbest'] = SelectKBest(score_func=f_classif, k=15)
        X_train_kbest = self.feature_selectors['kbest'].fit_transform(X_train_std, y_train)
        X_test_kbest = self.feature_selectors['kbest'].transform(X_test_std)
        
        # PCA
        self.feature_selectors['pca'] = PCA(n_components=0.95)  # Keep 95% variance
        X_train_pca = self.feature_selectors['pca'].fit_transform(X_train_std)
        X_test_pca = self.feature_selectors['pca'].transform(X_test_std)
        
        return {
            'original': (X_train, X_test, y_train, y_test),
            'standard': (X_train_std, X_test_std, y_train, y_test),
            'robust': (X_train_robust, X_test_robust, y_train, y_test),
            'kbest': (X_train_kbest, X_test_kbest, y_train, y_test),
            'pca': (X_train_pca, X_test_pca, y_train, y_test)
        }
    
    def train_traditional_models(self, data_dict):
        """Train traditional ML models with hyperparameter tuning"""
        results = {}
        
        # Random Forest with GridSearch
        print("Training Random Forest...")
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
        
        X_train, X_test, y_train, y_test = data_dict['original']
        rf_grid.fit(X_train, y_train)
        
        rf_best = rf_grid.best_estimator_
        rf_pred = rf_best.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        
        results['Random Forest'] = {
            'model': rf_best,
            'accuracy': rf_acc,
            'predictions': rf_pred,
            'data_type': 'original'
        }
        
        # XGBoost
        print("Training XGBoost...")
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='accuracy', n_jobs=-1)
        
        X_train_std, X_test_std, _, _ = data_dict['standard']
        xgb_grid.fit(X_train_std, y_train)
        
        xgb_best = xgb_grid.best_estimator_
        xgb_pred = xgb_best.predict(X_test_std)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        
        results['XGBoost'] = {
            'model': xgb_best,
            'accuracy': xgb_acc,
            'predictions': xgb_pred,
            'data_type': 'standard'
        }
        
        # LightGBM
        print("Training LightGBM...")
        lgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 50]
        }
        
        lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=5, scoring='accuracy', n_jobs=-1)
        lgb_grid.fit(X_train_std, y_train)
        
        lgb_best = lgb_grid.best_estimator_
        lgb_pred = lgb_best.predict(X_test_std)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        
        results['LightGBM'] = {
            'model': lgb_best,
            'accuracy': lgb_acc,
            'predictions': lgb_pred,
            'data_type': 'standard'
        }
        
        # SVM
        print("Training SVM...")
        svm_params = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'poly']
        }
        
        svm_model = SVC(random_state=42, probability=True)
        svm_grid = GridSearchCV(svm_model, svm_params, cv=5, scoring='accuracy', n_jobs=-1)
        
        X_train_robust, X_test_robust, _, _ = data_dict['robust']
        svm_grid.fit(X_train_robust, y_train)
        
        svm_best = svm_grid.best_estimator_
        svm_pred = svm_best.predict(X_test_robust)
        svm_acc = accuracy_score(y_test, svm_pred)
        
        results['SVM'] = {
            'model': svm_best,
            'accuracy': svm_acc,
            'predictions': svm_pred,
            'data_type': 'robust'
        }
        
        return results
    
    def create_deep_learning_model(self, input_dim, architecture='advanced'):
        """Create advanced deep learning model"""
        if architecture == 'advanced':
            # Advanced architecture with regularization
            model = Sequential([
                Dense(256, activation='relu', input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
                BatchNormalization(),
                Dropout(0.4),
                
                Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(32, activation='relu'),
                Dropout(0.2),
                
                Dense(1, activation='sigmoid')
            ])
        else:
            # Simple architecture
            model = Sequential([
                Dense(64, activation='relu', input_shape=(input_dim,)),
                Dropout(0.5),
                Dense(32, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_deep_learning_models(self, data_dict):
        """Train deep learning models"""
        results = {}
        
        # Model 1: Standard scaled data
        print("Training Deep Learning Model 1 (Standard Scaling)...")
        X_train_std, X_test_std, y_train, y_test = data_dict['standard']
        
        model1 = self.create_deep_learning_model(X_train_std.shape[1], 'advanced')
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
        ]
        
        history1 = model1.fit(
            X_train_std, y_train,
            validation_split=0.2,
            epochs=200,
            batch_size=16,
            callbacks=callbacks,
            verbose=0
        )
        
        dl1_pred_prob = model1.predict(X_test_std, verbose=0)
        dl1_pred = (dl1_pred_prob > 0.5).astype(int).flatten()
        dl1_acc = accuracy_score(y_test, dl1_pred)
        
        results['Deep Learning 1'] = {
            'model': model1,
            'accuracy': dl1_acc,
            'predictions': dl1_pred,
            'data_type': 'standard'
        }
        
        # Model 2: Feature selected data
        print("Training Deep Learning Model 2 (Feature Selection)...")
        X_train_kbest, X_test_kbest, _, _ = data_dict['kbest']
        
        model2 = self.create_deep_learning_model(X_train_kbest.shape[1], 'simple')
        
        history2 = model2.fit(
            X_train_kbest, y_train,
            validation_split=0.2,
            epochs=200,
            batch_size=16,
            callbacks=callbacks,
            verbose=0
        )
        
        dl2_pred_prob = model2.predict(X_test_kbest, verbose=0)
        dl2_pred = (dl2_pred_prob > 0.5).astype(int).flatten()
        dl2_acc = accuracy_score(y_test, dl2_pred)
        
        results['Deep Learning 2'] = {
            'model': model2,
            'accuracy': dl2_acc,
            'predictions': dl2_pred,
            'data_type': 'kbest'
        }
        
        return results
    
    def create_ensemble_model(self, individual_results, data_dict):
        """Create ensemble model from best individual models"""
        print("Creating Ensemble Model...")
        
        # Select top models based on accuracy
        sorted_models = sorted(individual_results.items(), 
                             key=lambda x: x[1]['accuracy'], reverse=True)
        
        # Get top 3 traditional models for voting
        top_traditional = []
        for name, result in sorted_models:
            if 'Deep Learning' not in name and len(top_traditional) < 3:
                top_traditional.append((name, result))
        
        # Create voting ensemble
        estimators = []
        for name, result in top_traditional:
            estimators.append((name.lower().replace(' ', '_'), result['model']))
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability voting
        )
        
        # Train on appropriate data (use most common data type)
        data_types = [result['data_type'] for _, result in top_traditional]
        most_common_type = max(set(data_types), key=data_types.count)
        
        X_train, X_test, y_train, y_test = data_dict[most_common_type]
        voting_clf.fit(X_train, y_train)
        
        ensemble_pred = voting_clf.predict(X_test)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        
        return {
            'Ensemble': {
                'model': voting_clf,
                'accuracy': ensemble_acc,
                'predictions': ensemble_pred,
                'data_type': most_common_type
            }
        }
    
    def train_all_models(self, data_path='parkinsons.data'):
        """Train all models and select the best one"""
        print("="*60)
        print("ADVANCED PARKINSON'S DISEASE MODEL TRAINING")
        print("="*60)
        
        # Load and preprocess data
        X, y = self.load_data(data_path)
        data_dict = self.preprocess_data(X, y)
        
        # Train traditional models
        print("\n1. Training Traditional ML Models...")
        traditional_results = self.train_traditional_models(data_dict)
        
        # Train deep learning models
        print("\n2. Training Deep Learning Models...")
        dl_results = self.train_deep_learning_models(data_dict)
        
        # Combine all results
        all_results = {**traditional_results, **dl_results}
        
        # Create ensemble
        print("\n3. Creating Ensemble Model...")
        ensemble_results = self.create_ensemble_model(all_results, data_dict)
        all_results.update(ensemble_results)
        
        # Store all results
        self.models = all_results
        
        # Select best model
        best_model_name = max(all_results.keys(), 
                            key=lambda k: all_results[k]['accuracy'])
        self.best_model = all_results[best_model_name]
        
        # Print results
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        for name, result in sorted(all_results.items(), 
                                 key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"{name:20} | Accuracy: {result['accuracy']:.4f} | Data: {result['data_type']}")
        
        print(f"\nBest Model: {best_model_name} (Accuracy: {self.best_model['accuracy']:.4f})")
        
        return all_results
    
    def predict_parkinson(self, features):
        """
        Predict Parkinson's disease using best model but with enhanced confidence calculation
        """
        if self.best_model is None:
            raise ValueError("No model trained. Call train_all_models() first.")
        
        # Convert features to DataFrame if it's a dict
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features
        
        # Ensure feature order matches training data
        if self.feature_names is not None:
            features_df = features_df[self.feature_names]
        
        # Apply preprocessing based on best model's data type
        data_type = self.best_model['data_type']
        
        if data_type == 'standard':
            features_processed = self.scalers['standard'].transform(features_df)
        elif data_type == 'robust':
            features_processed = self.scalers['robust'].transform(features_df)
        elif data_type == 'kbest':
            features_std = self.scalers['standard'].transform(features_df)
            features_processed = self.feature_selectors['kbest'].transform(features_std)
        elif data_type == 'pca':
            features_std = self.scalers['standard'].transform(features_df)
            features_processed = self.feature_selectors['pca'].transform(features_std)
        else:  # original
            features_processed = features_df.values
        
        # Make prediction using BEST MODEL ONLY (original behavior)
        model = self.best_model['model']
        
        if 'Deep Learning' in str(type(model)) or hasattr(model, 'predict_proba'):
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(features_processed)[0][1]
                else:  # Deep learning model
                    prob = model.predict(features_processed, verbose=0)[0][0]
                prediction = int(prob > 0.5)
            except:
                prediction = int(model.predict(features_processed)[0])
                prob = 0.5 + (prediction - 0.5) * 0.3  # Approximate probability
        else:
            prediction = int(model.predict(features_processed)[0])
            prob = 0.5 + (prediction - 0.5) * 0.3  # Approximate probability
        
        # ENHANCED CONFIDENCE CALCULATION (only this part changed)
        # 1. Basic probability confidence
        prob_confidence = abs(prob - 0.5) * 2
        
        # 2. Feature quality score 
        feature_quality = self.calculate_feature_quality(features)
        
        # 3. Model reliability (based on best model accuracy)
        model_reliability = self.best_model.get('accuracy', 0.8)
        
        # Combined enhanced confidence
        enhanced_confidence = (
            prob_confidence * 0.5 +      # How certain the prediction is
            feature_quality * 0.3 +      # How good the voice features are
            model_reliability * 0.2       # How reliable the model is
        )
        
        return {
            'prediction': prediction,
            'probability': float(prob),  # SAME as before - no change!
            'confidence': float(enhanced_confidence),  # ONLY this improved
            'feature_quality': float(feature_quality),
            'risk_level': 'High' if prob > 0.7 else 'Medium' if prob > 0.3 else 'Low'
        }
    
    def calculate_feature_quality(self, features):
        """
        Calculate feature quality score based on how normal/typical the values are
        Returns 0-1 score where 1 = very typical/normal features
        """
        try:
            # Key features that indicate good voice quality
            quality_indicators = {
                'MDVP:Jitter(%)': (0.2, 2.0),    # Good jitter range
                'MDVP:Shimmer': (0.01, 0.1),     # Good shimmer range  
                'NHR': (0.01, 0.2),              # Good noise ratio
                'HNR': (15, 30),                 # Good harmonics ratio
                'MDVP:Fo(Hz)': (80, 250)         # Normal F0 range
            }
            
            quality_scores = []
            
            for feature, (min_good, max_good) in quality_indicators.items():
                if feature in features:
                    value = features[feature]
                    # Score based on how close to "good" range the value is
                    if min_good <= value <= max_good:
                        score = 1.0  # Perfect
                    else:
                        # Distance penalty
                        if value < min_good:
                            distance = (min_good - value) / min_good
                        else:
                            distance = (value - max_good) / max_good
                        score = max(0.0, 1.0 - distance)
                    quality_scores.append(score)
            
            return np.mean(quality_scores) if quality_scores else 0.5
            
        except:
            return 0.5
    
    def save_models(self, filepath='parkinson_models.joblib'):
        """Save all trained models and preprocessors"""
        save_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors,
            'best_model': self.best_model,
            'feature_names': self.feature_names
        }
        
        joblib.dump(save_data, filepath)
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath='parkinson_models.joblib'):
        """Load trained models and preprocessors"""
        save_data = joblib.load(filepath)
        
        self.models = save_data['models']
        self.scalers = save_data['scalers']
        self.feature_selectors = save_data['feature_selectors']
        self.best_model = save_data['best_model']
        self.feature_names = save_data['feature_names']
        
        print(f"Models loaded from {filepath}")

def train_and_save_models():
    """Convenience function to train and save all models"""
    trainer = ParkinsonModelTrainer()
    results = trainer.train_all_models()
    trainer.save_models()
    return trainer, results

if __name__ == "__main__":
    # Train all models
    trainer, results = train_and_save_models()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("All models have been trained and saved.")
    print("Best model accuracy:", trainer.best_model['accuracy']) 