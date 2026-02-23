"""
Models module for California Housing Dataset
Contains regression, classification, SVM, and neural network models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, 
                           confusion_matrix, classification_report, precision_score, 
                           recall_score, f1_score)
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

class CaliforniaHousingModels:
    def __init__(self):
        """Initialize the models class"""
        self.regression_models = {}
        self.classification_models = {}
        self.neural_model = None
        self.label_encoder = LabelEncoder()
        
    # ==================== REGRESSION MODELS ====================
    
    def train_simple_linear_regression(self, X_train, y_train, feature_name='MedInc'):
        """Train simple linear regression using one feature"""
        print(f"Training Simple Linear Regression with {feature_name}...")
        
        model = LinearRegression()
        X_train_single = X_train[[feature_name]] if hasattr(X_train, 'columns') else X_train[:, [X_train.columns.get_loc(feature_name) if hasattr(X_train, 'columns') else 0]]
        model.fit(X_train_single, y_train)
        
        self.regression_models['simple_linear'] = model
        print(f"Simple Linear Regression trained successfully")
        print(f"Coefficient: {model.coef_[0]:.4f}")
        print(f"Intercept: {model.intercept_:.4f}")
        
        return model
    
    def train_multiple_linear_regression(self, X_train, y_train):
        """Train multiple linear regression using all features"""
        print("Training Multiple Linear Regression...")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        self.regression_models['multiple_linear'] = model
        print("Multiple Linear Regression trained successfully")
        
        # Print coefficients
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns
        else:
            feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
            
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        print("\nFeature Coefficients:")
        print(coef_df)
        
        return model
    
    def evaluate_regression_models(self, X_test, y_test):
        """Evaluate regression models on test set"""
        print("\n=== REGRESSION EVALUATION ON TEST SET ===")
        
        results = {}
        
        for name, model in self.regression_models.items():
            print(f"\n{name.replace('_', ' ').title()} Regression:")
            
            if name == 'simple_linear':
                X_test_eval = X_test[['MedInc']] if hasattr(X_test, 'columns') else X_test[:, [X_test.columns.get_loc('MedInc') if hasattr(X_test, 'columns') else 0]]
            else:
                X_test_eval = X_test
            
            y_pred = model.predict(X_test_eval)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {'MSE': mse, 'R2': r2}
            
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"R² Score: {r2:.4f}")
            
            # Create actual vs predicted plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Actual vs Predicted - {name.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'plots/{name}_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return results
    
    # ==================== CLASSIFICATION MODELS ====================
    
    def prepare_classification_data(self, y_train, y_val, y_test):
        """Prepare classification labels"""
        # Convert string labels to numeric
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        print(f"Class mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        return y_train_encoded, y_val_encoded, y_test_encoded
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression classifier"""
        print("Training Logistic Regression...")
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        self.classification_models['logistic'] = model
        print("Logistic Regression trained successfully")
        
        return model
    
    def train_decision_tree(self, X_train, y_train):
        """Train Decision Tree classifier"""
        print("Training Decision Tree...")
        
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        self.classification_models['decision_tree'] = model
        print("Decision Tree trained successfully")
        print(f"Tree depth: {model.get_depth()}")
        
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier"""
        print("Training Random Forest...")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        self.classification_models['random_forest'] = model
        print("Random Forest trained successfully")
        
        return model
    
    def train_svm_classifier(self, X_train, y_train, kernel='rbf'):
        """Train SVM classifier with specified kernel"""
        print(f"Training SVM with {kernel} kernel...")
        
        model = SVC(kernel=kernel, random_state=42, probability=True)
        model.fit(X_train, y_train)
        
        self.classification_models[f'svm_{kernel}'] = model
        print(f"SVM with {kernel} kernel trained successfully")
        
        return model
    
    def evaluate_classification_models(self, X_test, y_test):
        """Evaluate classification models on test set"""
        print("\n=== CLASSIFICATION EVALUATION ON TEST SET ===")
        
        results = {}
        
        for name, model in self.classification_models.items():
            print(f"\n{name.replace('_', ' ').title()}:")
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            plt.title(f'Confusion Matrix - {name.replace("_", " ").title()}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(f'plots/{name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Classification Report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, 
                                     target_names=self.label_encoder.classes_))
        
        return results
    
    # ==================== NEURAL NETWORK ====================
    
    def build_neural_network(self, input_dim):
        """Build neural network model"""
        print("Building Neural Network...")
        
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')  # 3 classes
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.neural_model = model
        print("Neural Network built successfully")
        print(model.summary())
        
        return model
    
    def train_neural_network(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train neural network with early stopping"""
        print("Training Neural Network...")
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, 
                                     restore_best_weights=True)
        
        history = self.neural_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(15, 5))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training vs Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/neural_network_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Neural Network trained successfully")
        return history
    
    def evaluate_neural_network(self, X_test, y_test):
        """Evaluate neural network on test set"""
        print("\n=== NEURAL NETWORK EVALUATION ON TEST SET ===")
        
        y_pred_proba = self.neural_model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - Neural Network')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('plots/neural_network_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    
    # ==================== MODEL SAVING ====================
    
    def save_models(self):
        """Save all trained models"""
        print("\nSaving models...")
        
        # Save regression models
        for name, model in self.regression_models.items():
            joblib.dump(model, f'saved_models/{name}_regression.joblib')
        
        # Save classification models
        for name, model in self.classification_models.items():
            joblib.dump(model, f'saved_models/{name}_classifier.joblib')
        
        # Save neural network
        if self.neural_model:
            self.neural_model.save('saved_models/neural_network.h5')
        
        # Save label encoder
        joblib.dump(self.label_encoder, 'saved_models/label_encoder.joblib')
        
        print("All models saved successfully!")
    
    def load_models(self):
        """Load all saved models"""
        print("Loading models...")
        
        # Load regression models
        try:
            self.regression_models['simple_linear'] = joblib.load('saved_models/simple_linear_regression.joblib')
            self.regression_models['multiple_linear'] = joblib.load('saved_models/multiple_linear_regression.joblib')
        except:
            print("Regression models not found")
        
        # Load classification models
        try:
            self.classification_models['logistic'] = joblib.load('saved_models/logistic_classifier.joblib')
            self.classification_models['decision_tree'] = joblib.load('saved_models/decision_tree_classifier.joblib')
            self.classification_models['random_forest'] = joblib.load('saved_models/random_forest_classifier.joblib')
            self.classification_models['svm_rbf'] = joblib.load('saved_models/svm_rbf_classifier.joblib')
        except:
            print("Classification models not found")
        
        # Load neural network
        try:
            self.neural_model = tf.keras.models.load_model('saved_models/neural_network.h5')
        except:
            print("Neural network not found")
        
        # Load label encoder
        try:
            self.label_encoder = joblib.load('saved_models/label_encoder.joblib')
        except:
            print("Label encoder not found")
        
        print("Models loaded successfully!")

def main():
    """Main function to train all models"""
    from preprocessing import CaliforniaHousingPreprocessor
    
    print("=== CALIFORNIA HOUSING ML PIPELINE ===")
    
    # Preprocess data
    preprocessor = CaliforniaHousingPreprocessor()
    preprocessor.load_data()
    preprocessor.split_data()
    preprocessor.fit_scaler()
    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.transform_data()
    
    # Create classification labels
    y_train_labels, _, _ = preprocessor.create_classification_labels(preprocessor.y_train)
    y_val_labels, _, _ = preprocessor.create_classification_labels(preprocessor.y_val)
    y_test_labels, _, _ = preprocessor.create_classification_labels(preprocessor.y_test)
    
    # Initialize models
    models = CaliforniaHousingModels()
    
    # ==================== REGRESSION ====================
    print("\n" + "="*50)
    print("PHASE 2: REGRESSION MODELS")
    print("="*50)
    
    models.train_simple_linear_regression(X_train_scaled, preprocessor.y_train)
    models.train_multiple_linear_regression(X_train_scaled, preprocessor.y_train)
    regression_results = models.evaluate_regression_models(X_test_scaled, preprocessor.y_test)
    
    # ==================== CLASSIFICATION ====================
    print("\n" + "="*50)
    print("PHASE 3: CLASSIFICATION MODELS")
    print("="*50)
    
    y_train_enc, y_val_enc, y_test_enc = models.prepare_classification_data(
        y_train_labels, y_val_labels, y_test_labels
    )
    
    models.train_logistic_regression(X_train_scaled, y_train_enc)
    models.train_decision_tree(X_train_scaled, y_train_enc)
    models.train_random_forest(X_train_scaled, y_train_enc)
    
    classification_results = models.evaluate_classification_models(X_test_scaled, y_test_enc)
    
    # ==================== SVM ====================
    print("\n" + "="*50)
    print("PHASE 4: SVM CLASSIFIER")
    print("="*50)
    
    models.train_svm_classifier(X_train_scaled, y_train_enc, kernel='linear')
    models.train_svm_classifier(X_train_scaled, y_train_enc, kernel='rbf')
    
    svm_results = models.evaluate_classification_models(X_test_scaled, y_test_enc)
    
    # ==================== NEURAL NETWORK ====================
    print("\n" + "="*50)
    print("PHASE 5: NEURAL NETWORK")
    print("="*50)
    
    models.build_neural_network(X_train_scaled.shape[1])
    models.train_neural_network(X_train_scaled, y_train_enc, X_val_scaled, y_val_enc)
    neural_results = models.evaluate_neural_network(X_test_scaled, y_test_enc)
    
    # Save all models
    models.save_models()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    
    return models, regression_results, classification_results, neural_results

if __name__ == "__main__":
    models, reg_results, class_results, neural_results = main()
