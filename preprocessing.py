"""
Preprocessing module for California Housing Dataset
Handles data loading, splitting, and scaling
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class CaliforniaHousingPreprocessor:
    def __init__(self, random_state=42):
        """Initialize the preprocessor with random state for reproducibility"""
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_names = None
        
    def load_data(self):
        """Load California Housing dataset"""
        print("Loading California Housing dataset...")
        housing = fetch_california_housing()
        
        # Create DataFrame
        self.data = pd.DataFrame(housing.data, columns=housing.feature_names)
        self.data['MedHouseVal'] = housing.target
        self.feature_names = housing.feature_names
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Features: {self.feature_names}")
        return self.data
    
    def check_missing_values(self):
        """Check for missing values in the dataset"""
        print("\nChecking for missing values...")
        missing = self.data.isnull().sum()
        print(missing)
        return missing
    
    def split_data(self):
        """Split data into train (70%), validation (15%), test (15%)"""
        print("\nSplitting data...")
        
        X = self.data[self.feature_names]
        y = self.data['MedHouseVal']
        
        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Second split: 15% val, 15% test (from the 30% temp)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state
        )
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        print(f"Train set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def fit_scaler(self):
        """Fit scaler on training data only"""
        print("Fitting StandardScaler on training data...")
        self.scaler.fit(self.X_train)
        print("Scaler fitted successfully")
    
    def transform_data(self):
        """Transform all datasets using the fitted scaler"""
        print("Transforming data...")
        
        X_train_scaled = self.scaler.transform(self.X_train)
        X_val_scaled = self.scaler.transform(self.X_val)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrames with column names
        self.X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        self.X_val_scaled = pd.DataFrame(X_val_scaled, columns=self.feature_names)
        self.X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        
        print("Data transformation completed")
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def save_scaler(self, filepath='saved_models/scaler.joblib'):
        """Save the fitted scaler"""
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath='saved_models/scaler.joblib'):
        """Load a saved scaler"""
        self.scaler = joblib.load(filepath)
        print(f"Scaler loaded from {filepath}")
        return self.scaler
    
    def create_classification_labels(self, y_data):
        """Create 3 classes from MedHouseVal: Low (33%), Medium (33%), High (33%)"""
        print("Creating classification labels...")
        
        # Calculate percentiles
        low_threshold = np.percentile(y_data, 33)
        high_threshold = np.percentile(y_data, 67)
        
        # Create labels
        labels = []
        for value in y_data:
            if value <= low_threshold:
                labels.append('Low')
            elif value <= high_threshold:
                labels.append('Medium')
            else:
                labels.append('High')
        
        return np.array(labels), low_threshold, high_threshold
    
    def create_eda_plots(self):
        """Create and save EDA plots"""
        print("Creating EDA plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Histogram of MedInc
        plt.figure(figsize=(10, 6))
        plt.hist(self.data['MedInc'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Median Income', fontsize=14, fontweight='bold')
        plt.xlabel('Median Income', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/medinc_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Scatter plot: MedInc vs MedHouseVal
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data['MedInc'], self.data['MedHouseVal'], 
                   alpha=0.6, color='coral', edgecolors='black', linewidth=0.5)
        plt.title('Median Income vs Median House Value', fontsize=14, fontweight='bold')
        plt.xlabel('Median Income', fontsize=12)
        plt.ylabel('Median House Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/medinc_vs_medhouseval_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Correlation heatmap
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.data.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("EDA plots saved to plots/ directory")
    
    def get_processed_data(self):
        """Return all processed data"""
        return {
            'X_train': self.X_train_scaled,
            'X_val': self.X_val_scaled,
            'X_test': self.X_test_scaled,
            'y_train': self.y_train,
            'y_val': self.y_val,
            'y_test': self.y_test,
            'feature_names': self.feature_names
        }

def main():
    """Main function to run preprocessing pipeline"""
    preprocessor = CaliforniaHousingPreprocessor()
    
    # Load and explore data
    data = preprocessor.load_data()
    preprocessor.check_missing_values()
    
    # Split data
    preprocessor.split_data()
    
    # Scale data
    preprocessor.fit_scaler()
    preprocessor.transform_data()
    
    # Save scaler
    preprocessor.save_scaler()
    
    # Create EDA plots
    preprocessor.create_eda_plots()
    
    # Create classification labels
    y_train_labels, low_thresh, high_thresh = preprocessor.create_classification_labels(preprocessor.y_train)
    y_val_labels, _, _ = preprocessor.create_classification_labels(preprocessor.y_val)
    y_test_labels, _, _ = preprocessor.create_classification_labels(preprocessor.y_test)
    
    print(f"\nClassification thresholds:")
    print(f"Low: <= {low_thresh:.3f}")
    print(f"Medium: {low_thresh:.3f} - {high_thresh:.3f}")
    print(f"High: > {high_thresh:.3f}")
    
    print("\nPreprocessing completed successfully!")
    return preprocessor

if __name__ == "__main__":
    preprocessor = main()
