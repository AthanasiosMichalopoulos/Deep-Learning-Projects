import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def knn_1(x_train_flat,y_train,x_test_flat,y_test,PCA_TR,normalization):
    ''' 
    
    '''
    if normalization==1:
        x_train_flat = x_train_flat / 255.0
        x_test_flat = x_test_flat / 255.0
    elif PCA_TR==1:
        print("Applying PCA (100 components)...")
        pca = PCA(n_components=100)
        x_train_flat = pca.fit_transform(x_train_flat)
        x_test_flat = pca.transform(x_test_flat)
    
    print("Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train_flat, y_train)   # use smaller subset for speed
    y_pred = knn.predict(x_test_flat)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))

    # Define class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']

    # Create heatmap with annotations and proper labels
    sns.heatmap(cm,cmap='Blues',annot=True, # Show values in cells
                fmt='d',  # Format as integers
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={'size': 10},  # Annotation font size
                cbar_kws={'shrink': 0.8}) # Color bar size

    plt.title("CIFAR-10 Confusion Matrix (KNN)", fontsize=16, pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Add grid lines for better separation
    plt.grid(False)

    # Ensure layout doesn't cut off labels
    plt.tight_layout()
    plt.show()

# 1️ Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()

# 2️ Flatten images from (32,32,3) → (3072,)
x_train_flat = x_train.reshape(len(x_train), -1)
x_test_flat = x_test.reshape(len(x_test), -1)

knn_1(x_train_flat,y_train,x_test_flat,y_test,1,0)
