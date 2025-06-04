import numpy as np
from Test import test_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define class names for binary classification
class_names = ['Non-FTU', 'FTU']
num_classes = len(class_names)

def evaluate_model():
    
    # Get predictions & true labels from test_model()
    all_labels, all_predictions = test_model()

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels= np.arange(num_classes))

    # Generate classification report
    class_report = classification_report(all_labels, all_predictions, target_names=class_names)

    print("Classification Report:\n", class_report)

    return conf_matrix, class_report

def plot_confusion_matrix(conf_matrix):

    disp = ConfusionMatrixDisplay(confusion_matrix= conf_matrix, display_labels= class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('confusion matrix for FTU and Non FTU classification')
    plt.gca().set_xlabel('Predicted labels')
    plt.gca().set_ylabel('True labels')
    plt.tight_layout()
    plt.show()
    

# Run evaluation if script is executed directly
if __name__ == '__main__':
    conf_matrix, class_report = evaluate_model()
    plot_confusion_matrix(conf_matrix)

