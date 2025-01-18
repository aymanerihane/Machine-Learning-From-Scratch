from tabulate import tabulate
import numpy as np
import pandas as pd
class ClassificationReport:
    def __init__(self, y_true, y_pred):
        """
        Initialize the classification report with true and predicted labels.
        :param y_true: Array-like, true labels.
        :param y_pred: Array-like, predicted labels.
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.tp = None  # True Positives
        self.tn = None  # True Negatives
        self.fp = None  # False Positives
        self.fn = None  # False Negatives
        self._compute_confusion_matrix()

    def _compute_confusion_matrix(self):
        """
        Compute the confusion matrix components.
        """
        self.tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        self.tn = np.sum((self.y_true == 0) & (self.y_pred == 0))
        self.fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        self.fn = np.sum((self.y_true == 1) & (self.y_pred == 0))

    def accuracy(self):
        """
        Calculate the accuracy.
        :return: Accuracy value.
        """
        total = self.tp + self.tn + self.fp + self.fn
        result = (self.tp + self.tn )/total
        
        return round(result * 100, 2) if total > 0 else 0


    def precision(self):
        """
        Calculate the precision.
        :return: Precision value.
        """
        return round(((self.tp / (self.tp + self.fp))*100),2) if (self.tp + self.fp) > 0 else 0

    def recall(self):
        """
        Calculate the recall (sensitivity).
        :return: Recall value.
        """
        return round(((self.tp / (self.tp + self.fn))*100),2) if (self.tp + self.fn) > 0 else 0

    def f1_score(self):
        """
        Calculate the F1 score.
        :return: F1 score value.
        """
        prec = self.precision()
        rec = self.recall()
        return round((((2 * prec * rec) / (prec + rec))),2) if (prec + rec) > 0 else 0

    def mcc(self):
        """
        Calculate the Matthews Correlation Coefficient.
        :return: MCC value.
        """
        numerator = (self.tp * self.tn) - (self.fp * self.fn)
        denominator = np.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn) *
            (self.tn + self.fp) * (self.tn + self.fn)
        )
        return round(((numerator / denominator)*100),2) if denominator > 0 else 0

    def report_data(self):
        """
        Generate a classification report.
        :return: Dictionary containing all metrics.
        """
        return {
            "Accuracy": self.accuracy(),
            "Precision": self.precision(),
            "Recall": self.recall(),
            "F1 Score": self.f1_score(),
            "MCC": self.mcc()
        }

    def generate_report(self):
        """
        Generate a table visualization of the classification report.
        """
        metrics = self.report_data()
        table = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
        print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))


