import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from logistic import Logistic


class FeatureSelection:
    def __init__(self, threshold=0.1, k_features=5, max_features=10):
        self.threshold = threshold
        self.k_features = k_features
        self.max_features = max_features  # Maximum number of features to select
        self.selected_features = []

    # 1. Filter Method (Correlation-based)
    def filter_features(self, X, y):
        """
            Filter Method: Correlation-based Feature Selection
            discription: This method calculates the correlation between each feature and the target variable.
            Formula: correlation = cov(X, y) / (std(X) * std(y))
        """
        if y is None:
            raise ValueError("Target vector 'y' is required for feature selection.")

        correlations = {}
        for feature in X.columns:
            cov = np.cov(X[feature], y)[0, 1] # cov return a 2x2 matrix : [[cov(x,x), cov(x,y)], [cov(y,x), cov(y,y)]]  so we need to get the value at [0,1] to get the cov(x,y)
            std_feature = np.std(X[feature])
            std_target = np.std(y)
            correlation = cov / (std_feature * std_target)
            correlations[feature] = correlation

        # Sorting features by absolute correlation values
        correlations_abs = {key: abs(value) for key, value in correlations.items()}
        sorted_features_filter = sorted(correlations_abs.items(), key=lambda x: x[1], reverse=True)   
        return sorted_features_filter

    # 2. Lasso Logistic Regression (Embedded Method)
    def lasso_logistic_regression(self, X, y, alpha, max_iter=1000, learning_rate=0.01):
        """
            Embedded Method: Lasso Logistic Regression
            discription: This method uses Lasso regularization to perform feature selection.
            Formula: L1 regularization = alpha * |coef|
        """
        X = np.column_stack([np.ones(X.shape[0]), X])  # Add intercept column (bias term)
        coef = np.zeros(X.shape[1])

        for _ in range(max_iter):
            y_pred = self.predict(X, coef)  # Sigmoid activation
            residuals = y_pred - y  # Calculate residuals

            gradient = np.dot(X.T, residuals) / X.shape[0]
            gradient[1:] += alpha * np.sign(coef[1:])  # Lasso penalty (L1 regularization)

            coef -= learning_rate * gradient

        return coef[1:], coef[0]  # Return coefficients excluding intercept and intercept itself

    def sigmoid(self, x):
        z = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict(self, X, coef, intercept=0):
        if len(coef) != X.shape[1]:
            raise ValueError("[Warning]Number of coefficients must match the number of features.")
        return self.sigmoid(X.dot(coef) + intercept)

    def embedded_method(self, X, y, alphas=None, cv=5):
        if alphas is None:
            alphas = np.logspace(-4, 4, 100) 

        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        mse_path = []

        for alpha in alphas:
            mse_fold = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                coef, intercept = self.lasso_logistic_regression(X_train, y_train, alpha)
                y_pred = self.predict(X_test, coef, intercept)  # Sigmoid prediction
                mse_fold.append(mean_squared_error(y_test, y_pred))

            mse_path.append(np.mean(mse_fold))

        best_alpha = alphas[np.argmin(mse_path)]
        print(f"Best alpha selected: {best_alpha}")

        coef, intercept = self.lasso_logistic_regression(X, y, best_alpha)
        selected_features_embedded = {X.columns[i]: abs(coef[i]) for i in range(len(coef)) if coef[i] != 0}

        # Sort features by their absolute coefficients (importance)
        sorted_features_embedded = sorted(selected_features_embedded.items(), key=lambda x: x[1], reverse=True)
        return sorted_features_embedded

    # 3. Recursive Feature Elimination (RFE)
    def rfe_feature_selection(self, X, y, n_features_to_select=None):
        if n_features_to_select is None:
            n_features_to_select = self.max_features

        def get_feature_importance(X, y):
            model = Logistic(epochs=1000)
            model.train_logistic_regression(X, y)
            if not model.weights_list:
                print(" Weighted list : ",model.weights_list)
                raise ValueError("[WARNING]Model weights_list is empty. Ensure the model is trained properly.")
            return np.abs(model.weights_list[-1])

        selected_features = list(X.columns)
        while len(selected_features) > n_features_to_select:
            importances = get_feature_importance(X[selected_features], y)
            least_important_feature_index = np.argmin(importances)
            del selected_features[least_important_feature_index]

        selected_features_rfe = {feature: 1 for feature in selected_features}
        sorted_features_rfe = sorted(selected_features_rfe.items(), key=lambda x: x[1], reverse=False)
        return sorted_features_rfe

    # 4. Concatenate and Sort Based on All Scores
    def concatenate_and_sort_scores(self, X, y):
        # Get feature selection scores from all methods
        filter_features = dict(self.filter_features(X, y))
        embedded_features = dict(self.embedded_method(X, y))
        rfe_features = dict(self.rfe_feature_selection(X, y))

        # Find common features across all methods
        common_features = set(filter_features.keys()) & set(embedded_features.keys()) & set(rfe_features.keys())

        # Calculate combined scores for all features
        feature_scores = {}
        for feature in X.columns:
            score = 0
            if feature in filter_features:
                score += filter_features[feature]
            if feature in embedded_features:
                score += embedded_features[feature]
            if feature in rfe_features:
                score += rfe_features[feature]
            feature_scores[feature] = score

        # Separate common features and remaining features
        common_features_scores = {f: feature_scores[f] for f in common_features}
        remaining_features_scores = {f: feature_scores[f] for f in feature_scores if f not in common_features}

        # Sort both groups separately
        sorted_common = sorted(common_features_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_remaining = sorted(remaining_features_scores.items(), key=lambda x: x[1], reverse=True)

        # Combine the sorted lists (common features first, then remaining)
        all_sorted_features = sorted_common + sorted_remaining

        # Select up to max_features
        self.selected_features = [feature for feature, score in all_sorted_features[:self.max_features]]

        print(f"[INFO]Number of selected features: {len(self.selected_features)}")
        print(f"[INFO]Selected features: {self.selected_features}")
        print(f"[INFO]Number of common features: {len(common_features)}")

        # Return the dataset with the selected features
        X_selected = X[self.selected_features]
        return X_selected

    # 5. Fit Method (perform union of all selected features)
    def fit(self, X, y):
        X_selected = self.concatenate_and_sort_scores(X, y)
        return X_selected

    # 6. Fit-Transform Method (train and apply transformation on data)
    def fit_transform(self, X, y):
        X_selected = self.concatenate_and_sort_scores(X, y)
        return X_selected

    # 7. Transform Method (apply transformation on unseen data)
    def transform(self, X):
        X_selected = X[self.selected_features]
        return X_selected
