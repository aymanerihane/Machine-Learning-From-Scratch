import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

# Classe de nœud de l’arbre CHAID
class Node:
    def __init__(self, feature=None, category=None, children=None, value=None):
        self.feature = feature
        self.category = category
        self.children = children or {}
        self.value = value


class CHAID:

    def __init__(self, max_depth=3, min_samples_split=10, alpha=0.05):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.alpha = alpha
        self.tree = None

    # Approximation de la fonction Gamma pour les calculs
    def gamma(self,n):
        # Stirling's approximation for large n
        if n > 1:
            return np.sqrt(2 * np.pi * n) * (n / np.e) ** n
        else:
            return 1  # Gamma(1) = 1
    
    def chi_square_pdf(self,chi2_stat, dof):
        if chi2_stat <= 0 or dof <= 0:
            return 0
        try:
            # Ensure stability by checking if the power calculation or exponential term goes out of bounds
            if chi2_stat > 1e10:  # Large chi-square statistic
                return 0
            term = (chi2_stat ** (dof / 2 - 1)) * np.exp(-chi2_stat / 2) / (2 ** (dof / 2) * self.gamma(dof / 2))
            if np.isnan(term) or np.isinf(term):
                return 0
            return term
        except Exception as e:
            print(f"Error in chi-square PDF calculation: {e}")
            return 0

    def chi_square_cdf(self,chi2_stat, dof):
        # Calculate CDF by numerical integration using the trapezoid rule
        x_vals = np.linspace(0, chi2_stat, 1000)  # Use a fine range of values for numerical integration
        y_vals = np.array([self.chi_square_pdf(x, dof) for x in x_vals])
        
        # Use trapezoid method from scipy.integrate for more stable integration
        cdf_value = trapezoid(y_vals, x_vals)
        return cdf_value
    
    # Approximation de la fonction de distribution cumulative (CDF) du khi-deux pour calculer la p-value
    def chi_square_p_value(self,chi2_stat, dof):
        return 1 - self.chi_square_cdf(chi2_stat, dof)
    
    # Fonction pour calculer la statistique du khi-deux
    def chi_square_test(self,observed):
        # Calculer les fréquences attendues
        expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
        
        # Calculer la statistique du khi-deux
        chi2_stat = np.sum((observed - expected) ** 2 / expected)
        
        # Calculer les degrés de liberté
        dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
        
        # Calculer la p-value à partir de la statistique du khi-deux et des degrés de liberté
        p_value = self.chi_square_p_value(chi2_stat, dof)
        
        return chi2_stat, p_value

    # Fonction pour tester chaque variable
    def evaluate_feature(self,data, feature, target='survived'):
        observed = pd.crosstab(data[feature], data[target])
        chi2_stat, p_value = self.chi_square_test(observed.values)
        return chi2_stat, p_value

    # Fonction pour construire l'arbre CHAID
    def build_chaid_tree(self,data, target='survived', depth=0):
        # Stop criteria: single class or max depth reached
        if len(np.unique(data[target])) == 1 or depth == self.max_depth:
            value = data[target].mode().iloc[0]
            return Node(value=value)
            # Select the best feature for split
        best_feature = None
        best_chi2 = 0 # we don't need the chi2 value for now
        best_p_value = 1  # Initialize with a high p-value
        
        for feature in data.columns.drop(target):
            chi2_stat, p_value = self.evaluate_feature(data, feature, target)
            if p_value < best_p_value:
                best_p_value = p_value
                best_chi2 = chi2_stat
                best_feature = feature
        if best_feature is None:
            return Node(value=data[target].mode().iloc[0])
        
        node = Node(feature=best_feature)
        for category in data[best_feature].unique():
            subset = data[data[best_feature] == category]
            node.children[category] = self.build_chaid_tree(subset, target, depth + 1, self.max_depth)
        
        return node

    # Fonction pour prédire avec l'arbre CHAID
    def predict(self,node, instance):
        if node.value is not None:
            return node.value
        feature_value = instance[node.feature]
        if feature_value in node.children:
            return self.predict(node.children[feature_value], instance)
        return 0  # Default to not survived

    def predict_tree(self,tree, data):
        return data.apply(lambda row: self.predict(tree, row), axis=1)


    def plot_chaid_tree(self,node, pos=(0.5, 1), level_width=12.5, vert_gap=1, ax=None, parent_pos=None, category_label=None, depth=0):
        """
        Recursively plot a CHAID decision tree from a custom Node structure with increased spacing between groups of leaf nodes.

        Args:
            node (Node): The current node of the tree.
            pos (tuple): Position of the current node (x, y).
            level_width (float): Horizontal spacing between nodes, adjusted for each group of leaves.
            vert_gap (float): Vertical spacing between levels.
            ax (matplotlib axis): The axis to plot on.
            parent_pos (tuple): Position of the parent node (for connecting lines).
            category_label (str): Label for the branch from parent to this node.
            depth (int): Depth of the current node in the tree.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(26, 20))
            ax.set_aspect(1.0)
            ax.axis('off')
        
        # Display leaf node value with label "Survived" or "Not Survived"
        if node.value is not None:
            label = "Survived" if node.value == 1 else "Not Survived"
            ax.text(pos[0], pos[1], label, ha='center', va='center', 
                    bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'))
        else:
            # Internal node: show the feature name
            ax.text(pos[0], pos[1], f"Feature: {node.feature}", ha='center', va='center', 
                    bbox=dict(facecolor='orange', edgecolor='black', boxstyle='round,pad=0.5'))

        # Draw a line from the parent node to the current node if not root
        if parent_pos:
            ax.plot([parent_pos[0], pos[0]], [parent_pos[1], pos[1]], 'k-')
            # Label the category on the branch
            if category_label is not None:
                ax.text((parent_pos[0] + pos[0]) / 2, (parent_pos[1] + pos[1]) / 2, 
                        category_label, ha='center', va='center', color='red')

        # Function to calculate total number of leaf nodes in a subtree
        def count_leaf_nodes(node):
            if node.value is not None:
                return 1
            return sum(count_leaf_nodes(child) for child in node.children.values())

        # Recursively plot each child node with spacing adjustments for leaf groups
        if node.children:
            total_leaves = sum(count_leaf_nodes(child) for child in node.children.values())
            cumulative_offset = -level_width / 2  # Start on the left side of the current position

            for category, child_node in node.children.items():
                # Calculate the horizontal span for this child node group
                group_leaves = count_leaf_nodes(child_node)
                group_width = (group_leaves / total_leaves) * level_width
                
                # Center the group within its calculated width
                dx = cumulative_offset + group_width / 2
                dy = -vert_gap
                child_pos = (pos[0] + dx, pos[1] + dy)
                
                # Recursive call to plot the child node, increasing depth
                self.plot_chaid_tree(child_node, pos=child_pos, level_width=group_width, 
                                vert_gap=vert_gap, ax=ax, parent_pos=pos, category_label=category, depth=depth + 1)
                
                # Update the cumulative offset by the width of this group
                cumulative_offset += group_width


