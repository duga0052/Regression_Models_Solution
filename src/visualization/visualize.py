import matplotlib.pyplot as plt
from sklearn import tree
import logging

def plot_decision_tree(model, feature_names, file_name='tree.png'):
    try:
        plt.figure(figsize=(20,10))
        tree.plot_tree(model, feature_names=feature_names, filled=True)
        plt.savefig(file_name, dpi=300)
        plt.show()
        logging.info("Decision Tree plotted successfully.")
    except Exception as e:
        logging.error(f"Error in plotting Decision Tree: {e}")
        raise