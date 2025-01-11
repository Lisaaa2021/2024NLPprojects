from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd

def annotation_aware(x):
    if x['labels'] == 1:
        text = "Offensive_  " + x['text']
    if x['labels'] == 0:
        text = "Non-offensive_  " + x['text']

    return text


def input_to_sen_embedding(input_df, add_label = True):
    """
    Given a input_df, output its sentence embedding dataframe with the 384d
    """
    sen_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    if not add_label:

        embeddings = sen_transformer.encode(list(input_df['text']))

        test_features = pd.DataFrame(embeddings)
        test_features.columns = test_features.columns.astype(str)
    else:
        input_df['text_annotation_aware'] = input_df.apply(annotation_aware, axis = 1)
        embeddings = sen_transformer.encode(list(input_df['text_annotation_aware']))

        test_features = pd.DataFrame(embeddings)
        test_features.columns = test_features.columns.astype(str)


    return test_features




def plotting(Y_true, Y_pred):

    '''
    Given a true label list and a predicted positive probability list,
    generate a ROC plot.
    This function is written with the help of ChatGPT
    '''
    roc_auc = roc_auc_score(list(Y_true), Y_pred)
    print(f"ROC-AUC Score: {roc_auc}")

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(Y_true, Y_pred)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random guessing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()
