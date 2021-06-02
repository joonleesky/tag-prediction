import wandb
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support as prfs


def plot_attn_weights(attn_weights):
    """
    attn_weights: (N, S, S)
    """
    plt.clf()
    N, S, _ = attn_weights.shape
    attn_weights = np.mean(attn_weights, 0)
    df_cm = pd.DataFrame(attn_weights, range(S), range(S))
    # plt.figure(figsize=(10,7))
    sns.set(font_scale=1.2)  # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 8})  # font size
    #wandb.log({"attn":plt})
    plt.show()
    plt.savefig(os.path.join(wandb.run.dir, 'attn_weights.png'),dpi=200)
    wandb.save('*png')


def custom_f1_score(true_list, pred_list):
    f1_list = [prfs(tr, pr, average='binary')[2] for (tr,pr) in zip(true_list, pred_list)]
    f1_score = sum(f1_list) / len(f1_list)

    return f1_score