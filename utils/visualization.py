import numpy as np
import pandas as pd
import seaborn as sns
import torch

from sklearn.decomposition import PCA
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

        # return {
        #     'acc_matrix': self.acc_matrix,
        #     'accs': self.accs,
        #     'accs_iplus1': self.accs_iplus1,
        #     'fwt': self.fwt,
        #     'bwt': self.bwt,
        #     'forgetting': self.forgetting,
        # }

def vis_acc_mat(mat):
    import matplotlib.pyplot as plt
    # put the xlabel on top.
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    df_cm = pd.DataFrame(mat, columns=np.arange(1, len(mat)+1), index=np.arange(1, len(mat)+1))
    df_cm.index.name = "Domain ID"
    df_cm.columns.name = "Domain ID"
    plt.figure(figsize=(20,20))

    sns.heatmap(
        df_cm, 
        # cmap="Blues", 
        annot=True, 
        annot_kws={"size": 16},
        vmin=0,
        vmax=100
    )  # font sizea
    return plt

def vis_curves(curves, names):
    import matplotlib.pyplot as plt
    df_cm = pd.DataFrame(np.vstack(curves).T, columns=names, index=np.arange(1, len(curves[0])+1))

    plt.figure(figsize=(20,20))

    sns.lineplot(data=df_cm)
    return plt

def vis_embeddings(dic):
    """Used with get_embeddings function, to visualize the embedding on all test domains."""
    import matplotlib.pyplot as plt

    # 3 types of the plots:
    # domain id as label
    # label as label
    # prediction as label.

    data_feats, data_labels, data_preds, data_dom_ids = [], [], [], []
    for k in dic.keys():
        data_feats.append(dic[k]['embeddings'])
        data_labels.append(dic[k]['labels'])
        data_preds.append(dic[k]['preds'])
        data_dom_ids.append(np.full((dic[k]['preds'].shape[0],), k))
    data_feats, data_labels, data_preds, data_dom_ids = np.vstack(data_feats), np.hstack(data_labels), np.hstack(data_preds), np.hstack(data_dom_ids)

    # assemble the pd.DataFrame for sns scatter.
    df_cm = pd.DataFrame(
        np.hstack([
            data_feats, 
            data_labels.reshape(-1,1), 
            data_preds.reshape(-1,1), 
            data_dom_ids.reshape(-1,1)
        ]), 
        columns=['PC1', 'PC2', 'labels', 'predictions', 'domain ids'], 
        index=np.arange(len(data_feats))
    )

    plt.figure(figsize=(24,8))
    marker_size = 10
    alpha = 0.5

    plt.subplot(131)
    sns.scatterplot(data=df_cm, x='PC1', y='PC2', hue='labels', style='labels', palette='deep', s=marker_size, alpha=alpha)

    plt.subplot(132)
    sns.scatterplot(data=df_cm, x='PC1', y='PC2', hue='predictions', style='predictions', palette='deep', s=marker_size, alpha=alpha)
    
    plt.subplot(133)
    sns.scatterplot(data=df_cm, x='PC1', y='PC2', hue='domain ids', style='domain ids', palette='deep', s=marker_size, alpha=alpha)

    return plt
    

def get_embeddings(model: ContinualModel, dataset: ContinualDataset, n=None):
    """get 2-dimensional embeddings of all domains' test set"""
    # visualize the embedding space not for all testing domains
    if n is None:
        n = len(dataset.test_loaders)
    assert n <= len(dataset.test_loaders) and n > 0

    status = model.net.training
    model.net.eval()

    dic = {} # return result
    for dom_id, test_loader in enumerate(dataset.test_loaders[:n]):
        dom_embeddings, dom_labels, dom_preds = [], [], []
        for data in test_loader:
            with torch.no_grad():
                inputs, labels, _ = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)

                # different from evaluate.
                logits, _, feats = model.net(inputs, returnt='all')
                _, preds = torch.max(logits.data, 1)

                # record everything to be visualized
                dom_embeddings.append(feats.cpu().numpy())
                dom_labels.extend(labels.cpu().numpy())
                dom_preds.extend(preds.cpu().numpy())

        # add everything to the dictionary.
        dic[dom_id] = {
            'embeddings': np.vstack(dom_embeddings),
            'labels': np.array(dom_labels, dtype=int),
            'preds': np.array(dom_preds, dtype=int)
        }
    
    # if the original embedding dimension is smaller than 2
    # ready to return.
    if dom_embeddings[0].shape[1] <= 2:
        return dic

    # PCA for 2-visualization
    ks, lens, total_emb = [], [], []
    for k, v in dic.items():
        ks.append(k)
        lens.append(v['embeddings'].shape[0])
        total_emb.append(v['embeddings'])
    total_emb = np.vstack(total_emb)

    # pca from sklearn
    pca = PCA(n_components=2)
    ld_embs = pca.fit_transform(total_emb)
    # slicing and indexing back
    for i, k in enumerate(ks):
        size = lens[i]
        lo, hi = sum(lens[:i]), sum(lens[:i]) + size
        dic[k]['embeddings'] = ld_embs[lo:hi]
    
    model.net.train(status)
    
    return dic