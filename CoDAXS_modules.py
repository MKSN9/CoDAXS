#this module contains all necessary functions for the CoDAXS jupyter notebook

#necessary packages and modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans


# Zero replacement
def zero_replacement(df):
    """
    Replaces 0 values in all columns of a DataFrame (excluding the index) with 999999, 
    and then replaces 999999 values with half of the column's minimum value.
    Also prints how many zeros were replaced in each column.
    
    Parameters:
    df (pd.DataFrame): The dataframe of selected elements
    
    Returns:
    pd.DataFrame: The dataframe of selected elements without 0
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Dictionary to store the number of zeros replaced in each column
    zero_counts = {}

    for col in df_copy.columns:
        if df_copy[col].dtype != 'object':  
            zero_count = (df_copy[col] == 0).sum()
            zero_counts[col] = zero_count

            # Step 1: Replace 0 values with 999999
            df_copy[col] = df_copy[col].replace(0, 999999)

            # Step 2: Replace 999999 values with half of the column minimum
            col_min_half = df_copy[col].min() / 2
            df_copy[col] = df_copy[col].replace(999999, col_min_half)

    # Print the number of zeros replaced 
    for col, count in zero_counts.items():
        print(f"{count} zeros were replaced in '{col}'")

    return df_copy
#centred-log-ratio transformation
def clr_transformation(df):
    """
    Performs a centered log-ratio (CLR) transformation on the input DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing element data.
    
    Returns:
    pd.DataFrame: The DataFrame after the CLR transformation.
    """
    # Step 1: Compute the geometric mean across rows
    g_edf = gmean(df, axis=1)  # Calculate geometric mean along rows
    g_edf = pd.DataFrame(g_edf, columns=["geomean"])  # Create DataFrame for geometric mean
    g_edf.index = df.index  # Retain the same index as the original elements DataFrame

    # Step 2: Compute the log-ratio between elements and their geometric mean
    clr = np.log(df / g_edf.values)  # Element-wise division and log transformation

    # Step 3: Center the log-ratio by subtracting the column-wise mean
    ele_clr = clr - np.mean(clr, axis=0)

    return ele_clr

#Singular value decomposition of elements_clr dataframe
def svd(elements_clr):

    # Perform SVD decomposition
    U, s, VT = np.linalg.svd(elements_clr, full_matrices=False)
    
    # Calculate explained variance
    expl_var = s**2 / np.sum(s**2)
    explained_variance = pd.DataFrame(expl_var, index=[f'{i+1}' for i in range(len(s))], columns=['explained variance'])

    loadings = pd.DataFrame(VT.T*s, index=elements_clr.columns, columns=[f'PC{i+1}' for i in range(len(s))])
    
    scores = pd.DataFrame(U * s, index=elements_clr.index, columns=[f'PC{i+1}' for i in range(len(s))])
    
    return explained_variance, loadings, scores


#agglomerative clustering
def agglomerative_clustering(scores, elements_clr, n_clusters, colors, output_folder, output_format, savefig=True, linewidth=1):
    # Create a new directory for agglomerative clustering results
    agglomerative_folder = output_folder / 'agglomerative_clustering'
    if not agglomerative_folder.exists():
        agglomerative_folder.mkdir(parents=True, exist_ok=True)

    # Perform the Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(scores)
    labels = clustering.labels_

    # Create a DataFrame with the index of scores and cluster labels
    agg_clusters = pd.DataFrame({'cluster': labels}, index=scores.index)
    agg_clusters.index.name = scores.index.name

    # First plot: PC1 vs PC2 scatter with cluster colors, legend, and gridlines at 0
    fig, ax = plt.subplots()
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    scatter = ax.scatter(scores['PC1'], scores['PC2'], c=[colors[j] for j in labels], s=5)
    ax.set_aspect('equal')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=8, label=f'Cluster {i+1}')
               for i in range(n_clusters)]
    ax.legend(handles=handles, title="Clusters", loc="center left", bbox_to_anchor=(1, 0.5))

    # Save and show the plot
    if savefig:
        plt.savefig(f"{agglomerative_folder}/cluster_scatter_plot_PC1_vs_PC2.{output_format}", bbox_inches='tight')
    plt.show()

    # Second plot: Depth vs clusters
    fig, ax = plt.subplots(figsize=(1, 10))
    y_label = elements_clr.index.name if elements_clr.index.name else 'Depth (xx)'
    ax.scatter(np.zeros(len(scores)), scores.index, c=[colors[j] for j in labels], marker="_", s=2500, linewidths=linewidth)
    ax.set_ylabel(y_label)
    ax.set_xticks([])
    ax.invert_yaxis()
    if savefig:
        plt.savefig(f"{agglomerative_folder}/cluster_depth_plot.{output_format}", bbox_inches='tight')
    plt.show()

    global_min_x, global_max_x = float('inf'), float('-inf')
    global_min_y, global_max_y = float('inf'), float('-inf')

    cluster_loadings = []  # Store SVD loadings and explained variance for each cluster

    for i in range(n_clusters):
        indices = np.where(labels == i)[0]
        if len(indices) > 0:
            # Perform SVD once per cluster
            U, s, VT = np.linalg.svd(elements_clr.iloc[indices], full_matrices=False)
            loadings = pd.DataFrame((VT.T * s), index=elements_clr.columns, columns=[f'PC{k+1}' for k in range(len(s))])
            explained_variance = s**2 / np.sum(s**2)

            # Store loadings and explained variance for later use
            cluster_loadings.append((loadings, explained_variance))

            # Update global min/max for consistent axis limits
            global_min_x = min(global_min_x, loadings['PC1'].min())
            global_max_x = max(global_max_x, loadings['PC1'].max())
            global_min_y = min(global_min_y, loadings['PC2'].min())
            global_max_y = max(global_max_y, loadings['PC2'].max())

    # Add padding to the global axis limits
    padding_x = 0.1 * (global_max_x - global_min_x)
    global_min_x -= padding_x
    global_max_x += padding_x

    padding_y = 0.1 * (global_max_y - global_min_y)
    global_min_y -= padding_y
    global_max_y += padding_y

    # Plot biplots for each cluster using the precomputed SVD loadings and explained variance
    for i, (loadings, explained_variance) in enumerate(cluster_loadings):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(loadings['PC1'], loadings['PC2'], s=4, color='Green')
        for j, row in loadings.iterrows():
            ax.plot([0, row['PC1']], [0, row['PC2']], color='k')
            ax.annotate(j, (row['PC1'], row['PC2']), xytext=(1, 1), textcoords='offset points', ha='right', fontsize=12)
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_xlabel(f'PC1 ({(explained_variance[0] * 100):.2f}%)', fontsize=15)
        ax.set_ylabel(f'PC2 ({(explained_variance[1] * 100):.2f}%)', fontsize=15)
        ax.set_title(f'Cluster {i+1} Biplot')
        ax.set_xlim(global_min_x, global_max_x)
        ax.set_ylim(global_min_y, global_max_y)
        plt.tight_layout()
        if savefig:
            plt.savefig(f"{agglomerative_folder}/biplot_cluster_{i+1}.{output_format}", bbox_inches='tight')
        plt.show()

    # Save the cluster labels DataFrame
    if savefig:
        agg_clusters.to_csv(f"{agglomerative_folder}/agglomerative_clustering_labels.csv")

    # Dendrogram plot
    Z = linkage(scores, method='ward')  # Use Ward's method for the hierarchical clustering linkage
    plt.figure(figsize=(10, 5))
    dendrogram(Z, labels=scores.index, color_threshold=0, no_labels=True)  
    plt.title("Dendrogram of Agglomerative Clustering")

    plt.ylabel("Ward's Distance")

    # Save dendrogram
    if savefig:
        plt.savefig(f"{agglomerative_folder}/dendrogram.{output_format}", bbox_inches='tight')

    plt.show()

    return agg_clusters

#kmeans elbow plot
def kmeans_elbow_plot(scores, output_folder,output_format, savefig=True, max_clusters=10):

    if isinstance(scores, pd.DataFrame):
        scores = scores.values
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scores)
        inertias.append(kmeans.inertia_)

    # Plot the elbow plot
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, max_clusters + 1), inertias, color='b', alpha=0.7)
    plt.title('Elbow Plot for KMeans Clustering')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (Within-Cluster Sum of Squared Errors)')
    plt.xticks(range(1, max_clusters + 1))
    plt.grid(False)
    if savefig:
        plt.savefig(f"{output_folder}/elbow_plot.{output_format}", bbox_inches='tight')
    plt.show()


#k-means clustering
def kmeans_clustering(scores, elements_clr, n_clusters, colors, output_folder, output_format, savefig=True, linewidth=1):
    # Create a new directory for kmeans clustering results
    kmeans_folder = output_folder / 'kmeans_clustering'
    if not kmeans_folder.exists():
        kmeans_folder.mkdir(parents=True, exist_ok=True)

    # Perform KMeans clustering
    clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(scores)
    labels = clustering.labels_

    # Create a DataFrame with the index of scores and cluster labels
    kmeans_clusters = pd.DataFrame({'cluster': labels}, index=scores.index)
    kmeans_clusters.index.name = scores.index.name

    # First plot: PC1 vs PC2 scatter with cluster colors, legend, and gridlines at 0
    fig, ax = plt.subplots()
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    scatter = ax.scatter(scores['PC1'], scores['PC2'], c=[colors[j] for j in labels], s=5)
    ax.set_aspect('equal')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=8, label=f'Cluster {i+1}')
               for i in range(n_clusters)]
    ax.legend(handles=handles, title="Clusters", loc="center left", bbox_to_anchor=(1, 0.5))

    # Save and show the plot
    if savefig:
        plt.savefig(f"{kmeans_folder}/cluster_scatter_plot_PC1_vs_PC2.{output_format}", bbox_inches='tight')
    plt.show()

    # Second plot: Depth vs clusters
    fig, ax = plt.subplots(figsize=(1, 10))
    y_label = elements_clr.index.name if elements_clr.index.name else 'Depth (xx)'
    ax.scatter(np.zeros(len(scores)), scores.index, c=[colors[j] for j in labels], marker="_", s=2500, linewidths=linewidth)
    ax.set_ylabel(y_label)
    ax.set_xticks([])
    ax.invert_yaxis()
    if savefig:
        plt.savefig(f"{kmeans_folder}/cluster_depth_plot.{output_format}", bbox_inches='tight')
    plt.show()

    # Determine the global min/max for the biplot axes
    global_min_x, global_max_x = float('inf'), float('-inf')
    global_min_y, global_max_y = float('inf'), float('-inf')

    for i in range(n_clusters):
        indices = np.where(labels == i)[0]
        if len(indices) > 0:
            U, S, VT = np.linalg.svd(elements_clr.iloc[indices], full_matrices=False)
            loadings = pd.DataFrame((VT.T * S), index=elements_clr.columns, columns=[f'PC{i+1}' for i in range(len(S))])

            # Update global min/max for consistent axis limits
            global_min_x = min(global_min_x, loadings['PC1'].min())
            global_max_x = max(global_max_x, loadings['PC1'].max())
            global_min_y = min(global_min_y, loadings['PC2'].min())
            global_max_y = max(global_max_y, loadings['PC2'].max())

    # Add some padding to the global axis limits
    padding = 0.1 * (global_max_x - global_min_x)
    global_min_x -= padding
    global_max_x += padding
    padding = 0.1 * (global_max_y - global_min_y)
    global_min_y -= padding
    global_max_y += padding

    # Biplots for each cluster
    for i in range(n_clusters):
        indices = np.where(labels == i)[0]
        if len(indices) > 0:
            U, s, VT = np.linalg.svd(elements_clr.iloc[indices], full_matrices=False)
            expl_var = s**2 / np.sum(s**2)
            explained_variance = pd.DataFrame(expl_var, index=[f'{i+1}' for i in range(len(s))], columns=['explained variance'])
            loadings = pd.DataFrame((VT.T * s), index=elements_clr.columns, columns=[f'PC{i+1}' for i in range(len(s))])

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(loadings['PC1'], loadings['PC2'], s=4, color='Green')
            for j, row in loadings.iterrows():
                ax.plot([0, row['PC1']], [0, row['PC2']], color='k')
                ax.annotate(j, (row['PC1'], row['PC2']), xytext=(1, 1), textcoords='offset points', ha='right', fontsize=12)
            ax.set_aspect('equal')
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5)
            ax.set_xlabel(f'PC1 ({(explained_variance.loc["1", "explained variance"] * 100):.2f}%)', fontsize=15)
            ax.set_ylabel(f'PC2 ({(explained_variance.loc["2", "explained variance"] * 100):.2f}%)', fontsize=15)
            ax.set_title(f'Cluster {i+1} Biplot')
            ax.set_xlim(global_min_x, global_max_x)
            ax.set_ylim(global_min_y, global_max_y)
            plt.tight_layout()
            if savefig:
                plt.savefig(f"{kmeans_folder}/biplot_cluster_{i+1}.{output_format}", bbox_inches='tight')
            plt.show()

    # Save the cluster labels DataFrame
    if savefig:
        kmeans_clusters.to_csv(f"{kmeans_folder}/kmeans_clustering_labels.csv")

    return kmeans_clusters
