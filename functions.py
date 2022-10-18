from sklearn import cluster, metrics
from sklearn import manifold, decomposition, preprocessing, pipeline
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import collections


def PCA_PC_plot(decomp_model):
    
    """
    Fonction permettant de visualiser la variance cumulée par PC d'une PCA.
    """
    
    explained_var = decomp_model.explained_variance_ratio_.cumsum()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(
        x=range(decomp_model.components_.shape[0]),
        y=explained_var
    )
    ax.set_xlabel('PC')
    ax.set_ylabel('Part cumulée de variance expliquée')
    ax.set_title('Part de variance expliquée par PC')
    ax.grid(visible=True, axis='both', color='lightgrey')
    ax.set_axisbelow(True)
    plt.show()
    

def assoc_fct(labels_true, labels_pred):
    
    """
    Fonction associant un ensemble de labels prédits aux labels réels, en considérant
    pour chaque label réel la classe prédit qui en contient la plus grande proportion.
    """
    
    labels_set = list(set(labels_true))
    labels_true = [labels_set.index(cat) for cat in labels_true]
    
    assert len(set(labels_pred)) == len(set(labels_true)),\
    ("Le nombre de labels prédits doit être strictement égal au nombre de labels réels")
    
    # Initialisation globale
    classes_done = []
    modes_done = []
    classes = list(set(labels_true))

    while True:
        # Initialisation pour chaque itération
        f_list = []
        m_list = []
        
        # Evaluation du mode par classe
        for i in classes:
            # Mapping de la classe itérée
            map_1 = [lab==i for lab in labels_true]
            # Exclusion des classes prédites déjà identifiées
            map_2 = [u not in modes_done for u in labels_pred]
            # Mapping combiné
            mapping = [m1 & m2 for m1, m2 in zip(map_1, map_2)]
            # Identification du mode et de la fréquence associée
            count = collections.Counter(labels_pred[mapping])
            count_list = dict(count)
            if len(count_list) == 0:
                mode = list(set(labels_pred) - set(modes_done))
                mode = mode[0]
                freq = 0
            else:
                mode_freq = max(count_list.items(), key=lambda x:x[1])
                mode = mode_freq[0]
                freq = mode_freq[1]
            # Ajout aux list
            m_list.append(mode)
            f_list.append(freq)
        
        # Sortie de la boucle si la condition est remplie
        if len(set(modes_done+m_list)) == len(set(labels_pred)):
            classes_done += classes
            modes_done += m_list
            
            sort_idx = np.argsort(classes_done)
            classes_done = [classes_done[n] for n in sort_idx]
            modes_done = [modes_done[n] for n in sort_idx]
            break
    
        # Identification de la classe réelle et du node associé avec la plus forte fréquence
        class_done = classes[np.argmax(f_list)]
        mode_done = m_list[np.argmax(f_list)]
        
        # Mise à jour des list
        classes_done.append(class_done)
        modes_done.append(mode_done)
        classes.remove(class_done)

    return classes_done, modes_done


def remap_fct(labels_true, labels_pred):
    
    """
    Fonction remplaçant les classes prédites par des classes associées.
    """
    
    classes_done, modes_done = assoc_fct(labels_true, labels_pred)
    
    labels_rev = labels_pred.copy()
    for i in range(len(classes_done)):
        value_pred = modes_done[i]
        mapping = labels_pred == value_pred
        labels_rev[mapping] = i
        
    return labels_rev


def tsne_ARI(features, categories, perplexity):
    
    """
    Fonction projetant des données en 2D à l'aide d'un t-SNE, puis réalisant un clustering
    en K-means sur ces données projetées.
    Retourne:
    - ARI calculé entre classes réelles et clusters du K-means
    - coordonnées du t-SNE
    - clusters associés au K-means
    """
    
    cat_labels = list(set(categories))
    tsne = manifold.TSNE(n_components=2,
                         perplexity=perplexity, 
                         init='random',
                         learning_rate=200,
                         random_state=1)
    X_tsne = tsne.fit_transform(features)

    # Détermination des clusters à partir des données après t-SNE
    cls = cluster.KMeans(n_clusters=len(cat_labels),
                         n_init=100,
                         random_state=2)
    cls.fit(X_tsne)
    ARI = metrics.adjusted_rand_score(categories, cls.labels_)
    
    return ARI, X_tsne, cls.labels_

    
def cluster_fct(features, categories, model_type=None, explained_var_target=.9):
    
    """
    Fonction réalisant :
    - Une réduction dimensionnelle si précisé (PCA ou SVD), avec un nombre de dimensions
    permettant d'expliquer le ratio 'explained_var_target' de la variance
    - Une étape de t-SNE sur ces données réduites suivie d'un clustering K-means,
    avec perplexité optimisée pour maximiser l'ARI résultant du K-means
    
    Retourne :
    - ARI calculé
    - Coordonnées du t-SNE
    - Labels / clusters résultant du K-means
    """
    
    assert model_type in [None, 'svd', 'pca'], ("Le modèle doit être None, 'svd', ou 'pca'")
    assert (0 < explained_var_target <= 1) or (explained_var_target == None),\
        ("Le ratio de variance doit être compris entre 0 et 1, ou bien None pour un choix manuel")
    
    if model_type != None:
        print("Etape de réduction dimensionnelle")
        if model_type == 'svd':
            # Réduction dimensionnelle au travers d'une SVD (méthode adaptée aux inputs sparse)
            model = decomposition.TruncatedSVD(random_state=1,
                                               n_components=min(features.shape))
        elif model_type == 'pca':
            # Réduction dimensionnelle au travers d'une ACP
            model = decomposition.PCA(random_state=1, n_components=explained_var_target)
            
        else:
            raise Exception("Modèle de réduction dimensionnelle non reconnu.")

        model.fit(features)

        if explained_var_target == None:
            # Plot pour déterminer le nombre de PC à conserver
            PCA_PC_plot(model)
            n_components = int(input('Nombre de PC à retenir :'))
            model.set_params(**{'n_components': n_components})
            
        elif model_type == 'svd':
            explained_var = model.explained_variance_ratio_.cumsum()
            n_components = sum(explained_var <= explained_var_target)
            model.set_params(**{'n_components': n_components})
            
        elif model_type == 'pca':
            n_components = model.n_components_  
            
        # Normalisation des données, le SVD ne résultant pas en des vecteurs normalisés
        norm = preprocessing.Normalizer()
        model = pipeline.Pipeline([('decomp_model', model), ('normalizer', norm)])
        
        print(f"{n_components:,d} features retenues, sur {min(features.shape):,d} au total.")
        features = model.fit_transform(features)
    
    # Projection en 2D au travers d'un t-SNE, avec optimisation de la perplexité
    print("Etape de tSNE")
    ARI_list = []
    perplexity_grid = [30, 50, 100]
    for per in tqdm.tqdm(perplexity_grid):
        ARI = tsne_ARI(features, categories, per)[0]
        ARI_list.append(ARI)
    
    perplexity = perplexity_grid[np.argmax(ARI_list)]
    print(f"Perplexité retenue : {perplexity}")
    
    print("Finalisation")
    ARI, X_tsne, labels = tsne_ARI(features, categories, perplexity)
    labels = remap_fct(categories, labels)
    
    return ARI, X_tsne, labels


def projected_plot(X_proj, categories, labels, ARI):
    
    """
    Fonction permettant de visualiser une projection en 2D, et affichant les catégories
    réelles ("categories") et les labels attribués ("labels"), ainsi que l'ARI.
    """
    
    cat_labels = list(set(categories))
    cat_num = [cat_labels.index(cat) for cat in categories]   
    
    fig = plt.figure(figsize=(15,6))
    
    ax = fig.add_subplot(1,2,1)
    scatter = ax.scatter(X_proj[:,0],
                         X_proj[:,1],
                         c=cat_num,
                         cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0],
              labels=cat_labels,
              loc="best",
              title="Catégorie")
    ax.set_title('Représentation des produits par catégories réelles')
    
    ax = fig.add_subplot(1,2,2)
    scatter = ax.scatter(X_proj[:,0],
                         X_proj[:,1],
                         c=labels,
                         cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0],
              labels=set(labels),
              loc="best",
              title="Clusters")
    ax.set_title('Représentation des produits par clusters')
    
    print(f"ARI : {ARI}")
    return fig    