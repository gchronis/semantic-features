import torch
import lightning
from minicons import cwe
import pandas as pd
import os
import glob
import re
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from model import FFNModule, FeatureNormPredictor, FFNParams, TrainingParams



corpus = "validation_coca"
data_dir = "/home/gsc685/data/"
out_dir = "/home/gsc685/data/features/"
embedding_model = 'roberta-base'
model_dir = '/home/shared/semantic_features/saved_models/new/roberta_models_all/roberta_to_buchanan_layer7.ckpt'
layer = 7
k_means_n = 5
n_samples = 1000 # max number of samples for each lemma


# helper function to batch process inputs
def batch_iterable(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]



token_file = '/home/gsc685/data/coca_sampled_tokenized_sentences.csv'
# just focus on one word for now
# token_files = [os.path.join(data_dir, corpus, 'human.csv')]
all_toks = pd.read_csv(token_file)
# join sentences to token dataset
#all_toks = all_toks.merge(sents, left_on='id', right_on='id', how='left')
unique_lemmas = all_toks['lemma'].unique()


"""
load the models 
"""
lm = cwe.CWE(embedding_model)

# feature prediction model
feature_model = FeatureNormPredictor.load_from_checkpoint(
    checkpoint_path=model_dir,
    map_location=None
).to('cuda')
print("feature model hyperparameters: ")
for key,value in feature_model.hparams.items():
    print("    {}: {}".format(key, value))
    
    
# get length of output
dummy = [ ("colorless green ideas sleep furiously", "ideas")]
emb = lm.extract_representation(dummy, layer=layer)
predicted= feature_model(emb.cuda())
squeezed = predicted.squeeze(0).cpu().detach().numpy()
num_dims = squeezed.shape[0]
#print(num_dims)

for lemma in tqdm(unique_lemmas):
    print("predicting features for ", lemma)

    
    """
    prepare the data
    """    
    
    # get tokens for this lemma
    df = all_toks[all_toks['lemma'] == lemma]

    if len(df) < 5:
        print("not enough data for this lemma, skipping")
        continue

    # filter sentences that are less than 300 in length
    df["sentence"] = df["sentence"].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii', 'ignore'))
    #df["sentence"] = df["sentence"].apply(lambda x: x.lower())

    # Take at most n sentences.
    df = df.sample(min(n_samples, len(df)), random_state=42) # use a fixed seed for reproducibility

    print("dataset size: ", df.shape)
            
    # data as list of tuples which is how minicons wants it
    data = list(zip(df['sentence'], df['word']))
    #print(data)
    

    
    
    """
    run the model over the data
    """

    feats = [] * len(data)
    embeddings = [] * len(data)
    nans = []
    batch_size = 75
    #allfeats = np.empty((0, num_dims)).to('cuda') # zero rows and output-size columns
    #allfeats = torch.empty((0, num_dims), device='cuda')
    print("predicting features and embeddings")
    for i, batch in enumerate(batch_iterable(data, batch_size)):

        try:
            embs = lm.extract_representation(batch, layer=layer)
            predicted= feature_model(embs.cuda())
            vecs = predicted.squeeze(0)
        except:
            print(batch)

            for s, t in batch:
                try: assert t in s
                except: print(s, t)
            embs = lm.extract_representation(batch, layer=layer)
            vecs = torch.empty((batch_size,num_dims), device='cuda')


        # figure out what's going wrong
        #print(embs.shape)
        # Check for NaN values
        nan_mask = torch.isnan(embs)
        # Print the locations where NaNs are present
        nan_locations = torch.nonzero(nan_mask)
        rows_with_nan = torch.any(nan_mask, dim=1).nonzero(as_tuple=True)[0].numpy()
        #print("rows with nan: ")
        #print(rows_with_nan)

        for k in rows_with_nan:
            print("can't get emb for ")
            print(batch[k])

        nan_indices = [row + (batch_size*i) for row in rows_with_nan] # get indices of problem data
                                        
        # print(nan_locations.shape)
        # if nan_locations.shape != torch.Size([0,2]):
            
        #     problem = nan_locations[0][0]
        #     print("cant get embedding for")
        #     print(batch[problem])

        #     print(torch.unique(nan_locations[:,0], dim=0)) # get the first column and then extract unique values
        #     raise(Exception("heiowfie")) 
        #     nan_locations = (batch_size*i) * torch.unique(nan_locations, dim=0).numpy() # get indices of problem data
        #     print(f"NaN values are located at: {nan_locations}")

        # else:
        #     nan_locations = []
        nans += nan_indices

        feats[i*batch_size:i*batch_size+batch_size] = vecs.detach().cpu().numpy()
        embeddings[i*batch_size:i*batch_size+batch_size] = embs.detach().cpu().numpy()
        #feats.append(vecs)
        #embeddings.append(embs)
    #feats = torch.cat(feats).detach().cpu().numpy()
    #print(embeddings.shape)
    print("couldnt get embs for data at indices", nans )

    embeddings = np.delete(embeddings, nans, axis=0)
    try:
        feats = np.delete(feats, nans, axis = 0)
    except:
        print("skipping ", lemma, ": no time to figure out whats happening here")
        #print("feature shape: ", feats)
        #print("embedding shape: ", embeddings.shape)
        #print("nan shape: ", len(nans))
        continue 
    good_indices = np.delete( np.arange(len(data)), nans, axis =0)

    print(len(embeddings))
    print(len(feats))

    """
    cluster embeddings
    :embeddings: an np.ndarray of bert embeddings of dimension n_words, n_dims] (i.e. [200,768])
    :return: an 1D np.ndarray containing cluster ids of shape N_samples
    """

    print("initializing clusterizer")
    kmeans_obj = KMeans(n_clusters=k_means_n, n_init=10)

    if len(embeddings) < k_means_n and len(embeddings) >= 1:
        print("not enough data to cluster, using all data points as clusters")   
        kmeans_obj = KMeans(n_clusters=len(embeddings), n_init=10)
    elif len(embeddings) < 1:
        print("no data to cluster, skipping")
        continue
        #kmeans_obj = KMeans(n_clusters=k_means_n, n_init=10)

    kmeans_obj.fit(embeddings)

    #label_list = kmeans_obj.labels_
    #cluster_centroids = kmeans_obj.cluster_centers_

    clusters = kmeans_obj.fit_predict(embeddings)
    print("number of clustered data points: ", len(clusters))


    """
    normalize features 
    """

        # get the names of the features
    buchanan_norms = pd.read_csv('/home/gsc685/semantic-features/feature-norms/buchanan/cue_feature_words.csv')
    name_col = 'translated'
    freq_col = 'frequency_'+name_col
    feature_labels = buchanan_norms[name_col].unique().tolist()
    feature_labels.sort() #FUUUUUUUUCJJJJJJJJKKKKK


    ids = []
    sources = []
    #sent = []
    lemmas = []
    senses = []
    word_forms = []
    poses = []
    cluster = []
    feature = []
    predicted_value = []

    for cluster_index, i in enumerate(good_indices): # you have lists of different indexing. clusters and embeddings and features are all squished together and indexed wrong

        row = df.iloc[i] # row info for token level data
        j = 0
        feature_vec = feats[cluster_index] # get the features for this sample

        for value in feature_vec:
            #print(feature_labels[j])
            ids.append(row.token_id) # the sentence id
            #lemmas.append(row.lemma)
            #word_forms.append(row.word)
            #poses.append(row.pos_coarse)
            #sent.append(row.sentence)
            cluster.append(clusters[cluster_index])
            feature.append(feature_labels[j])
            predicted_value.append(value)
            j+=1
        j=0

    tidy_df = pd.DataFrame.from_records(
        {"token_id": ids,
        #"sent": sent, 
        #"lemma": lemmas, 
        #"pos": poses,
        "cluster": cluster, 
        "feature": feature, 
        "predicted_value": predicted_value}
    )

    
    # save to disk
    outpath = os.path.join(out_dir, corpus, lemma + "_feature_vectors_roberta_buchanan_layer7.csv")
    tidy_df.to_csv(outpath) 