import torch
import lightning
from minicons import cwe
import pandas as pd
import os
import glob
import re
import numpy as np
from tqdm import tqdm

from model import FFNModule, FeatureNormPredictor, FFNParams, TrainingParams



corpora = ["acl", "coca"]
data_dir = "/home/gsc685/data/collected_tokens/"
embedding_model = 'bert-base-uncased'
model_dir = '/home/shared/semantic_features/saved_models/bert_models_all/bert_to_buchanan_layer7.ckpt'
layer = 7


# helper function to batch process inputs
def batch_iterable(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


# iterate through the token files
for corpus in corpora:
    print("predicting features for ", corpus, " corpus")

    
    # token_files = glob.glob(data_dir + corpus + "/*.csv") 
    # just focus on one word for now
    token_files = [os.path.join(data_dir, corpus, 'human.csv')]

    for filename in token_files:
        print("predicting features for ", filename)


        
        """
        prepare the data
        """

        # pull the word we are predicting from the filename
        pattern = r'([a-zA-Z0-9_-]+)(?=\.csv$)'
        word = re.search(pattern, filename).group(0)
        print(word)
        
        tokens_path = os.path.join(data_dir, corpus, filename)
        tokens = pd.read_csv(tokens_path)
        tokens['word'] = word
        print(tokens.shape)
                
        # data as list of tuples
        data = list(zip(tokens['sentence'], tokens['word']))
        #print(data)
        
        """
        load the models 
        """
        lm = cwe.CWE(embedding_model)


        model = FeatureNormPredictor.load_from_checkpoint(
            checkpoint_path=model_dir,
            map_location=None
        ).to('cuda')

        print("model hyperparameters: ")
        for key,value in model.hparams.items():
            print("    {}: {}".format(key, value))
            
            
        # get length of output
        dummy = [ ("colorless green ideas sleep furiously", "ideas")]
        emb = lm.extract_representation(dummy, layer=layer)
        predicted= model(emb.cuda())
        squeezed = predicted.squeeze(0).cpu().detach().numpy()
        num_dims = squeezed.shape[0]
        print(num_dims)
        
        
        """
        run the model over the data
        """
    
        feats = [] * len(data)
        batch_size = 75
        #allfeats = np.empty((0, num_dims)).to('cuda') # zero rows and output-size columns
        #allfeats = torch.empty((0, num_dims), device='cuda')
        for i, batch in tqdm(enumerate(batch_iterable(data, batch_size))):
                       
            try:
                emb = lm.extract_representation(batch, layer=layer)
                predicted= model(emb.cuda())
                vecs = predicted.squeeze(0)
                
            except:
                vecs = torch.empty((batch_size,num_dims), device='cuda')
            #feats[i*batch_size:i*batch_size+batch_size] = vecs
            feats.append(vecs)
        
        feats = torch.cat(feats).detach().cpu().numpy()
        # buchanan
        # ratings_df = pd.read_csv('feature-norms/buchanan/cue_feature_words.csv', na_values=['na'])
        # # fill in 0 for na's
        # ratings_df.fillna(value=0, inplace=True)
        # feature_cols = ratings_df["translated"].unique()
        # print(feature_cols)
        
        
        outpath = os.path.join(data_dir, corpus, word + "_feature_vectors_bert_buchanan_layer7.txt")
        np.savetxt(outpath, feats)  # %d is used for integer formatting

        # for i in range(len(feature_cols)):
        #     print(feature_cols[i]," : ", squeezed[i].item())
