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
layer = 7


model_dir = '/home/shared/semantic_features/saved_models/bert_models_all/bert_to_buchanan_layer{}.ckpt'.format(layer)

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
        run the model over the data to get embeddings and features
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
        
        ## create numpy matrix with N_samples x N_features
        feats = torch.cat(feats).detach().cpu().numpy()

        """
        run clustering model to get clusters from embeddings
        """
        

        """
        normalize 
        """

         # get the names of the features
        buchanan_norms = pd.read_csv('/home/gsc685/semantic-features/feature-norms/buchanan/cue_feature_words.csv')
        name_col = 'translated'
        freq_col = 'frequency_'+name_col
        feature_labels = buchanan_norms[name_col].unique().tolist()



        ids = []
        sources = []
        sent = []
        cluster = []
        feature = []
        predicted_value = []
        for i, (index, row) in enumerate(sample_df.iterrows()):

            j = 0
            feature_vec = feats[i] # get the features for this sample

            for value in feature_vec:
                #print(feature_labels[j])
                ids.append(i)
                sent.append(row.sentence)
                sources.append(row.source)
                cluster.append(clusters[i])
                feature.append(feature_labels[j])
                predicted_value.append(value)
                j+=1
            j=0

        tidy_df = pd.DataFrame.from_records(
            {"id": ids,
             "source": sources,
            "rent": sent, 
            "word": word, 
            "cluster": cluster, 
            "feature": feature, 
            "predicted_value": predicted_value}
        )

        tidy_df.to_csv('./tidy_feature_predictions/{}/{}/{}_buchanan_layer_7.csv'.format(model_name,source,word)) 
        
        outpath = os.path.join(data_dir, corpus, word + "_feature_vectors_bert_buchanan_layer" + layer + ".txt")
        np.savetxt(outpath, feats)  # %d is used for integer formatting

        # for i in range(len(feature_cols)):
        #     print(feature_cols[i]," : ", squeezed[i].item())
