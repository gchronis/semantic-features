import torch
import lightning
from minicons import cwe
import pandas as pd

from model import FFNModule, FeatureNormPredictor, FFNParams, TrainingParams

def test(): 
    device = torch.device("cuda:0")
    model = FeatureNormPredictor.load_from_checkpoint(
        #checkpoint_path='saved_models/albert8_to_binder_opt_stop.ckpt',
        checkpoint_path = '/home/shared/semantic_features/saved_models/new/albert_models_all/albert_to_binder_layer8.ckpt',
        map_location=None
    ).to(device)
    
    print("model hyperparameters: ")
    for key,value in model.hparams.items():
        print("    {}: {}".format(key, value))

    # get a sample embedding to test
    data = [
        ("the crow is an aggressive bird", "crow")
    ]
    lm = cwe.CWE('albert-xxlarge-v2')
    emb = lm.extract_representation(data, layer=11)
    predicted= model(emb.to(device))
    squeezed = predicted.squeeze(0)
    print(squeezed.shape)

    ratings_df = pd.read_csv('feature-norms/binder/WordSet1_Ratings.csv', na_values=['na'])
    # fill in 0 for na's
    ratings_df.fillna(value=0, inplace=True)
    feature_cols = ratings_df.iloc[:,5:70].columns
    for i in range(len(feature_cols)):
        print(feature_cols[i]," : ", squeezed[i].item())


def test_roberta():
    device = torch.device("cuda:0")
    model = FeatureNormPredictor.load_from_checkpoint(
        #checkpoint_path='saved_models/albert8_to_binder_opt_stop.ckpt',
        checkpoint_path = '/home/shared/semantic_features/saved_models/new/roberta_models_all/roberta_to_buchanan_layer7.ckpt',
        map_location=None
    ).to(device)
    
    print("model hyperparameters: ")
    for key,value in model.hparams.items():
        print("    {}: {}".format(key, value))

    # get a sample embedding to test
    data = [
        ("the crow is an colorful bird with black soft wings", "crow")
    ]
    lm = cwe.CWE('roberta-base')
    emb = lm.extract_representation(data, layer=8)
    predicted= model(emb.to(device))
    squeezed = predicted.squeeze(0)
    print(squeezed.shape)



        # get the names of the features
    buchanan_norms = pd.read_csv('/home/gsc685/semantic-features/feature-norms/buchanan/cue_feature_words.csv')
    name_col = 'translated'
    freq_col = 'frequency_'+name_col
    feature_labels = buchanan_norms[name_col].unique().tolist()
    feature_labels.sort()


    zipped = zip(feature_labels, [x.item() for x in squeezed])

    # Sort by the second element in each tuple
    sorted_zipped = sorted(zipped, key=lambda x: x[1])
    print(sorted_zipped[-30:] ) # top 30 features

    #for i in range(len(feature_cols)):
    #    print(feature_cols[i]," : ", squeezed[i].item())


def test_roberta_all_layers():
    device = torch.device("cuda:0")
    model = FeatureNormPredictor.load_from_checkpoint(
        #checkpoint_path='saved_models/albert8_to_binder_opt_stop.ckpt',
        checkpoint_path = '/home/shared/semantic_features/saved_models/new/roberta_models_all/roberta_to_buchanan_layer7.ckpt',
        map_location=None
    ).to(device)
    
    print("model hyperparameters: ")
    for key,value in model.hparams.items():
        print("    {}: {}".format(key, value))

     # get the names of the features
    buchanan_norms = pd.read_csv('/home/gsc685/semantic-features/feature-norms/buchanan/cue_feature_words.csv')
    name_col = 'translated'
    freq_col = 'frequency_'+name_col
    feature_labels = buchanan_norms[name_col].unique().tolist()
    feature_labels.sort()

    # get a sample embedding to test
    data = [
        ("the crow is an colorful bird with black soft wings", "crow")
    ]
    lm = cwe.CWE('roberta-base')
    emb = lm.extract_representation(data, layer='all')
    print(len(emb))

    # run the model
    for i, layer_emb in enumerate(emb):
        predicted= model(layer_emb.to(device))
        squeezed = predicted.squeeze(0)
        print(squeezed.shape)

        print("Layer ", i)

        zipped = zip(feature_labels, [x.item() for x in squeezed])

        # Sort by the second element in each tuple
        sorted_zipped = sorted(zipped, key=lambda x: x[1])
        print(sorted_zipped[-10:])

    #for i in range(len(feature_cols)):
    #    print(feature_cols[i]," : ", squeezed[i].item())

if __name__ == '__main__':
    test_roberta_all_layers()
    test_roberta()
    test()
