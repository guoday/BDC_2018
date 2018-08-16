import pandas as pd
import tensorflow as tf
import utils
import os
from collections import Counter
from sklearn.model_selection import train_test_split
import sklearn.metrics
import cnn
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def create_hparams():
    return tf.contrib.training.HParams(
        k=8,
        batch_size=64,
        optimizer="sgd", #adam, sgd, or ada
        learning_rate=0.04,
        num_display_steps=100,
        num_eval_steps=1000000,
        eval_batch_size=1024,
        batch_num=10,
        dey_cont=4,
        pay_cont=1,
        l2=0.000000,
        vocab_threshold=10,   #词频低于vocab_threshold的词过滤掉  
        max_len=50,         #序列特征的最大长度
        batch_norm_decay=0.995,
        dropout=0.5,
        dim=100,
        layer_sizes=[128,128,128],
        activation=['relu','relu','relu'],
        init_method='he_normal',
        init_value=0.1,
        cnn_len=[3,4,5],
        filter_dim=100,
        data_path='data',
        model_path='model',
        sub_name='sub',
        single_features=['register_day_diff', 'register_type', 'device_type', 'launch_cont', 'launch_day_diff', 'create_cont', 'create_day_diff','activity_cont','activity_day_diff','activity_day_cont'],
        seq_features=['seq_launch','seq_activity'], # set None if empty
        num_features=None
        )

def build_vocabulary(train_df,hparams):
    print("build vocabulary.....")
    word2index={}
    for s in hparams.single_features:
        groupby_size=train_df.groupby(s).size()
        vals=dict(groupby_size[groupby_size>=hparams.vocab_threshold])
        word2index[s]={}
        for v in vals:
            word2index[s][v]=len(word2index[s])+2

    for s in hparams.seq_features:
        val_num=len(train_df[s].values[0].split()[0].split('_'))
        word2index[s]=[]
        word_list=[]
        for idx in range(val_num):
            word2index[s].append({})
            word_list.append([])
        for vals in train_df[s].values:
            for val in vals.split(' '):
                for idx,v in enumerate(val.split('_')):
                    word_list[idx].append(v)
        for idx,w_list in enumerate(word_list):
            w_list=Counter(w_list)
            for val in w_list:
                if w_list[val]>= hparams.vocab_threshold:
                    word2index[s][idx][val]=len(word2index[s][idx])+2
    print("done!")
    return word2index

if __name__ == '__main__':
    hparams=create_hparams()
    utils.print_hparams(hparams)
    if hparams.seq_features is None:
        hparams.seq_features=[]
    if hparams.num_features is None:
        hparams.num_features=[]
    train_df=pd.read_csv(os.path.join(hparams.data_path,'train.csv'))
    dev_df=pd.read_csv(os.path.join(hparams.data_path,'dev.csv'))
    test_df=pd.read_csv(os.path.join(hparams.data_path,'test.csv'))
    hparams.word2index=build_vocabulary(train_df,hparams)
    
    cnn.train(train_df,dev_df,test_df,hparams)  
        
