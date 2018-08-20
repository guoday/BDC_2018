import pandas as pd
import tensorflow as tf
import os
from collections import Counter
from sklearn.model_selection import train_test_split
import sklearn.metrics
import random
import numpy as np
import random
from sklearn import preprocessing
def create_hparams():
    return tf.contrib.training.HParams(
        k=16,
        batch_size=64,
        optimizer="sgd", #adam, sgd, or ada
        learning_rate=0.04,
        num_display_steps=1000,
        num_eval_steps=10000,
        eval_batch_size=1024,
        batch_num=10,
        dey_cont=4,
        pay_cont=3,
        l2=0.000000,
        vocab_threshold=10,   #词频低于vocab_threshold的词过滤掉  
        max_len=50,         #序列特征的最大长度
        batch_norm_decay=0.995,
        dropout=0.2,
        dim=16,
        cross_layer_sizes=[16,16],
        layer_sizes=[128,128,128],
        activation=['relu','relu','relu'],
        cross_activation='identity',
        init_method='he_normal',
        init_value=0.1,
        forget_bias=1.0,
        cnn_len=[2,3,4],
        filter_dim=16,
        data_path='/home/kesci/',
        model_path='/home/kesci/input/model/',
        sub_name='cnn',
        kfold=False,
        single_features=['register_day_diff', 'register_type', 'device_type', 
        'launch_cont', 'launch_day_diff', 'create_cont', 'create_day_cont', 
        'create_day_diff', 'activity_cont', 'activity_day_diff', 'activity_day_cont',],
        seq_features=['active_seq', 'create_seq', 'activity_seq'], # set None if empty
        pretrain=None,
        multi_features=None,
        num_features=['active_day_fft_min', 'active_day_fft_max', 'active_day_fft_mean', 
        'active_day_fft_var', 'active_day_fft_median' ],
        )

def build_vocabulary(train_df,hparams,dev_df,test_df):
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
        if s in hparams.pretrain:
            for vals in dev_df[s].values:
                for val in vals.split(' '):
                    for idx,v in enumerate(val.split('_')):
                        word_list[idx].append(v)
            for vals in test_df[s].values:
                for val in vals.split(' '):
                    for idx,v in enumerate(val.split('_')):
                        word_list[idx].append(v)
        for idx,w_list in enumerate(word_list):
            w_list=Counter(w_list)
            for val in w_list:
                if s in hparams.pretrain:
                    if val in w2v_model[s]:
                        word2index[s][idx][val]=len(word2index[s][idx])+2
                elif w_list[val]>= hparams.vocab_threshold:
                    word2index[s][idx][val]=len(word2index[s][idx])+2
    print("done!")
    return word2index
    
def norm(train_df,dev_df,test_df,features):   
    if features:
        df=pd.concat([train_df,dev_df,test_df])[features].fillna(-1000)
        scaler = preprocessing.QuantileTransformer(random_state=0)
        scaler.fit(df[features]) 
        train_df[features]=scaler.transform(train_df[features].fillna(-1000))
        test_df[features]=scaler.transform(test_df[features].fillna(-1000))
        dev_df[features]=scaler.transform(dev_df[features].fillna(-1000))
    
        
        
if __name__ == '__main__':
    hparams=create_hparams()
    print_hparams(hparams)
    if hparams.seq_features is None:
        hparams.seq_features=[]
    if hparams.num_features is None:
        hparams.num_features=[]
    if hparams.pretrain is None:
        hparams.pretrain=[]
    if hparams.multi_features None:
        hparams.multi_features=[]        
    if hparams.kfold is False:
        train_df=pd.read_csv(os.path.join(hparams.data_path,'train.csv'))
        dev_df=pd.read_csv(os.path.join(hparams.data_path,'dev.csv'))
        test_df=pd.read_csv(os.path.join(hparams.data_path,'test.csv'))
        train_df=train_df.append(dev_df)
        train_df, dev_df,_,_ = train_test_split(train_df,train_df,test_size=0.05, random_state=2018)
        #hparams.seq_features+=hparams.multi_features
        hparams.word2index=build_vocabulary(train_df,hparams,dev_df,test_df)
        features=hparams.num_features
        norm(train_df,dev_df,test_df,features)   
        train(train_df,dev_df,test_df,hparams)
    else:
        train_df=pd.read_csv(os.path.join(hparams.data_path,'train.csv'))
        dev_df=pd.read_csv(os.path.join(hparams.data_path,'dev.csv'))
        test_df=pd.read_csv(os.path.join(hparams.data_path,'test.csv'))
        train_df=train_df.append(dev_df)
        train_df, dev_df,_,_ = train_test_split(train_df,train_df,test_size=0.05, random_state=2018)
        features=hparams.num_features
        norm(train_df,dev_df,test_df,features) 
        index=list(range(len(train_df)))
        random.shuffle(index)
        for i in range(5):
            train(train_df,dev_df,test_df,hparams)
            temp=index[int((5-i)/5.0*len(index)):]+index[:int((5-i)/5.0*len(index))]
            hparams.word2index=build_vocabulary(train_df.iloc[temp],hparams)
            train(train_df.iloc[temp],dev_df,test_df,hparams,i)
        index=list(range(len(train_df)))
        random.shuffle(index)
        for i in range(5):
            train(train_df,dev_df,test_df,hparams)
            temp=index[int((5-i)/5.0*len(index)):]+index[:int((5-i)/5.0*len(index))]
            hparams.word2index=build_vocabulary(train_df.iloc[temp],hparams)
            train(train_df.iloc[temp],dev_df,test_df,hparams,i)

test_df[['user_id','res']].to_csv('answer.csv',index=False)
