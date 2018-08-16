import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import utils
import os
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
def create_hparams():
    return tf.contrib.training.HParams(
        #train_iterval=[[1,16],[8,16],[6,14],[4,12],[2,10]],
        train_iterval=[[1,16],[8,23]],
        test_iterval=[[15,30]],
        B_data_path="data",
        A_data_path="A_data",
        
        )

def pre_data(hparams,create=True,data_path=None):
    if create==False:        
        app_launch_df=pd.read_csv(os.path.join(data_path,'app_launch_log.csv'))
        user_activity_df=pd.read_csv(os.path.join(data_path,'user_activity_log.csv'))
        user_register_df=pd.read_csv(os.path.join(data_path,'user_register_log.csv'))
        video_create_df=pd.read_csv(os.path.join(data_path,'video_create_log.csv'))
        return app_launch_df,user_activity_df,user_register_df,video_create_df
    else:
        app_launch_df=pd.read_csv(os.path.join(data_path,'app_launch_log.txt'),sep='\t',header=None).sort_index(by=[0,1])
        app_launch_df.columns=['user_id','launch_day']
        app_launch_df.to_csv(os.path.join(data_path,'app_launch_log.csv'),index=False)
        
        user_activity_df=pd.read_csv(os.path.join(data_path,'user_activity_log.txt'),sep='\t',header=None).sort_index(by=[0,1])
        user_activity_df.columns=['user_id','activity_day','page','video_id','author_id','action_type']
        user_activity_df.to_csv(os.path.join(data_path,'user_activity_log.csv'),index=False)
        
        user_register_df=pd.read_csv(os.path.join(data_path,'user_register_log.txt'),sep='\t',header=None).sort_index(by=[0,1])
        user_register_df.columns=['user_id','register_day','register_type','device_type']
        user_register_df.to_csv(os.path.join(data_path,'user_register_log.csv'),index=False)
        
        video_create_df=pd.read_csv(os.path.join(data_path,'video_create_log.txt'),sep='\t',header=None).sort_index(by=[0,1])
        video_create_df.columns=['user_id','create_day']
        video_create_df.to_csv(os.path.join(data_path,'video_create_log.csv'),index=False)
        
        return app_launch_df,user_activity_df,user_register_df,video_create_df

def norm(train_df,dev_df,test_df,features):   
    if features:
        df=pd.concat([train_df,dev_df,test_df])
        scaler = preprocessing.QuantileTransformer(random_state=0)
        scaler.fit(df[features]) 
        train_df[features]=scaler.transform(train_df[features])
        test_df[features]=scaler.transform(test_df[features])
        dev_df[features]=scaler.transform(dev_df[features])

def label_encoder(train_df,dev_df,test_df,features):
    df=pd.concat([train_df,dev_df,test_df])
    for f in features:
        enc=LabelEncoder()
        enc.fit(df[f])
        train_df[f]=enc.transform(train_df[f])
        test_df[f]=enc.transform(test_df[f])
        dev_df[f]=enc.transform(dev_df[f])

def get_gbdt(train_df,dev_df,test_df,features):
    num=10
    clf = xgb.XGBClassifier(learning_rate=0.01, n_estimators=num, max_depth=6, min_child_weight=1, gamma=0.5,subsample=0.6,
                        colsample_bytree=0.6, objective='binary:logistic', scale_pos_weight=1, reg_alpha=1e-05,
                        reg_lambda=1, seed=0,n_jobs=30)  
    train_y = np.array(train_df['label'])
    train_x=train_df[features]
    dev_x=dev_df[features]
    test_x=test_df[features]  
    clf.fit(train_x, train_y)
    out_df = pd.DataFrame(clf.apply(train_x))
    out_df.columns = ['G' + str(i) for i in range(1, num + 1)]
    train_df = pd.concat([train_df, out_df], axis=1)
    out_df = pd.DataFrame(clf.apply(dev_x))
    out_df.columns = ['G' + str(i) for i in range(1, num + 1)]
    dev_df = pd.concat([dev_df, out_df], axis=1)  
    out_df = pd.DataFrame(clf.apply(test_x))
    out_df.columns = ['G' + str(i) for i in range(1, num + 1)]
    test_df = pd.concat([test_df, out_df], axis=1)    
    
    
    features=['G' + str(i) for i in range(1, num + 1)]
    print(features)
    return train_df,dev_df,test_df
        
def create_id(app_launch_df,user_activity_df,user_register_df,video_create_df,interval):
    temp_register=user_register_df[(user_register_df['register_day']>=interval[0])&(user_register_df['register_day']<=interval[1])]
    temp_activity=user_activity_df[(user_activity_df['activity_day']>=interval[0])&(user_activity_df['activity_day']<=interval[1])]
    temp_launch=app_launch_df[(app_launch_df['launch_day']>=interval[0])&(app_launch_df['launch_day']<=interval[1])]
    temp_create=video_create_df[(video_create_df['create_day']>=interval[0])&(video_create_df['create_day']<=interval[1])] 
    df=pd.concat([temp_register[['user_id']],temp_activity[['user_id']],temp_launch[['user_id']],temp_create[['user_id']]])
    df=df.drop_duplicates()
    return df

def create_label(df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval,train=True):
    if train:
        temp_register=user_register_df[(user_register_df['register_day']>=interval[1]+1)&(user_register_df['register_day']<=interval[1]+7)]
        temp_activity=user_activity_df[(user_activity_df['activity_day']>=interval[1]+1)&(user_activity_df['activity_day']<=interval[1]+7)]
        temp_launch=app_launch_df[(app_launch_df['launch_day']>=interval[1]+1)&(app_launch_df['launch_day']<=interval[1]+7)]
        temp_create=video_create_df[(video_create_df['create_day']>=interval[1]+1)&(video_create_df['create_day']<=interval[1]+7)]
        temp=pd.concat([temp_register[['user_id']],temp_activity[['user_id']],temp_launch[['user_id']],temp_create[['user_id']]])
        before_register=user_register_df[(user_register_df['register_day']>=1)&(user_register_df['register_day']<=interval[1])]
        idx=set(temp['user_id'])
        """
        idx=set(temp['user_id'])&set(before_register['user_id'])
        print(len(idx),len(set(df['user_id'])))
        print(len(set(df['user_id'])&idx)*1.0/len(idx))
        exit()
        """
        label=[]
        for val in df['user_id'].values:
            if val in idx:
                label.append(1)
            else:
                label.append(0)
        df['label']=label
    else:
        df['label']=[-1]*len(df['user_id'])
    return df

def create_single_features(df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval):
    features=[]
       
    temp_register=user_register_df[user_register_df['register_day']<=interval[1]]
    temp_activity=user_activity_df[(user_activity_df['activity_day']>=interval[0])&(user_activity_df['activity_day']<=interval[1])]
    temp_launch=app_launch_df[(app_launch_df['launch_day']>=interval[0])&(app_launch_df['launch_day']<=interval[1])]
    temp_create=video_create_df[(video_create_df['create_day']>=interval[0])&(video_create_df['create_day']<=interval[1])]
    
    #注册时间与预测时间的差值，若区间内无注册，则为-1
    df=pd.merge(df,temp_register,on='user_id',how='left')
    df['register_day_diff']=interval[1]+1-df['register_day']
    df['register_day_diff']=df['register_day_diff'].apply(lambda x: np.nan if x >interval[1]+1-interval[0] else x)
    df=df.fillna(-1)
    for key in df:
        df[key]=df[key].astype(int)
    features.extend(['register_day_diff','register_type','device_type']) 
    
    #启动次数
    groupby_size=temp_launch.groupby('user_id').size()
    df['launch_cont']=df['user_id'].apply(lambda x:groupby_size[x] if x in groupby_size else -1)
    features.append('launch_cont')
    
    #最近启动时间差
    groupby_max=temp_launch.groupby('user_id').max()['launch_day']
    df['launch_day_diff']=df['user_id'].apply(lambda x:interval[1]+1-groupby_max[x] if x in groupby_max else -1)
    features.append('launch_day_diff')
    
    #video创建次数
    groupby_size=temp_create.groupby('user_id').size()
    df['create_cont']=df['user_id'].apply(lambda x:groupby_size[x] if x in groupby_size else 0)
    df['create_cont']=df['create_cont'].apply(lambda x: int(np.log(x+1)/np.log(2)))
    features.append('create_cont')
    
    #video天数
    temp_df=temp_create[['user_id','create_day']].drop_duplicates()
    groupby_size=temp_df.groupby('user_id').size()
    df['create_day_cont']=df['user_id'].apply(lambda x:groupby_size[x] if x in groupby_size else 0)
    features.append('create_day_cont')
    
    #最近启动时间差
    groupby_max=temp_create.groupby('user_id').max()['create_day']
    df['create_day_diff']=df['user_id'].apply(lambda x:interval[1]+1-groupby_max[x] if x in groupby_max else -1)
    features.append('create_day_diff')

    #活动次数
    groupby_size=temp_activity.groupby('user_id').size()
    df['activity_cont']=df['user_id'].apply(lambda x:groupby_size[x] if x in groupby_size else 0)
    features.append('activity_cont')
    
    #最近活动时间差
    groupby_max=temp_activity.groupby('user_id').max()['activity_day']
    df['activity_day_diff']=df['user_id'].apply(lambda x:interval[1]+1-groupby_max[x] if x in groupby_max else -1)
    features.append('activity_day_diff')
    
    #活跃天数
    temp_df=temp_activity[['user_id','activity_day']].drop_duplicates()
    groupby_size=temp_df.groupby('user_id').size()
    df['activity_day_cont']=df['user_id'].apply(lambda x:groupby_size[x] if x in groupby_size else 0)
    df['activity_cont']=df['activity_cont'].apply(lambda x: int(np.log(x+1)/np.log(2)))
    features.append('activity_day_cont')    
      
    #用户activity记录，01串
    dic={}
    groupby_size=dict(temp_activity.groupby(['user_id','activity_day']).size())
    seq_activity=[]
    for val in df['user_id']:
        temp=[]
        for day in range(interval[0],interval[1]+1):
            if (val,day) in groupby_size:
                temp.append('1')
            else:
                temp.append('0')
        seq_activity.append(''.join(temp))
    df['activity_01']=seq_activity
    features.append('activity_01')    
    
    #用户video记录，01串
    dic={}
    groupby_size=dict(temp_create.groupby(['user_id','create_day']).size())
    seq_create=[]
    for val in df['user_id']:
        temp=[]
        for day in range(interval[0],interval[1]+1):
            if (val,day) in groupby_size:
                temp.append('1')
            else:
                temp.append('0')
        seq_create.append(''.join(temp))
    df['create_01']=seq_create
    features.append('create_01')  
    
    #用户登陆记录，01串
    dic={}
    groupby_size=dict(temp_launch.groupby(['user_id','launch_day']).size())
    seq_launch=[]
    for val in df['user_id']:
        temp=[]
        for day in range(interval[1]-6,interval[1]+1):
            if (val,day) in groupby_size:
                temp.append('1')
            else:
                temp.append('0')
        seq_launch.append(''.join(temp))
    df['launch_01']=seq_launch
    features.append('launch_01')
    
    return df,features
    
def create_seq_features(df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval): 
    features=[]    
    temp_register=user_register_df[(user_register_df['register_day']>=interval[0])&(user_register_df['register_day']<=interval[1])]
    temp_activity=user_activity_df[(user_activity_df['activity_day']>=interval[0])&(user_activity_df['activity_day']<=interval[1])]
    temp_launch=app_launch_df[(app_launch_df['launch_day']>=interval[0])&(app_launch_df['launch_day']<=interval[1])]
    temp_video=video_create_df[(video_create_df['create_day']>=interval[0])&(video_create_df['create_day']<=interval[1])] 

    
    #用户activity记录，时间差+当天创建总数
    dic={}
    groupby_size=dict(temp_activity.groupby(['user_id','activity_day']).size())
    seq_activity=[]
    for val in df['user_id']:
        temp=[]
        for day in range(interval[1]-6,interval[1]+1):
            if (val,day) in groupby_size:
                temp.append(str(interval[1]+1-day)+'_'+str(int(np.log(groupby_size[(val,day)]+1)/np.log(2))))
        if len(temp)==0:
            temp.append('UNK_UNK')
        seq_activity.append(' '.join(temp))
    df['seq_activity']=seq_activity
    features.append('seq_activity')



    #用户登陆记录, 时间差+时间间隔
        
    dic={}
    groupby_size=dict(temp_launch.groupby(['user_id','launch_day']).size())
    seq_launch=[]
    for val in df['user_id']:
        temp=[]
        for day in range(interval[1]-6,interval[1]+1):
            if (val,day) in groupby_size:
                temp.append(str(interval[1]+1-day))
        if len(temp)==0:
            temp.append('UNK')
        seq_launch.append(' '.join(temp))
    df['seq_launch']=seq_launch
    features.append('seq_launch')
    

    return df,features        
        
def create_num_features(df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval):   
    features=[]    
    temp_register=user_register_df[(user_register_df['register_day']>=interval[0])&(user_register_df['register_day']<=interval[1])]
    temp_activity=user_activity_df[(user_activity_df['activity_day']>=interval[0])&(user_activity_df['activity_day']<=interval[1])]
    temp_launch=app_launch_df[(app_launch_df['launch_day']>=interval[0])&(app_launch_df['launch_day']<=interval[1])]
    temp_video=video_create_df[(video_create_df['create_day']>=interval[0])&(video_create_df['create_day']<=interval[1])] 
    temp_activity['activity_day_diff']=interval[1]-temp_activity['activity_day']
    activity_groupby=temp_activity.groupby(['user_id'])
    var_groupby=activity_groupby.apply(lambda x: abs(np.var(np.fft.fft(x['activity_day_diff'])))) 
    mean_groupby=activity_groupby.apply(lambda x: abs(np.mean(np.fft.fft(x['activity_day_diff'])))) 
    median_groupby=activity_groupby.apply(lambda x: abs(np.median(np.fft.fft(x['activity_day_diff'])))) 
    max_groupby=activity_groupby.apply(lambda x: abs(np.max(np.fft.fft(x['activity_day_diff'])))) 
    min_groupby=activity_groupby.apply(lambda x: abs(np.min(np.fft.fft(x['activity_day_diff'])))) 
    df['activity_day_diff_var']=df['user_id'].apply(lambda x:var_groupby[x] if x in var_groupby else np.nan)
    df['activity_day_diff_mean']=df['user_id'].apply(lambda x:mean_groupby[x] if x in mean_groupby else np.nan)
    df['activity_day_diff_median']=df['user_id'].apply(lambda x:median_groupby[x] if x in median_groupby else np.nan)
    df['activity_day_diff_max']=df['user_id'].apply(lambda x:max_groupby[x] if x in max_groupby else np.nan)
    df['activity_day_diff_min']=df['user_id'].apply(lambda x:min_groupby[x] if x in min_groupby else np.nan)
    features.extend(['activity_day_diff_var','activity_day_diff_mean','activity_day_diff_median','activity_day_diff_max','activity_day_diff_min']) 
    return df,features
    
    
    
if __name__ == '__main__':
    hparams=create_hparams()
    utils.print_hparams(hparams)
    app_launch_df,user_activity_df,user_register_df,video_create_df=pre_data(hparams,create=False,data_path=hparams.B_data_path)
        
    #构造训练集，区间是hparams.train_iterval，以后7天做label
    train_dfs=[]
    for interval in hparams.train_iterval:
        train_df=create_id(app_launch_df,user_activity_df,user_register_df,video_create_df,interval)
        train_df=create_label(train_df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval,train=True)
        train_df,num_features=\
        create_num_features(train_df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval)
        train_df,single_features=\
        create_single_features(train_df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval)
        train_df,seq_features=\
        create_seq_features(train_df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval)
        train_dfs.append(train_df)

    train_df=pd.concat(train_dfs)
    
    #构造测试集，区间是hparams.test_iterval
    test_dfs=[]
    for interval in hparams.test_iterval:
        test_df=create_id(app_launch_df,user_activity_df,user_register_df,video_create_df,interval)
        test_df=create_label(test_df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval,train=False)
        test_df,num_features=\
        create_num_features(test_df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval)
        test_df,single_features=\
        create_single_features(test_df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval)
        test_df,seq_features=\
        create_seq_features(test_df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval)
        test_dfs.append(test_df)
    test_df=pd.concat(test_dfs)   
    
    train_df, dev_df,_,_ = train_test_split(train_df,train_df,test_size=0.1, random_state=2018)
    if hparams.A_data_path:
        app_launch_df,user_activity_df,user_register_df,video_create_df=pre_data(hparams,create=True,data_path=hparams.A_data_path)        
        train_dfs=[]
        train_dfs.append(train_df)
        for interval in hparams.train_iterval:
            train_df=create_id(app_launch_df,user_activity_df,user_register_df,video_create_df,interval)
            train_df=create_label(train_df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval,train=True)
            train_df,num_features=\
            create_num_features(train_df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval)
            train_df,single_features=\
            create_single_features(train_df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval)
            train_df,seq_features=\
            create_seq_features(train_df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval)
            train_dfs.append(train_df)
        train_df=pd.concat(train_dfs)
        
    encoder_features=['activity_01','launch_01','create_01']
    label_encoder(train_df,dev_df,test_df,encoder_features)

    print('Num Features:',num_features)
    print('Single Features:',single_features)
    print('Sequence Features:',seq_features)
    train_df.to_csv(os.path.join(hparams.B_data_path,'train.csv'),index=False)
    dev_df.to_csv(os.path.join(hparams.B_data_path,'dev.csv'),index=False)
    test_df.to_csv(os.path.join(hparams.B_data_path,'test.csv'),index=False)
    
        
        
        
        
        
        
    
    
    
    
    
