import tensorflow as tf
import pandas as pd
import gc
import os
import numpy as np
def create_hparams():
    return tf.contrib.training.HParams(
        train_iterval=[1,16],
        dev_iterval=[8,23],
        test_iterval=[15,30],
        data_path="/mnt/datasets/fusai/",
        )
hparams=create_hparams()


def pre_data(hparams,data_path=None):
    app_launch_df=pd.read_csv(os.path.join(data_path,'app_launch_log.txt'),sep='\t',header=None,dtype={0:np.int32,1:np.int8}).sort_index(by=[0,1])
    app_launch_df.columns=['user_id','launch_day']
    
    user_activity_df=pd.read_csv(os.path.join(data_path,'user_activity_log.txt'),sep='\t',header=None,dtype={0:np.int32,1:np.int8,2:np.int8,3:np.int32,4:np.int32,5:np.int8}).sort_index(by=[0,1])
    user_activity_df.columns=['user_id','activity_day','page','video_id','author_id','action_type']
    
    user_register_df=pd.read_csv(os.path.join(data_path,'user_register_log.txt'),dtype={0:np.int32,1:np.int8},sep='\t',header=None).sort_index(by=[0,1])
    user_register_df.columns=['user_id','register_day','register_type','device_type']
    
    video_create_df=pd.read_csv(os.path.join(data_path,'video_create_log.txt'),sep='\t',header=None,dtype={0:np.int32,1:np.int8}).sort_index(by=[0,1])
    video_create_df.columns=['user_id','create_day']
    
    return app_launch_df,user_activity_df,user_register_df,video_create_df
        

def create_id(app_launch_df,user_activity_df,user_register_df,video_create_df,interval):
    temp_register=user_register_df[(user_register_df['register_day']<=interval[1])]
    temp_activity=user_activity_df[(user_activity_df['activity_day']>=interval[0])&(user_activity_df['activity_day']<=interval[1])]
    temp_launch=app_launch_df[(app_launch_df['launch_day']>=interval[0])&(app_launch_df['launch_day']<=interval[1])]
    temp_create=video_create_df[(video_create_df['create_day']>=interval[0])&(video_create_df['create_day']<=interval[1])] 
    df=pd.concat([temp_register[['user_id']],temp_activity[['user_id']],temp_launch[['user_id']],temp_create[['user_id']]])
    df=df.drop_duplicates()
    del temp_register
    del temp_activity
    del temp_launch
    del temp_create
    gc.collect()
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
        label=[]
        for val in df['user_id'].values:
            if val in idx:
                label.append(1)
            else:
                label.append(0)
        df['label']=label
        del temp_register
        del temp_activity
        del temp_launch
        del temp_create
        gc.collect()     
    else:
        df['label']=[-1]*len(df['user_id'])
   
    return df


app_launch_df,user_activity_df,user_register_df,video_create_df=pre_data(hparams,data_path=hparams.data_path)

#构造测试集，区间是hparams.test_iterval
print("testing",hparams.test_iterval)
test_df=create_id(app_launch_df,user_activity_df,user_register_df,video_create_df,hparams.test_iterval)
print("test creating id done!")
test_df=create_label(test_df,app_launch_df,user_activity_df,user_register_df,video_create_df,hparams.test_iterval,train=False)
print("test creating label done!")


#构造验证集，区间是hparams.dev_iterval，以后7天做label
print("dev",hparams.dev_iterval)
dev_df=create_id(app_launch_df,user_activity_df,user_register_df,video_create_df,hparams.dev_iterval)
print("dev creating id done!")
dev_df=create_label(dev_df,app_launch_df,user_activity_df,user_register_df,video_create_df,hparams.dev_iterval,train=True)
print("dev creating label done!")


#构造训练集，区间是hparams.train_iterval，以后7天做label
print("training",hparams.train_iterval)
train_df=create_id(app_launch_df,user_activity_df,user_register_df,video_create_df,hparams.train_iterval)
print("train creating id done!")
train_df=create_label(train_df,app_launch_df,user_activity_df,user_register_df,video_create_df,hparams.train_iterval,train=True)
print("train creating label done!")

print("train shape",train_df.shape)
print("dev shape",dev_df.shape)
print("test shape",test_df.shape)

###################################
#############提取单值特征##########
###################################
def create_single_features(df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval):
    features=[]
    temp_register=user_register_df[user_register_df['register_day']<=interval[1]]
    temp_activity=user_activity_df[(user_activity_df['activity_day']>=interval[0])&(user_activity_df['activity_day']<=interval[1])]
    temp_launch=app_launch_df[(app_launch_df['launch_day']>=interval[0])&(app_launch_df['launch_day']<=interval[1])]
    temp_create=video_create_df[(video_create_df['create_day']>=interval[0])&(video_create_df['create_day']<=interval[1])]
    
    #注册时间与预测时间的差值
    dic={}
    for item in temp_register[['user_id','register_day','register_type','device_type']].values:
        dic[item[0]]=(item[1],item[2],item[3])

    df['register_day']=df['user_id'].apply(lambda x:dic[x][0])
    df['register_day_diff']=interval[1]+1-df['register_day']
    df['register_type']=df['user_id'].apply(lambda x:dic[x][1])
    df['device_type']=df['user_id'].apply(lambda x:dic[x][2])
    df['register_day_diff']=df['register_day_diff'].apply(lambda x: min(x,interval[1]+1-interval[0]))
    del dic
    gc.collect()
    features.extend(['register_day_diff','register_type','device_type']) 

    
    #启动次数
    groupby_size=temp_launch.groupby('user_id').size()
    df['launch_cont']=df['user_id'].apply(lambda x:groupby_size[x] if x in groupby_size else 0)
    features.append('launch_cont')
    del groupby_size
    gc.collect()  
    
    #最近启动时间差
    groupby_max=temp_launch.groupby('user_id').max()['launch_day']
    df['launch_day_diff']=df['user_id'].apply(lambda x:interval[1]+1-groupby_max[x] if x in groupby_max else interval[1]+1-interval[0])
    features.append('launch_day_diff')
    del groupby_max
    gc.collect() 
    
    #video创建次数
    groupby_size=temp_create.groupby('user_id').size()
    df['create_cont']=df['user_id'].apply(lambda x:groupby_size[x] if x in groupby_size else 0)
    features.append('create_cont')
    del groupby_size
    gc.collect()  
    
    #video天数
    temp_df=temp_create[['user_id','create_day']].drop_duplicates()
    groupby_size=temp_df.groupby('user_id').size()
    df['create_day_cont']=df['user_id'].apply(lambda x:groupby_size[x] if x in groupby_size else 0)
    features.append('create_day_cont')
    del temp_df
    del groupby_size
    gc.collect() 
    
    #最近启动时间差
    groupby_max=temp_create.groupby('user_id').max()['create_day']
    df['create_day_diff']=df['user_id'].apply(lambda x:interval[1]+1-groupby_max[x] if x in groupby_max else interval[1]+1-interval[0])
    features.append('create_day_diff')
    del groupby_max
    gc.collect() 
    
    #活动次数
    groupby_size=temp_activity.groupby('user_id').size()
    df['activity_cont']=df['user_id'].apply(lambda x:groupby_size[x] if x in groupby_size else 0)
    features.append('activity_cont')
    del groupby_size
    gc.collect()  
    
    #最近活动时间差
    groupby_max=temp_activity.groupby('user_id').max()['activity_day']
    df['activity_day_diff']=df['user_id'].apply(lambda x:interval[1]+1-groupby_max[x] if x in groupby_max else interval[1]+1-interval[0])
    features.append('activity_day_diff')
    del groupby_max
    gc.collect()  
    
    #活跃天数
    temp_df=temp_activity[['user_id','activity_day']].drop_duplicates()
    groupby_size=temp_df.groupby('user_id').size()
    df['activity_day_cont']=df['user_id'].apply(lambda x:groupby_size[x] if x in groupby_size else 0)
    features.append('activity_day_cont')    
    del temp_df
    del groupby_size
    gc.collect() 
    
    del temp_register
    del temp_activity
    del temp_launch
    del temp_create
    gc.collect()    
    
    
    return df,features

#测试集提取单值特征
print("testing",hparams.test_iterval)
test_df,_=create_single_features(test_df,app_launch_df,user_activity_df,user_register_df,video_create_df,hparams.test_iterval)
print("test creating single features done!")



#验证集提取单值特征
print("dev",hparams.dev_iterval)
dev_df,_=create_single_features(dev_df,app_launch_df,user_activity_df,user_register_df,video_create_df,hparams.dev_iterval)
print("dev creating single features done!")



#训练集提取单值特征
print("training",hparams.train_iterval)
train_df,single_features=create_single_features(train_df,app_launch_df,user_activity_df,user_register_df,video_create_df,hparams.train_iterval)
print("train creating single features done!")


print("train shape",train_df.shape)
print("dev shape",dev_df.shape)
print("test shape",test_df.shape)
print("single features v1",single_features)


###################################
#############提取序列特征##########
###################################
from collections import Counter
def create_seq_features(df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval): 
    features=[]    
    temp_register=user_register_df[(user_register_df['register_day']>=interval[0])&(user_register_df['register_day']<=interval[1])]
    temp_activity=user_activity_df[(user_activity_df['activity_day']>=interval[0])&(user_activity_df['activity_day']<=interval[1])]
    temp_launch=app_launch_df[(app_launch_df['launch_day']>=interval[0])&(app_launch_df['launch_day']<=interval[1])]
    temp_create=video_create_df[(video_create_df['create_day']>=interval[0])&(video_create_df['create_day']<=interval[1])] 
    
    activity_groupby=temp_activity[['user_id','activity_day']].groupby('user_id')
    activity_groupby=activity_groupby.apply(lambda x: list(x['activity_day']))
    create_groupby=temp_create[['user_id','create_day']].groupby('user_id')
    create_groupby=create_groupby.apply(lambda x: list(x['create_day']))
    launch_groupby=temp_launch[['user_id','launch_day']].groupby('user_id')
    launch_groupby=launch_groupby.apply(lambda x: list(x['launch_day']))
    
    def f(x):
        days=set()
        if x in launch_groupby:
            days=days|set(launch_groupby[x])
        if x in create_groupby:
            days=days|set(create_groupby[x])
        if x in activity_groupby:
            days=days|set(activity_groupby[x])
        temp=[]
        for i in range(interval[1]-6,interval[1]+1):
            if i in days:
                temp.append(str(interval[1]+1-i)+'_'+'1')
            else:
                temp.append(str(interval[1]+1-i)+'_'+'0')
                
        return ' '.join(temp)  
    df['active_seq']=df['user_id'].apply(f) 
    features.append('active_seq')
    
    def f(x):
        days=[]
        if x in create_groupby:
            days=create_groupby[x]
        days=dict(Counter(days))
        temp=[]
        for i in range(interval[1]-6,interval[1]+1):
            if i in days:
                temp.append(str(interval[1]+1-i)+'_'+str(int(np.log(days[i]+1)/np.log(2))))
            else:
                temp.append(str(interval[1]+1-i)+'_'+'0')
                
        return ' '.join(temp)  
        
    df['create_seq']=df['user_id'].apply(f) 
    features.append('create_seq')
    def f(x):
        days=[]
        if x in activity_groupby:
            days=activity_groupby[x]
        days=dict(Counter(days))
        temp=[]
        for i in range(interval[1]-6,interval[1]+1):
            if i in days:
                temp.append(str(interval[1]+1-i)+'_'+str(days[i]))
            else:
                temp.append(str(interval[1]+1-i)+'_'+'0')
                
        return ' '.join(temp)  
    df['activity_seq']=df['user_id'].apply(f) 
    features.append('activity_seq')
    
    del temp_register
    del temp_activity
    del temp_launch
    del temp_create
    gc.collect()    
    
    return df,features   



#测试集提取序列特征
print("testing",hparams.test_iterval)
test_df,_=create_seq_features(test_df,app_launch_df,user_activity_df,user_register_df,video_create_df,hparams.test_iterval)
print("test creating seq features done!")



#验证集提取序列特征
print("dev",hparams.dev_iterval)
dev_df,_=create_seq_features(dev_df,app_launch_df,user_activity_df,user_register_df,video_create_df,hparams.dev_iterval)
print("dev creating seq features done!")



#训练集提取序列特征
print("training",hparams.train_iterval)
train_df,seq_features=create_seq_features(train_df,app_launch_df,user_activity_df,user_register_df,video_create_df,hparams.train_iterval)
print("train creating seq features done!")


print("train shape",train_df.shape)
print("dev shape",dev_df.shape)
print("test shape",test_df.shape)
print("seq features v1",seq_features)




###################################
###########提取浮点数特征##########
###################################
def create_num_features(df,app_launch_df,user_activity_df,user_register_df,video_create_df,interval):   
    features=[]    
    temp_register=user_register_df[(user_register_df['register_day']>=interval[0])&(user_register_df['register_day']<=interval[1])]
    temp_activity=user_activity_df[(user_activity_df['activity_day']>=interval[0])&(user_activity_df['activity_day']<=interval[1])]
    temp_launch=app_launch_df[(app_launch_df['launch_day']>=interval[0])&(app_launch_df['launch_day']<=interval[1])]
    temp_create=video_create_df[(video_create_df['create_day']>=interval[0])&(video_create_df['create_day']<=interval[1])] 
    
    activity_groupby=temp_activity[['user_id','activity_day']].drop_duplicates().groupby('user_id')
    activity_groupby=activity_groupby.apply(lambda x: set(x['activity_day']))
    create_groupby=temp_create[['user_id','create_day']].drop_duplicates().groupby('user_id')
    create_groupby=create_groupby.apply(lambda x: set(x['create_day']))
    launch_groupby=temp_launch[['user_id','launch_day']].drop_duplicates().groupby('user_id')
    launch_groupby=launch_groupby.apply(lambda x: set(x['launch_day']))
    

    df['active_day_fft_min']=df['active_days'].apply(lambda x:abs(np.min(x)))
    features.append('active_day_fft_min') 
    df['active_day_fft_max']=df['active_days'].apply(lambda x:abs(np.max(x)))
    features.append('active_day_fft_max')
    df['active_day_fft_mean']=df['active_days'].apply(lambda x:abs(np.mean(x)))
    features.append('active_day_fft_mean')
    df['active_day_fft_var']=df['active_days'].apply(lambda x:abs(np.var(x)))
    features.append('active_day_fft_var')
    df['active_day_fft_median']=df['active_days'].apply(lambda x:abs(np.median(x)))
    features.append('active_day_fft_median')

    del temp_register
    del temp_activity
    del temp_launch
    del temp_create
    gc.collect()     
    return df,features


#测试集提取浮点数特征
print("testing",hparams.test_iterval)
test_df,_=create_num_features(test_df,app_launch_df,user_activity_df,user_register_df,video_create_df,hparams.test_iterval)
print("test creating num features done!")



#验证集提取浮点数特征
print("dev",hparams.dev_iterval)
dev_df,_=create_num_features(dev_df,app_launch_df,user_activity_df,user_register_df,video_create_df,hparams.dev_iterval)
print("dev creating num features done!")



#训练集提取浮点数特征
print("training",hparams.train_iterval)
train_df,num_features=create_num_features(train_df,app_launch_df,user_activity_df,user_register_df,video_create_df,hparams.train_iterval)
print("train creating num features done!")


print("train shape",train_df.shape)
print("dev shape",dev_df.shape)
print("test shape",test_df.shape)
print("num features v1",num_features)
train_df.to_csv('/home/kesci/train.csv',index=False)
dev_df.to_csv('/home/kesci/dev.csv',index=False)
test_df.to_csv('/home/kesci/test.csv',index=False)