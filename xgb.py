import pandas as pd
import xgboost as xgb
import numpy as np
import sklearn.metrics
import lightgbm as lgb
import random
def evalf1(preds,dtrain):
    labels=dtrain.get_label()
    if len(dev_df)!=len(preds):
        return 'logloss',sklearn.metrics.log_loss(labels,preds)
    else:
        best_f1=0
        best_thresold=0
        dev_df['res']=preds
        for i in range(100):
            threshold=0.01*i
            dev_df['pred']=dev_df['res'].apply(lambda x: 1 if x>threshold else 0)
            f1=sklearn.metrics.f1_score(dev_df['label'], dev_df['pred'])
            if f1>best_f1:
                best_f1=f1
                best_thresold=threshold
        return 'f1',best_f1
    
def runLGB(train_x,train_y,test_x,test_y):
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=16, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=370, objective='binary',
        subsample=0.9, colsample_bytree=0.7, subsample_freq=1,feature_fraction=0.9,
        learning_rate=0.035, min_child_weight=1, random_state=2018, n_jobs=20
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y),(test_x,test_y)], eval_metric='logloss',early_stopping_rounds=1000)
    return clf

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=433):
    param = {}
    param['objective'] = 'binary:logistic'
    param['booster']='gbtree'
    param['eta'] = 0.03
    param['gamma'] = 1.0
    param['colsample_bylevel']= 0.7  
    param['lambda']=5  
    param['max_depth'] =4
    param['silent'] = 1
    param['min_child_weight'] = 1
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['seed'] = seed_val
    param['eval_metric']='logloss'
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds,watchlist, feval=evalf1,early_stopping_rounds=1000,maximize=True)
    else:
        xgtest = xgb.DMatrix(test_X)
        watchlist = [ (xgtrain,'train') ]
        model = xgb.train(plst, xgtrain, num_rounds,watchlist)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

features_to_use=['register_day_diff', 'register_type', 'device_type', 'launch_cont', 'launch_day_diff', 
                 'create_cont', 'create_day_diff','activity_cont','activity_day_diff','activity_day_cont',
                'activity_day_diff_var','activity_day_diff_mean','activity_day_diff_median','activity_day_diff_max',
                'activity_day_diff_min','registet_launch_day_diff']

train_df=pd.read_csv('data/train.csv')
dev_df=pd.read_csv('data/dev.csv')
test_df=pd.read_csv('data/test.csv')
dev_df['index']=list(range(len(dev_df)))


clf=runLGB(train_df[features_to_use],train_df['label'],dev_df[features_to_use],dev_df['label'])
dev_df['res']=clf.predict_proba(dev_df[features_to_use])[:,1]
test_df['res']=clf.predict_proba(test_df[features_to_use])[:,1]
best_f1=0
best_thresold=0
for i in range(100):
    threshold=0.01*i
    dev_df['pred']=dev_df['res'].apply(lambda x: 1 if x>threshold else 0)
    f1=sklearn.metrics.f1_score(dev_df['label'], dev_df['pred'])
    if f1>best_f1:
        best_f1=f1
        best_thresold=threshold
print(best_f1)

dev_df['prob']=dev_df['res']              
dev_df['res']=dev_df['res'].apply(lambda x: 1 if x>best_thresold else 0)
res=dev_df[dev_df['res']==1]
res[['index','prob']].to_csv('output/lgb_dev.txt', index=False, header=False) 
test_df['prob']=test_df['res']
test_df['res']=test_df['res'].apply(lambda x: 1 if x>best_thresold else 0)
res=test_df[test_df['res']==1]
res[['user_id','prob']].to_csv('output/lgb_test.txt', index=False, header=False)



preds,model=runXGB(train_df[features_to_use],train_df['label'],dev_df[features_to_use],dev_df['label'])

dev_df['res']=preds
best_f1=0
best_thresold=0
for i in range(100):
    threshold=0.01*i
    dev_df['pred']=dev_df['res'].apply(lambda x: 1 if x>threshold else 0)
    f1=sklearn.metrics.f1_score(dev_df['label'], dev_df['pred'])
    if f1>best_f1:
        best_f1=f1
        best_thresold=threshold

dev_df['prob']=dev_df['res']              
dev_df['res']=dev_df['res'].apply(lambda x: 1 if x>best_thresold else 0)
f1=sklearn.metrics.f1_score(dev_df['label'], dev_df['res'])
res=dev_df[dev_df['res']==1]
res[['index','prob']].to_csv('output/xgb_dev.txt', index=False, header=False) 
res=dev_df[dev_df['label']==1]
res[['index']].to_csv('output/dev_label.txt', index=False, header=False) 

print('Dev inference done!')
print('Dev thresold',round(best_thresold,4))
print("Dev f1:",round(f1,5))
print("Test inference ...")  
xgtest = xgb.DMatrix(test_df[features_to_use])
test_df['res']=model.predict(xgtest)
print('Test inference done!')
test_df['prob']=test_df['res']
test_df['res']=test_df['res'].apply(lambda x: 1 if x>best_thresold else 0)
res=test_df[test_df['res']==1]
print(len(res))
res[['user_id','prob']].to_csv('output/xgb_test.txt', index=False, header=False)




