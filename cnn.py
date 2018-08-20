import tensorflow as tf
import random
import pandas as pd
import numpy as np
import pickle as pkl
import gzip
import numpy as np
import random
import math
import json
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import sklearn.metrics
import time
import os
from sklearn import metrics
tf.set_random_seed(2018)
random.seed(2018)
np.random.seed(2018)
def print_step_info(prefix, global_step, info):
    print_out(
      "%sstep %d lr %g logloss %.6f epoch %d gN %.2f, %s" %
      (prefix, global_step, info["learning_rate"],
       info["train_ppl"],info["epoch"], info["avg_grad_norm"], time.ctime()))   
class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self,df,hparams,batch_size=None,shuffle=False):
        if batch_size:
            self.batch_size=batch_size
        else:
            self.batch_size=hparams.batch_size
        self.shuffle=shuffle
        if shuffle:
            df=df.sample(frac=1)
        self.df=df
        self.data={}
        self.hparams=hparams
        for s in self.hparams.single_features+['label']:
            self.data[s]=df[s].values
        for s in self.hparams.seq_features:
            self.data[s]=df[s].values     
        for s in self.hparams.num_features:
            self.data[s]=df[s].values  
        self.idx=0
        
    def reset(self):          
        self.idx=0

    def next(self):
        if self.idx>=len(self.data['label']):
            self.reset()
            raise StopIteration
            
        data={}
        for s in self.hparams.single_features+['label']:
            temp=[]
            idx=self.idx
            while idx<len(self.data[s]) and len(temp)!=self.batch_size:
                if s == 'label':
                    temp.append(self.data[s][idx])
                elif self.data[s][idx] in self.hparams.word2index[s]:
                    temp.append(self.hparams.word2index[s][self.data[s][idx]])
                else:
                    temp.append(0)
                idx+=1
            data[s]=temp
        for s in self.hparams.seq_features:
            temp=[]
            temp_len=[]
            idx=self.idx
            while idx<len(self.data[s]) and len(temp)!=self.batch_size:
                vals=self.data[s][idx].split()
                if len(vals)>self.hparams.max_len:
                    vals=vals[-self.hparams.max_len:]
                vals=[v.split('_') for v in vals]
                vals=[[self.hparams.word2index[s][num][v] if v in self.hparams.word2index[s][num] else 0 for num,v in enumerate(val)] for val in vals]
                temp.append(vals)
                temp_len.append(len(vals))
                idx+=1
            max_len=max(temp_len)+4
            temp=[t+[[0]*len(self.hparams.word2index[s])]*(max_len-len(t))  for t in temp]
            data[s]=temp
            data[s+'_len']=temp_len
        
        
        for s in self.hparams.num_features:
            temp=[]
            idx=self.idx
            while idx<len(self.data[s]) and len(temp)!=self.batch_size:
                temp.append(self.data[s][idx])
                idx+=1
            data[s]=temp
        self.idx=idx 
        return data


    
class Model(BaseModel):
    def __init__(self,hparams):
        tf.set_random_seed(2018)
        random.seed(2018)
        np.random.seed(2018)
        self.hparams=hparams
        self.initializer = self._get_initializer(hparams)
        self.cross_params=[]
        self.layer_params=[]
        self.single_ids={}
        self.num_feaures={}
        self.seq_ids={}
        self.seq_len={}
        self.emb_v2={}
        self.mulit_mask={}
        self.label = tf.placeholder(shape=(None), dtype=tf.float32)
        self.use_dropout=tf.placeholder(tf.bool)
        for s in hparams.single_features:
            self.single_ids[s]=tf.placeholder(shape=(None,), dtype=tf.int32)
            self.emb_v2[s]= tf.Variable(tf.truncated_normal(shape=[len(hparams.word2index[s])+2,hparams.k], mean=0.0, stddev=0.0001),name='emb_v2_'+s)
            
        for s in self.hparams.num_features:
            self.num_feaures[s]=tf.placeholder(shape=(None,), dtype=tf.float32)
            self.emb_v2[s]= tf.Variable(tf.truncated_normal(shape=[hparams.batch_num,hparams.k], mean=0.0, stddev=0.0001),name='emb_v2_'+s)

        for s in hparams.seq_features:
            self.seq_ids[s]=tf.placeholder(shape=(None,None,len(hparams.word2index[s])), dtype=tf.int32)
            self.seq_len[s]=tf.placeholder(shape=(None,), dtype=tf.int32)
            self.mulit_mask[s] = tf.sequence_mask(self.seq_len[s],tf.shape(self.seq_ids[s])[1],dtype=tf.float32)
            self.emb_v2[s]={}
            for idx in range(len(hparams.word2index[s])):
                self.emb_v2[s][idx]= tf.Variable(tf.truncated_normal(shape=[len(hparams.word2index[s][idx])+2,hparams.k], mean=0.0, stddev=0.0001),name='emb_v2_'+s+'_'+str(idx))
        self.build_graph(hparams)   
        self.optimizer(hparams)
        params = tf.trainable_variables()

        print_out("# Trainable variables")
        for param in params:
             print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),param.op.device))   
      

        
        

    def build_graph(self, hparams):
        #lookup
        emb_inp_v2={}
        for s in hparams.single_features:
            emb_inp_v2[s]=tf.gather(self.emb_v2[s], self.single_ids[s])  
            tf.add_to_collection('l2_loss',tf.nn.l2_loss(emb_inp_v2[s])*hparams.l2)
            emb_inp_v2[s]=tf.cond(self.use_dropout, lambda: tf.nn.dropout(emb_inp_v2[s],1-hparams.dropout), lambda: emb_inp_v2[s])

        for s in hparams.seq_features:
            emb_inp_v2[s]=[]
            for idx in range(len(hparams.word2index[s])):
                emb_inp_v2[s].append(tf.gather(self.emb_v2[s][idx], self.seq_ids[s][:,:,idx]))
            emb_inp_v2[s]=tf.concat(emb_inp_v2[s],-1)
            tf.add_to_collection('l2_loss',tf.nn.l2_loss(emb_inp_v2[s])*hparams.l2)
            if s not in hparams.multi_features:
                emb_inp_v2[s]=tf.cond(self.use_dropout, lambda: tf.nn.dropout(emb_inp_v2[s],1-hparams.dropout), lambda: emb_inp_v2[s])
            
        index=[(i+0.5)/hparams.batch_num for i in range(hparams.batch_num)]    
        index=tf.constant(index) 
        for s in self.hparams.num_features:
            distance=1/(tf.abs(self.num_feaures[s][:,None]-index[None,:])+0.00001)
            weights=tf.nn.softmax(distance,-1)
            emb_inp_v2[s]=tf.reduce_sum(self.emb_v2[s][None,:,:]*weights[:,:,None],1)
            tf.add_to_collection('l2_loss',tf.nn.l2_loss(emb_inp_v2[s])*hparams.l2)
            emb_inp_v2[s]=tf.cond(self.use_dropout, lambda: tf.nn.dropout(emb_inp_v2[s],1-hparams.dropout), lambda: emb_inp_v2[s])            
        #CNN
        for s in hparams.seq_features:
            if s not in hparams.multi_features:
                with tf.variable_scope("encoder_"+s,initializer=self.initializer) as scope:
                    temp=[]
                    for cnn_dim in hparams.cnn_len:
                        filters = tf.get_variable(name="f_"+str(cnn_dim),shape=[cnn_dim,len(hparams.word2index[s])*hparams.k, hparams.filter_dim],dtype=tf.float32)
                        curr_out = tf.nn.conv1d(emb_inp_v2[s], filters=filters, stride=1, padding='VALID') 
                        curr_out=tf.reduce_max(curr_out,-2)
                        temp.append(curr_out)
                    temp=tf.concat(temp,-1)
                    W= layers_core.Dense(hparams.k,activation=tf.nn.relu, use_bias=False, name="trans_"+s)
                    emb_inp_v2[s]=W(temp)
            else:
                emb_inp_v2[s]=tf.reduce_sum(emb_inp_v2[s]*self.mulit_mask[s][:,:,None],axis=1) /tf.cast(self.seq_len[s],tf.float32)[:,None]

    
     
        y=[]
        for s in emb_inp_v2:
                y.append(emb_inp_v2[s][:,None,:])

        y=tf.concat(y,1)
        y=tf.transpose(y,[0,2,1])
        filters = tf.get_variable(name="filter",shape=[1,len( emb_inp_v2), hparams.dim],dtype=tf.float32)
        y = tf.nn.conv1d(y, filters=filters, stride=1, padding='VALID') 
        y=tf.transpose(y,[0,2,1])
        filters = tf.get_variable(name="filter_1",shape=[1,hparams.k, hparams.dim],dtype=tf.float32)
        out = tf.nn.conv1d(y, filters=filters, stride=1, padding='VALID') 
        out=tf.reshape(out,[-1,hparams.dim*hparams.dim])
        
        y=[out]
        for s in self.hparams.num_features:
            y.append(self.num_feaures[s][:,None])
            
        out=tf.concat(y,-1)    
        #dnn
        #out=self.HighwayNetwork(out)
        dnn_logits=self._build_dnn(hparams,out,hparams.dim*hparams.dim+len(self.hparams.num_features) )[:,0]

        
        score=dnn_logits
        self.prob=tf.sigmoid(score)
        logit_1=tf.log(self.prob+0.000001)
        logit_0=tf.log(1-self.prob+0.000001)
        self.loss=-tf.reduce_mean(self.label*logit_1+(1-self.label)*logit_0)
        self.cost=-tf.reduce_mean(self.label*logit_1+(1-self.label)*logit_0)+tf.add_n(tf.get_collection('l2_loss'))
        self.saver_ffm = tf.train.Saver()
        


            
    def optimizer(self,hparams):
        self.lrate=tf.Variable(hparams.learning_rate,trainable=False)
        if hparams.optimizer == "sgd":
            opt = tf.train.GradientDescentOptimizer(self.lrate)
        elif hparams.optimizer == "adam":
            opt = tf.train.AdamOptimizer(self.lrate,beta1=0.9, beta2=0.999, epsilon=1e-8)
        elif hparams.optimizer == "ada":
            opt =tf.train.AdagradOptimizer(learning_rate=self.lrate,initial_accumulator_value=1e-8)  
        params = tf.trainable_variables()

        gradients = tf.gradients(self.cost,params,colocate_gradients_with_ops=True)
        clipped_grads, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)  
        self.grad_norm =gradient_norm 
        self.update = opt.apply_gradients(zip(clipped_grads, params)) 
        
    def HighwayNetwork(self,inputs, num_layers=1, function='relu', scope='HN'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if function == 'relu':
                function = tf.nn.relu
            elif function == 'tanh':
                function = tf.nn.tanh
            else:
                raise NotImplementedError
            hidden_size = inputs.get_shape().as_list()[-1]
            memory = inputs
            for layer in range(num_layers):
                with tf.variable_scope('layer_%d' % (layer)):
                    H = layers_core.Dense(hidden_size,activation=function, use_bias=True, name="h")
                    T = layers_core.Dense(hidden_size,activation=function, use_bias=True, name="t")
                    h = H(memory)
                    t = T(memory)
                    memory = h * t + (1-t) * memory
            outputs = tf.cond(self.use_dropout,lambda :tf.nn.dropout(memory,1-self.hparams.dropout),lambda :memory)
            return outputs
          
    def dey_lrate(self,sess,lrate):
        sess.run(tf.assign(self.lrate,lrate))
        
    def train(self,sess,iterator):
        data=iterator.next()
        dic={}
        for s in self.single_ids:
            dic[self.single_ids[s]]=data[s]
            
        for s in self.seq_ids:
            dic[self.seq_ids[s]]=data[s] 
            dic[self.seq_len[s]]=data[s+'_len'] 
        for s in self.num_feaures:
            dic[self.num_feaures[s]]=data[s]
            
        dic[self.use_dropout]=True 
        dic[self.label]=data['label']

        return sess.run([self.loss,self.update,self.grad_norm],feed_dict=dic)
        
    def infer(self,sess,iterator):         
        data=iterator.next()
        dic={}
        for s in self.single_ids:
            dic[self.single_ids[s]]=data[s]
            
        for s in self.seq_ids:
            dic[self.seq_ids[s]]=data[s] 
            dic[self.seq_len[s]]=data[s+'_len'] 
        for s in self.num_feaures:
            dic[self.num_feaures[s]]=data[s]
            
        dic[self.label]=data['label']
            
        dic[self.use_dropout]=False  
        return sess.run([self.prob,self.loss],feed_dict=dic)
    
    def batch_norm_layer(self, x, train_phase, scope_bn):
        z = tf.cond(train_phase, lambda: batch_norm(x, decay=self.hparams.batch_norm_decay, center=True, scale=True, updates_collections=None,is_training=True, reuse=None, trainable=True, scope=scope_bn), lambda: batch_norm(x, decay=self.hparams.batch_norm_decay, center=True, scale=True, updates_collections=None,is_training=False, reuse=True, trainable=True, scope=scope_bn))
        return z


        
    def _build_dnn(self, hparams, embed_out, embed_layer_size):
        #embed_out=self.batch_norm_layer(embed_out,self.use_dropout,'Norm')
        w_fm_nn_input = embed_out
        last_layer_size = embed_layer_size
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(w_fm_nn_input)
        with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.layer_sizes):
                curr_w_nn_layer = tf.get_variable(name='w_nn_layer' + str(layer_idx),
                                                  shape=[last_layer_size, layer_size],
                                                  dtype=tf.float32)
                curr_b_nn_layer = tf.get_variable(name='b_nn_layer' + str(layer_idx),
                                                  shape=[layer_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer())
                curr_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx],
                                                       curr_w_nn_layer,
                                                       curr_b_nn_layer)
                scope = "nn_part" + str(idx)
                activation = hparams.activation[idx]
                curr_hidden_nn_layer = self._active_layer(logit=curr_hidden_nn_layer,
                                                          scope=scope,
                                                          activation=activation,
                                                          layer_idx=idx)
                
                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size
                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)
                
                
            w_nn_output = tf.get_variable(name='w_nn_output',
                                          shape=[last_layer_size, 1],
                                          dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                          shape=[1],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
            
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            nn_output = tf.nn.xw_plus_b(hidden_nn_layers[-1], w_nn_output, b_nn_output)
            
            return nn_output
        



def train(train_df,dev_df,test_df,hparams,idx=None):
    tf.set_random_seed(2018)
    random.seed(2018)
    np.random.seed(2018)
    tf.reset_default_graph()
    train_iterator= TextIterator(train_df,hparams,shuffle=True)   
    dev_iterator= TextIterator(dev_df,hparams,hparams.eval_batch_size)
    test_iterator= TextIterator(test_df,hparams,hparams.eval_batch_size)
    model=Model(hparams)
    config_proto = tf.ConfigProto(log_device_placement=0,allow_soft_placement=0)
    config_proto.gpu_options.allow_growth = True
    sess=tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())  
    
    dey_cont=0
    pay_cont=0
    global_step=0 
    train_loss=0
    train_norm=0
    best_loss=0
    epoch=False
    epoch_cont=0
    start_time = time.time()
    while True:
        try:
            cost,_,norm=model.train(sess,train_iterator)
            global_step+=1
            train_loss+=cost
            train_norm+=norm
        except StopIteration:
            epoch=True
            epoch_cont+=1
        if global_step%hparams.num_display_steps==0 or epoch:
            info={}
            info['learning_rate']=hparams.learning_rate
            info["train_ppl"]= train_loss / hparams.num_display_steps
            info["avg_grad_norm"]=train_norm/hparams.num_display_steps
            info["epoch"]=epoch_cont
            train_loss=0
            train_norm=0
            print_step_info("  ", global_step, info)
            if global_step%hparams.num_eval_steps==0 or epoch:
                    epoch=False
                    preds=[]
                    losses=0
                    while True:
                        try:
                            pred,loss=model.infer(sess,dev_iterator)
                            preds+=list(pred)
                            losses+=loss*len(pred)
                        except StopIteration:
                            break
                    dev_df['res']=preds
                    fpr, tpr, thresholds = metrics.roc_curve(dev_df['label']+1, dev_df['res'], pos_label=2)
                    auc=metrics.auc(fpr, tpr)
                    if best_loss<auc:
                        dey_cont=0
                        model.saver_ffm.save(sess,os.path.join(hparams.model_path, 'model_'+str(hparams.sub_name)))
                        best_loss=auc
                        T=(time.time()-start_time)
                        start_time = time.time()
                        print_out("# Epcho-time %.2fs logloss %.6f Eval AUC %.6f  Best AUC %.6f." %(T,losses/len(preds),auc,best_loss))
                    else:
                        dey_cont+=1
                        if dey_cont==hparams.dey_cont:
                            dey_cont=0
                            model.saver_ffm.restore(sess,os.path.join(hparams.model_path, 'model_'+str(hparams.sub_name)))
                            pay_cont+=1
                            hparams.learning_rate/=2.0
                            model.dey_lrate(sess,hparams.learning_rate)
                        T=(time.time()-start_time)
                        start_time = time.time()                            
                        print_out("# Epcho-time %.2fs logloss %.6f Eval AUC %.6f  Best AUC %.6f." %(T,losses/len(preds),auc,best_loss))
                        
                    if pay_cont==hparams.pay_cont:
                        model.saver_ffm.restore(sess,os.path.join(hparams.model_path, 'model_'+str(hparams.sub_name)))
                        break
    if True:
        print("Dev inference ...")
        preds=[]
        while True:
            try:
                pred,_=model.infer(sess,dev_iterator)
                preds+=list(pred)
            except StopIteration:
                break                            
        dev_df['res']=preds
        fpr, tpr, thresholds = metrics.roc_curve(dev_df['label']+1, dev_df['res'], pos_label=2)
        auc=metrics.auc(fpr, tpr)
        print('Dev inference done!')
        print("Dev auc:",round(auc,5))
        if idx:
            dev_df[['user_id','res']].to_csv('/home/kesci/work/cnn_dev_result'+str(idx)+'.csv', index=False )
        else:
            dev_df[['user_id','res']].to_csv('/home/kesci/work/cnn_dev_result.csv', index=False)
       
        print("Test inference ...")  
        preds=[]

        while True:
            try:
                pred,_=model.infer(sess,test_iterator)
                preds+=list(pred)
            except StopIteration:
                break 
        print('Test inference done!')
        test_df['res']=preds
        if idx:
            test_df[['user_id','res']].to_csv('/home/kesci/work/cnn_result'+str(idx)+'.csv', index=False)
        else:
            test_df[['user_id','res']].to_csv('/home/kesci/work/cnn_result.csv', index=False)
            test_df[['user_id','res']].to_csv('/home/kesci/work/cnn_result.txt', index=False, header=False)
        
            
    return test_df