import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.random.seed(401)
import tensorflow as tf
tf.set_random_seed(401)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, auc, roc_curve, roc_auc_score, f1_score
import data
import argparse
import sys
import pickle
import math
import random
random.seed(401)


class CASTNet:
    
    def init_weights(self, input_dim, output_dim, name, std=0.1, reg=None):
        return tf.get_variable(name,shape=[input_dim, output_dim], regularizer = reg)

    def init_bias(self, output_dim, name):
        return tf.get_variable(name,shape=[output_dim],initializer=tf.constant_initializer(1.0))
    
    def __init__(self, sess, conf):
        self.sess = sess
        self.HIDDEN_LAYER_SIZE = conf.hidden_layer_size
        self.NO_EPOCHS = conf.no_epochs
        self.TIME_STEPS = conf.window_size
        self.LEAD_TIME = conf.lead_time
        self.BATCH_SIZE = conf.batch_size
        self.TEMPORAL_FEATURE_SIZE = conf.temporal_feature_size
        self.STATIC_FEATURE_SIZE = conf.static_feature_size
        self.NO_LOCATIONS = conf.no_locations
        self.LEARNING_RATE = conf.learning_rate
        self.GROUP_LASSO = conf.group_lasso
        self.GL_REG_COEF = conf.gl_reg_coef
        self.RNN_DROPOUT = conf.rnn_dropout
        self.EMBEDDING_SIZE = conf.embedding_size
        self.DATASET_NAME = conf.dataset_name
        self.TEST_TIME = conf.test_time
        self.NO_SPATIAL_HEADS = conf.num_spatial_heads
        self.ORTHOGONAL_LOSS_COEF = conf.orthogonal_loss_coef
        self.DISP_ITER = 25
        
        self.results = {}
        self.results['preds'] = []
        self.results['loss'] = []
        
        with tf.variable_scope('placeholders'):
            self.crime_input_local = tf.placeholder(tf.float32, [None, self.TIME_STEPS, self.TEMPORAL_FEATURE_SIZE], name='crime_local_x')
            self.crime_input_global = tf.placeholder(tf.float32, [None, self.TIME_STEPS, self.NO_LOCATIONS, self.TEMPORAL_FEATURE_SIZE], name='crime_global_x')
            self.static_input = tf.placeholder(tf.float32, [None, self.STATIC_FEATURE_SIZE], name='static_x')
            self.sample_indices_input = tf.placeholder(tf.int32, [None], name='indices')
            self.dist_input = tf.placeholder(tf.float32, [None, self.NO_LOCATIONS], name='dist_x')
            self.target = tf.placeholder(tf.float32, [None], name='target_y')
            self.keep_prob = tf.placeholder(tf.float32)

            self.inf_batch_size = tf.shape(self.static_input)[0]
            self.identity = tf.eye(self.NO_SPATIAL_HEADS)
        
        
        with tf.variable_scope('static_variables'):
            self.static_input_weight = self.init_weights(self.STATIC_FEATURE_SIZE, self.HIDDEN_LAYER_SIZE, name='static_input_weight')
            self.static_input_bias = self.init_bias(self.HIDDEN_LAYER_SIZE, name='static_input_bias')
        
        with tf.variable_scope('local_variables'):
            self.local_lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.HIDDEN_LAYER_SIZE)
            self.local_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(self.local_lstm_cell, output_keep_prob=(1.0 - self.keep_prob))
        
        with tf.variable_scope('global_variables'):
            self.global_spatial_att_input_weights = []
            self.global_spatial_att_w_weights = []
            for i in range(0, self.NO_SPATIAL_HEADS):
                global_spatial_input_weight = self.init_weights(self.TEMPORAL_FEATURE_SIZE, self.TEMPORAL_FEATURE_SIZE, name='global_multi_head_att_input_weight_' + str(i))
                global_spatial_w_weight = tf.get_variable(name='global_spatial_att_w_weight_' + str(i), 
                                                             shape=[self.TEMPORAL_FEATURE_SIZE], 
                                                             regularizer = None)
                self.global_spatial_att_input_weights.append(global_spatial_input_weight)
                self.global_spatial_att_w_weights.append(global_spatial_w_weight)
            
            self.loc_emb_query_weight = self.init_weights(self.EMBEDDING_SIZE, self.HIDDEN_LAYER_SIZE, name='loc_emb_query_weight')
            
            
            self.global_lstm_cells = []
            for i in range(0, self.NO_SPATIAL_HEADS):
                global_lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.HIDDEN_LAYER_SIZE)
                global_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(global_lstm_cell, output_keep_prob=(1.0 - self.keep_prob)) 
                self.global_lstm_cells.append(global_lstm_cell)
        
        
        with tf.variable_scope('spatiotemporal_variables'):
            self.spatiotemporal_att_input_weight = self.init_weights(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE, name='spatiotemporal_att_input_weight')
            self.spatiotemporal_att_w_weight = tf.get_variable(name='spatiotemporal_att_w_weight', 
                                                               shape=[self.HIDDEN_LAYER_SIZE], 
                                                               regularizer = None)
        
        with tf.variable_scope('local_global_att_variables'):
            self.local_global_att_input_weight = self.init_weights(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE, name='local_global_att_input_weight')
            self.local_global_att_w_weight = tf.get_variable(name='local_global_att_w_weight', 
                                                               shape=[self.HIDDEN_LAYER_SIZE], 
                                                               regularizer = None)
        
        
        with tf.variable_scope('temporal_att_variables'):
            # Global temporal attention weights
            self.global_temporal_att_input_weights = []
            self.global_temporal_att_w_weights = []
            for i in range(0, self.NO_SPATIAL_HEADS):
                global_temporal_att_input_weight = self.init_weights(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE, name='global_temporal_att_input_weight_' + str(i))
                global_temporal_att_w_weight = tf.get_variable(name='global_temporal_att_w_weight_' + str(i), 
                                                             shape=[self.HIDDEN_LAYER_SIZE], 
                                                             regularizer = None)
                self.global_temporal_att_input_weights.append(global_temporal_att_input_weight)
                self.global_temporal_att_w_weights.append(global_temporal_att_w_weight)

            # Local temporal attention weights
            self.local_temporal_att_input_weight = self.init_weights(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE, name='local_temporal_att_input_weight')
            self.local_temporal_att_w_weight = tf.get_variable(name='local_temporal_att_w_weight', 
                                                             shape=[self.HIDDEN_LAYER_SIZE], 
                                                             regularizer = None)
            
        
        with tf.variable_scope('pp_variables'):
            self.Wem = self.init_weights(self.NO_LOCATIONS, self.EMBEDDING_SIZE, name='Wem')
        
        with tf.variable_scope('final_layer_variables'):
            self.final_input_weight = self.init_weights((self.HIDDEN_LAYER_SIZE * 3) + self.EMBEDDING_SIZE, 1, name='final_input_weight')
            self.final_input_bias = self.init_bias(1, name='final_input_bias')
        
        
        # construct network
        self.loc_embedded = self._location_embedding()   #(B, self.EMBEDDING_SIZE)
        self.local_rnn_outputs = self._getLocalOutputs()   #(B, D)
        self.global_rnn_outputs = self._getGlobalOutputs()   #(B, D)
        self.static_outputs = self._getStaticOutputs()   #(B, D)
        self._loc_emb_query()
        self.spatiotemporal_outputs = self._getSpatiotemporalOutputs()   #(B, D)
        self.outputs = self._getOutputs()   #(B, 1)
        self.loss = self._loss(self.outputs)
        
        '''
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(v)
        '''
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
        
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        
        self.tf_init = tf.global_variables_initializer()
        
        
    def _getGlobalOutputs(self):
        
        self.unstacked_global_spatial_att_outputs_arr = []
        unstacked_global_spatial_att_weights = []
        for i in range(0, self.NO_SPATIAL_HEADS):
            M = tf.tanh(tf.einsum("baij,jk->baik", self.crime_input_global, self.global_spatial_att_input_weights[i]))   #(B, W, L, D)
            a = tf.nn.softmax(tf.einsum("baij,j->bai", M, self.global_spatial_att_w_weights[i]))   #(B, W, L)
            unstacked_global_spatial_att_weights.append(a)
            r = tf.einsum("baij,bai->baj", self.crime_input_global, a)   #(B, W, D)
            self.unstacked_global_spatial_att_outputs_arr.append(r)
        
        self.global_spatial_att_weights = tf.stack(unstacked_global_spatial_att_weights)   #(K, B, W, L)
        self.global_spatial_att_weights = tf.transpose(self.global_spatial_att_weights, perm=[1, 2, 0, 3])    #(B, W, K, L)
        
        self._dist_query()
        
        self.global_rnn_outputs = []
        self.unstacked_global_temporal_att_weights = []
        for i in range(0, self.NO_SPATIAL_HEADS):
            with tf.variable_scope(str(i) + '_global_lstm_variables'):
                x_global = tf.unstack(self.unstacked_global_spatial_att_outputs_arr[i], self.TIME_STEPS, 1)
                global_outputs, global_states = tf.contrib.rnn.static_rnn(self.global_lstm_cells[i], x_global, dtype=tf.float32)
                global_outputs_stacked = tf.stack(global_outputs)
                global_outputs_stacked = tf.transpose(global_outputs_stacked, perm=[1, 0, 2])   #(B, W, D)

                temp_att_global = tf.nn.softmax(self.dist_query[i])   #(B, W)
                self.unstacked_global_temporal_att_weights.append(temp_att_global)  #[K, B, W]
                r_temp_global = tf.einsum("aij,ai->aj", global_outputs_stacked, temp_att_global)   #(B, D)
                self.global_rnn_outputs.append(r_temp_global)
        
        self.global_temporal_att_weights = tf.stack(self.unstacked_global_temporal_att_weights)   #(K, B, W)
        self.global_temporal_att_weights = tf.transpose(self.global_temporal_att_weights, perm=[1, 2, 0])   #(B, W, K)
        self.stacked_global_rnn_outputs = tf.stack(self.global_rnn_outputs)   #(K, B, D)
        self.stacked_global_rnn_outputs = tf.transpose(self.stacked_global_rnn_outputs, perm=[1, 0, 2])   #(B, K, D)
        return self.stacked_global_rnn_outputs
    
    
    def _getLocalOutputs(self):
        
        with tf.variable_scope('local_lstm_variables'):
            x_local = tf.unstack(self.crime_input_local, self.TIME_STEPS, 1)
            local_outputs, local_states = tf.contrib.rnn.static_rnn(self.local_lstm_cell, x_local, dtype=tf.float32)
            local_outputs_stacked = tf.stack(local_outputs)
            local_outputs_stacked = tf.transpose(local_outputs_stacked, perm=[1, 0, 2])   #(B, T, D)
            M_temp_local = tf.tanh(tf.einsum("aij,jk->aik", local_outputs_stacked, self.local_temporal_att_input_weight))   #(B, T, D)
            temp_att_local = tf.nn.softmax(tf.einsum("aij,j->ai", M_temp_local, self.local_temporal_att_w_weight))   #(B, T)
            r_temp_local = tf.einsum("aij,ai->aj", local_outputs_stacked, temp_att_local)   #(B, D)
            local_rnn_outputs = r_temp_local
            return local_rnn_outputs   #(B, D)
    
    
    def _getStaticOutputs(self):
        
        static_outputs = tf.add(tf.matmul(self.static_input, self.static_input_weight), self.static_input_bias)   #(B, D)
        static_outputs = tf.nn.sigmoid(static_outputs)   #(B, D)
        return static_outputs   #(B, D)
    
    
    def _getSpatiotemporalOutputs(self):
        
        M = tf.tanh(tf.add(tf.einsum("aij,jk->aik", self.global_rnn_outputs, self.spatiotemporal_att_input_weight), self.loc_emb_query))   #(B, K, D)
        M_2 = tf.einsum("aij,j->ai", M, self.spatiotemporal_att_w_weight) #(B, K)
        self.community_att_weights = tf.nn.softmax(M_2)  #(B, K)
        r = tf.einsum("aij,ai->aj", self.global_rnn_outputs, self.community_att_weights)  #(B, D)
        concated_st = tf.concat([r, self.local_rnn_outputs], 1)   #(B, 2*D)
        return concated_st  #(B, 2*D)
    
        
    def _getOutputs(self):
        
        outputs = tf.concat([self.spatiotemporal_outputs, self.static_outputs, self.loc_embedded], 1)
        outputs = tf.add(tf.matmul(outputs, self.final_input_weight), self.final_input_bias)   #(B, 1)
        return outputs   #(B, 1)
    
    
    def _location_embedding(self):
        
        loc_embedded = tf.nn.embedding_lookup(self.Wem, self.sample_indices_input)
        return loc_embedded # (B, T, H)
    
    
    def _orthogonal_loss(self):
        
        global_spatial_att_weights = tf.reduce_mean(self.global_spatial_att_weights, axis=1) #(B, K, L)
        transposed_stacked_global_spatial_att_weights = tf.transpose(global_spatial_att_weights, perm=[0, 2, 1])   #(B, L, K)
        mult = tf.einsum("bij,bjk->bik", global_spatial_att_weights, transposed_stacked_global_spatial_att_weights)   #(B, K, K)
        orthogonal_output = tf.square(tf.norm(tf.subtract(mult, self.identity), axis=[-2, -1]))   #(B, W)
        return tf.reduce_sum(orthogonal_output)
        
    
    def _loss(self, outputs):
        
        self.regression_loss = tf.losses.mean_squared_error(self.target, tf.squeeze(outputs))
        self.orthogonal_loss = self.ORTHOGONAL_LOSS_COEF * self._orthogonal_loss()
        loss = tf.add(self.regression_loss, self.orthogonal_loss)
        
        if(self.GROUP_LASSO == True):
            v = tf.trainable_variables()
            #self.gl_loss = self.GL_REG_COEF * self.group_regularization(v)
            self.gl_loss = self.group_regularization(v)
            loss = tf.add(loss, self.gl_loss)
        
        return loss
    
    
    def _get_batches(self, lst):
        
        data_size = len(lst)
        random.shuffle(lst)
        
        if data_size % self.BATCH_SIZE > 0:
            num_batches = int(data_size / self.BATCH_SIZE) + 1
        else:
            num_batches = int(data_size / self.BATCH_SIZE)
                
        for batch_num in range(num_batches):
            start_index = batch_num * self.BATCH_SIZE
            end_index = min((batch_num + 1) * self.BATCH_SIZE, data_size)
            yield lst[start_index:end_index]
    
    
    def _loc_emb_query(self):
        
        self.loc_emb_query = tf.tile(tf.expand_dims(self.loc_embedded, 1), (1, self.NO_SPATIAL_HEADS, 1))  #(B, K, D)
        
    
    def _dist_query(self):
        
        self.reweight_global_spatial_temporal = tf.einsum('abcd,ad->abc', self.global_spatial_att_weights, self.dist_input) #BWK
        self.reweight_global_spatial_temporal = tf.transpose(self.reweight_global_spatial_temporal, perm=[2,0,1]) #KBW
        self.dist_query = tf.unstack(self.reweight_global_spatial_temporal)
    
    
    def l21_norm(self, W):
        # Computes the L21 norm of a symbolic matrix W
        return tf.reduce_sum(tf.norm(W, axis=1))

    def group_regularization(self, v):
        # Computes a group regularization loss from a list of weight matrices corresponding
        # to the different layers.
        const_coeff = lambda W: tf.sqrt(tf.cast(W.get_shape().as_list()[1], tf.float32))
        
        static_inp_weights = [W for W in v if 'static_input_weight' in W.name]
        local_lstms = [tf.slice(W, [0, 0], [self.TEMPORAL_FEATURE_SIZE, self.HIDDEN_LAYER_SIZE * 4]) for W in v if 'local_lstm_variables/rnn/basic_lstm_cell/kernel' in W.name]
        
        self.GL_REG_COEF_LOCAL = 1. if self.DATASET_NAME == 'Cincinnati' else 0.1
        self.GL_REG_COEF_GLOBAL = 1.
        self.GL_REG_COEF_STATIC = 0.2
        
        return tf.add_n([(self.GL_REG_COEF_STATIC * self.GL_REG_COEF * tf.reduce_sum([tf.multiply(const_coeff(W),self.l21_norm(W)) for W in static_inp_weights])), 
                         (self.GL_REG_COEF_LOCAL * self.GL_REG_COEF * tf.reduce_sum([tf.multiply(const_coeff(W), self.l21_norm(W)) for W in local_lstms])),
                         (self.GL_REG_COEF_GLOBAL * self.GL_REG_COEF * tf.reduce_sum([tf.multiply(const_coeff(W), self.l21_norm(W)) for W in self.global_spatial_att_input_weights]))])
    
    
    def train(self, train_crime_local, train_crime_global, train_static, train_sample_indices, train_dist, train_y, valid_crime_local, valid_crime_global, valid_static, valid_sample_indices, valid_dist, valid_y, test_crime_local, test_crime_global, test_static, test_sample_indices, test_dist, test_y):
        
        self.sess.run(self.tf_init)
        
        self.save_model_idx = 0
        
        for epoch in range(0, self.NO_EPOCHS):
            
            batch_indices = self._get_batches(range(train_crime_local.shape[0]))
            epoch_loss = 0.
            iter_loss = 0.
            
            iter_regression_loss = 0.
            iter_gl_loss = 0.
            iter_orthogonal_loss = 0.
            epoch_regression_loss = 0.
            epoch_gl_loss = 0.
            epoch_orthogonal_loss = 0.
            
            batch_idx = 0
            for batch in batch_indices:
                
                batch_idx += 1
                
                batch_train_crime_local = train_crime_local[batch, :, :]
                batch_train_crime_global = train_crime_global[batch, :, :, :]
                batch_train_static = train_static[batch]
                batch_train_sample_indices = train_sample_indices[batch]
                batch_train_dist = train_dist[batch]
                batch_train_y = train_y[batch]

                feed_dict = {
                            self.crime_input_local: batch_train_crime_local,
                            self.crime_input_global: batch_train_crime_global,
                            self.static_input: batch_train_static,
                            self.sample_indices_input: batch_train_sample_indices, 
                            self.dist_input: batch_train_dist,
                            self.target: batch_train_y,
                            self.keep_prob: self.RNN_DROPOUT
                        }
                
                _, loss_, global_step, regression_loss, gl_loss, orthogonal_loss = self.sess.run([self.train_op, self.loss, self.global_step, self.regression_loss, self.gl_loss, self.orthogonal_loss], feed_dict=feed_dict)
                
                iter_loss += loss_
                iter_regression_loss += regression_loss
                iter_gl_loss += gl_loss
                iter_orthogonal_loss += orthogonal_loss
                
                epoch_loss += loss_
                epoch_regression_loss += regression_loss
                epoch_gl_loss += gl_loss
                epoch_orthogonal_loss += orthogonal_loss
                
            
                if ((global_step+1) % self.DISP_ITER) == 0:
                    
                    if(self.TEST_TIME == False):
                        valid_preds = self.predict(valid_crime_local, valid_crime_global, valid_static, valid_sample_indices, valid_dist, valid_y)
                        self.results['preds'].append(valid_preds)
                    elif(self.TEST_TIME == True):
                        test_preds = self.predict(test_crime_local, test_crime_global, test_static, test_sample_indices, test_dist, test_y)
                        self.results['preds'].append(test_preds)
                    
                    self.results['loss'].append(iter_loss / float(self.DISP_ITER))
                    
                    print "Iter: {}, total_loss: {}, regres_loss: {}, gl_loss: {}, ortho_loss: {}".format(((global_step+1) / self.DISP_ITER), round(iter_loss / float(self.DISP_ITER), 3), round(iter_regression_loss / float(self.DISP_ITER), 3), round(iter_gl_loss / float(self.DISP_ITER), 3), round(iter_orthogonal_loss / float(self.DISP_ITER), 3))
                    iter_loss = 0.
                    iter_regression_loss = 0.
                    iter_gl_loss = 0.
                    iter_orthogonal_loss = 0.
            
            epoch_loss /= float(batch_idx)
            epoch_regression_loss /= float(batch_idx)
            epoch_gl_loss /= float(batch_idx)
            epoch_orthogonal_loss /= float(batch_idx)
            
            if ((epoch+1) % 1) == 0:
                print "Epoch: {}, total_loss: {}, regres_loss: {}, gl_loss: {}, ortho_loss: {}".format(epoch+1, round(epoch_loss, 3), round(epoch_regression_loss, 3), round(epoch_gl_loss, 3), round(epoch_orthogonal_loss, 3))
                print '-----------------------------------------------------------------------------'
                sys.stdout.flush()
    
    
    def predict(self, crime_data_local, crime_data_global, static_data, sample_indices, dist_data, target_data):
        
        feed_dict = {
                    self.crime_input_local: crime_data_local,
                    self.crime_input_global: crime_data_global,
                    self.static_input: static_data,
                    self.sample_indices_input: sample_indices,
                    self.dist_input: dist_data,
                    self.target: target_data,
                    self.keep_prob: 0.0
                }
        
        predictions = self.sess.run([self.outputs], feed_dict=feed_dict)
        return np.squeeze(predictions)
    
    def save_results(self):
        
        save_path = './Results/' + self.DATASET_NAME + '_test_time_' + str(self.TEST_TIME) + '.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(self.results, f)
    
    
    def save(self, save_path):
        
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, save_path)

    
    def load(self, load_path):
        
        self.saver = tf.train.Saver()
        print('Loading ' + load_path)
        self.saver.restore(self.sess, load_path)
    
        
        