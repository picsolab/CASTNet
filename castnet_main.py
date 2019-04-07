import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.random.seed(401)
import tensorflow as tf
tf.set_random_seed(401)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, auc, roc_curve, roc_auc_score, f1_score
import data
import castnet_model
import argparse
import sys
import pickle
import helper

class Config:
    hidden_layer_size = 32
    no_epochs = 200
    train_ratio = 0.75
    test_ratio = 0.15
    window_size = 10
    lead_time = 1
    time_unit = 7
    group_lasso = False  # True -> gl, False -> no-gl
    dist = None
    gl_reg_coef = None
    rnn_dropout = 0.1
    temporal_feature_size = None
    static_feature_size = None
    no_locations = None
    batch_size = 50
    learning_rate = 0.001
    embedding_size = 32
    test_time = False
    dataset_name = None
    num_spatial_heads = None
    orthogonal_loss_coef = None


if __name__ == '__main__':
    
    conf = Config()
    
    parser = argparse.ArgumentParser(description='Non-Event or Event')
    parser.add_argument('--dataset_name', default='None', type=str)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--window_size', default=10, type=int)
    parser.add_argument('--lead_time', default=1, type=int)
    parser.add_argument('--gl_reg_coef', default=0.0025, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--group_lasso', default='True', type=str)
    parser.add_argument('--dist', default='root_d', type=str)
    parser.add_argument('--test_time', default='False', type=str)
    parser.add_argument('--num_spatial_heads', default=4, type=int)
    parser.add_argument('--orthogonal_loss_coef', default=0.01, type=float)

    args = parser.parse_args()
    conf.dataset_name = args.dataset_name
    conf.hidden_size = args.hidden_size
    conf.window_size = args.window_size
    conf.lead_time = args.lead_time
    conf.gl_reg_coef = args.gl_reg_coef
    conf.dropout = args.dropout
    conf.dist = args.dist
    conf.num_spatial_heads = args.num_spatial_heads
    conf.orthogonal_loss_coef = args.orthogonal_loss_coef
    conf.group_lasso = helper.str2bool(args.group_lasso)
    conf.test_time = helper.str2bool(args.test_time)
    
    train_crime_local, train_crime_global, train_static, train_sample_indices, train_dist, train_y, valid_crime_local, valid_crime_global, valid_static, valid_sample_indices, valid_dist, valid_y, test_crime_local, test_crime_global, test_static, test_sample_indices, test_dist, test_y = data.readData(conf.dataset_name, conf.window_size, conf.lead_time, conf.train_ratio, conf.test_ratio, conf.dist, conf.time_unit)
    
    print 'window_size:', conf.window_size, 'lead_time:', conf.lead_time, 'time_unit:', conf.time_unit, 'train_ratio:', conf.train_ratio, 'test_ratio:', conf.test_ratio, 'group_lasso:', conf.group_lasso, 'gl_reg_coef:', conf.gl_reg_coef, 'orthogonal_loss_coef:', conf.orthogonal_loss_coef, 'num_spatial_heads:', conf.num_spatial_heads
    sys.stdout.flush()
    
    if(conf.test_time == True):
        train_crime_global = np.concatenate([train_crime_global, valid_crime_global], axis=0)
        train_crime_local = np.concatenate([train_crime_local, valid_crime_local], axis=0)
        train_static = np.concatenate([train_static, valid_static], axis=0)
        train_sample_indices = np.concatenate([train_sample_indices, valid_sample_indices], axis=0)
        train_dist = np.concatenate([train_dist, valid_dist], axis=0)
        train_y = np.concatenate([train_y, valid_y], axis=0)
    
    train_crime_global = np.swapaxes(train_crime_global, 1, 2)
    valid_crime_global = np.swapaxes(valid_crime_global, 1, 2)
    test_crime_global = np.swapaxes(test_crime_global, 1, 2)
    
    
    print 'train_crime_global:', train_crime_global.shape, 'train_crime_local:', train_crime_local.shape, 'train_static:', train_static.shape, 'train_dist:', train_dist.shape, 'train_y:', train_y.shape
    print 'valid_crime_global:', valid_crime_global.shape, 'valid_crime_local:', valid_crime_local.shape, 'valid_static:', valid_static.shape, 'valid_dist:', valid_dist.shape, 'valid_y:', valid_y.shape
    print 'test_crime_global:', test_crime_global.shape, 'test_crime_local:', test_crime_local.shape, 'test_static:', test_static.shape, 'test_dist:', test_dist.shape, 'test_y:', test_y.shape

    
    temporal_feature_size = train_crime_global.shape[-1]
    no_locations = train_crime_global.shape[2]
    static_feature_size = train_static.shape[1]
    
    conf.temporal_feature_size = temporal_feature_size
    conf.no_locations = no_locations
    conf.static_feature_size = static_feature_size
    
    #tf.reset_default_graph()
    sess = tf.Session()
    
    castnet = castnet_model.CASTNet(sess = sess, conf = conf)
    castnet.train(train_crime_local, train_crime_global, train_static, train_sample_indices, train_dist, train_y, valid_crime_local, valid_crime_global, valid_static, valid_sample_indices, valid_dist, valid_y, test_crime_local, test_crime_global, test_static, test_sample_indices, test_dist, test_y)
    
    castnet.save_results()

