'''
    LPD-Net Model: FN-SF-VLAD
    Feature Network + FN-Parallel structure (P) + Series-FC structure (SF)
    # Thanks to Mikaela Angelina Uy, modified from PointNetVLAD
    author: suo_ivy
    created: 10/26/18
'''
import os
import sys
import tensorflow as tf

#Taken from Charles Qi's pointnet code
MODELS_DIR = os.path.dirname(__file__)
sys.path.append(MODELS_DIR)
sys.path.append(os.path.join(MODELS_DIR, '../utils'))
import tf_util
#from transform_nets import input_transform_net, feature_transform_net, neural_feature_net

#Adopted from Antoine Meich
import loupe as lp



def placeholder_inputs(batch_num_queries, num_pointclouds_per_query, num_point, input_dim=13):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_num_queries, num_pointclouds_per_query, num_point, input_dim))
    return pointclouds_pl

#Adopted from the original pointnet code
def forward(point_cloud, is_training, bn_decay=None, params=None):
    # Network:  DGCNN    
    # INPUT:    batch_num_queries X num_pointclouds_per_query X num_points_per_pointcloud X input_dim
    # OUTPUT:   batch_num_queries X num_pointclouds_per_query X output_dim

    batch_num_queries = point_cloud.get_shape()[0].value
    num_pointclouds_per_query = point_cloud.get_shape()[1].value
    num_points = point_cloud.get_shape()[2].value
    CLUSTER_SIZE = params["CLUSTER_SIZE"]       # default: 64
    OUTPUT_DIM = params["FEATURE_OUTPUT_DIM"]   # default: 256
    k = params["KNN"]                           # default: 20
    INPUT_DIM = params["INPUT_DIM"]             # default: 13
    point_cloud = tf.reshape(point_cloud, [batch_num_queries*num_pointclouds_per_query, num_points, INPUT_DIM])
    # BxNxC
    
    with tf.variable_scope('fastdgcnn') as sc:
        dpist = tf_util.pairwise_distance_mask(point_cloud, k=k)
        # BxNxN
        # -------------------------------------------------------------------------
        x = tf_util.conv1d(point_cloud, 64, 1,
                            padding='VALID', stride=1,
                            bn=True, is_training=is_training,
                            scope='conv1', bn_decay=bn_decay)           # BxNx64
        x1 = tf.matmul(dpist, x)                                        # BxNxN X BxNx64 -> BxNx64
        x1 = x1 / float(k)
        t1 = x1 - x
        t1 = tf_util.conv1d(t1, 64, 1,
                            padding='VALID', stride=1,
                            bn=True, is_training=is_training,
                            scope='conv1_a', bn_decay=bn_decay)         # BxNx64
        t1 = tf_util.conv1d(t1, 64, 1,
                            padding='VALID', stride=1,
                            bn=True, is_training=is_training,
                            scope='conv1_b', bn_decay=bn_decay)         # BxNx64
        x1 = t1 + x1
        # -------------------------------------------------------------------------
        x = tf_util.conv1d(x1, 64, 1,
                            padding='VALID', stride=1,
                            bn=True, is_training=is_training,
                            scope='conv2', bn_decay=bn_decay)           # BxNx64
        x2 = tf.matmul(dpist, x)                                        # BxNxN X BxNx64 -> BxNx64
        x2 = x2 / float(k)
        t2 = x2 - x
        t2 = tf_util.conv1d(t2, 64, 1,
                            padding='VALID', stride=1,
                            bn=True, is_training=is_training,
                            scope='conv2_a', bn_decay=bn_decay)         # BxNx64
        t2 = tf_util.conv1d(t2, 64, 1,
                            padding='VALID', stride=1,
                            bn=True, is_training=is_training,
                            scope='conv2_b', bn_decay=bn_decay)         # BxNx64
        x2 = t2 + x2
        # -------------------------------------------------------------------------
        x = tf.concat([x1, x2], axis=-1)                        # BxNx64 * 2 -> BxNx128
        
        x = tf_util.conv1d(x, 1024, 1,
                            padding='VALID', stride=1,
                            bn=True, is_training=is_training,
                            scope='conv5', bn_decay=bn_decay)           # BxNx1024
        x = tf.expand_dims(x, axis=2)                                   # BxNx1024 -> BxNx1x1024

    with tf.variable_scope('VLAD') as vlad:
        net = tf_util.max_pool2d(x, [num_points, 1], padding='VALID', scope='maxpool')
        net = tf.reshape(net, [-1, 1024])   # batch_size X feature_size
        
        #net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope="fc1", bn_decay=bn_decay)
        output = tf_util.fully_connected(net, OUTPUT_DIM, bn=True, is_training=is_training, scope="fc1", bn_decay=bn_decay)

        #normalize to have norm 1
        output = tf.nn.l2_normalize(output,1)
        #output =  tf.reshape(output,[batch_num_queries,num_pointclouds_per_query,OUTPUT_DIM])
        output =  tf.reshape(output,[batch_num_queries,num_pointclouds_per_query,OUTPUT_DIM], name="last_output")

    return output


def best_pos_distance(query, pos_vecs):
    with tf.name_scope('best_pos_distance') as scope:
        #batch = query.get_shape()[0]
        num_pos = pos_vecs.get_shape()[1]
        query_copies = tf.tile(query, [1,int(num_pos),1]) #shape num_pos x output_dim
        best_pos=tf.reduce_min(tf.reduce_sum(tf.squared_difference(pos_vecs,query_copies),2),1)
        #best_pos=tf.reduce_max(tf.reduce_sum(tf.squared_difference(pos_vecs,query_copies),2),1)
        return best_pos



##########Losses for PointNetVLAD###########

#Returns average loss across the query tuples in a batch, loss in each is the average loss of the definite negatives against the best positive
def triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
     # ''', end_points, reg_weight=0.001):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m=tf.fill([int(batch), int(num_neg)],margin)
    triplet_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))
    return triplet_loss

#Lazy variant
def lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m=tf.fill([int(batch), int(num_neg)],margin)
    triplet_loss=tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))
    return triplet_loss


def softmargin_loss(q_vec, pos_vecs, neg_vecs):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    ones=tf.fill([int(batch), int(num_neg)],1.0)
    soft_loss=tf.reduce_mean(tf.reduce_sum(tf.log(tf.exp(tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2)))+1.0),1))
    return soft_los

def lazy_softmargin_loss(q_vec, pos_vecs, neg_vecs):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    ones=tf.fill([int(batch), int(num_neg)],1.0)
    soft_loss=tf.reduce_mean(tf.reduce_max(tf.log(tf.exp(tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2)))+1.0),1))
    return soft_loss

def quadruplet_loss_sm(q_vec, pos_vecs, neg_vecs, other_neg, m2):
    soft_loss= softmargin_loss(q_vec, pos_vecs, neg_vecs)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= soft_loss+second_loss

    return total_loss   

def lazy_quadruplet_loss_sm(q_vec, pos_vecs, neg_vecs, other_neg, m2):
    soft_loss= lazy_softmargin_loss(q_vec, pos_vecs, neg_vecs)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= soft_loss+second_loss

    return total_loss


def quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):
    trip_loss= triplet_loss(q_vec, pos_vecs, neg_vecs, m1)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= trip_loss+second_loss

    return total_loss 

def lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):
    trip_loss= lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, m1)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= trip_loss+second_loss

    return total_loss  





