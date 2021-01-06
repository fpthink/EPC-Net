import argparse
import importlib
import logging
import os
import sys
import time
import tensorflow as tf
import numpy as np
import yaml
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
#from loading_pointclouds import *
from loading_pointclouds import load_pc_file, get_queries_dict, get_query_tuple, get_sets_dict
from sklearn.neighbors import KDTree
from time_util import AverageMeter, check_makedirs

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

logger = get_logger()

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='configs/pointnetvlad_original.yaml', help='config file')
    #parser.add_argument('--save_path_teacher', type=str, default=None, required=True, help='evaluate')
    parser.add_argument('--save_path_student', type=str, default=None, required=True, help='evaluate')
    parser.add_argument('--weight', type=str, default=None, help='weight')
    parser.add_argument('--resume_teacher', type=str, default=None, help='resume')
    parser.add_argument('--resume_student', type=str, default=None, help='resume')
    #parser.add_argument('--resume', type=str, default="model_epoch7_iter6101.ckpt", help='resume')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r'))
    cfg["save_path_teacher"] = args.save_path_student
    cfg["save_path_student"] = args.save_path_student
    cfg["weight"] = args.weight
    cfg["resume_teacher"] = args.resume_teacher
    #cfg["resume_student"] = args.resume_student
    cfg["resume_student"] = args.resume_teacher.replace("teacher", "student")
    
    if cfg["FOR_DEBUG"]:
        if cfg["DATA_TYPE"] == "baseline":
            train_baseline = "training_queries_baseline_short.pickle"
            test_baseline = "test_queries_baseline_short.pickle"
        elif cfg["DATA_TYPE"] == "lpd_baseline":
            train_baseline = "training_queries_baseline_D13_short.pickle"
            test_baseline = "test_queries_baseline_D13_short.pickle"
        else:
            logger.info("DATA_TYPE is not support, only support: 'baseline' and 'lpd_baseline'")
            exit()
    else:
        if cfg["DATA_TYPE"] == "baseline":
            train_baseline = "training_queries_baseline_v1.pickle"
            test_baseline = "test_queries_baseline_v1.pickle"
        elif cfg["DATA_TYPE"] == "lpd_baseline":
            train_baseline = "training_queries_baseline_D13_v1.pickle"
            test_baseline = "test_queries_baseline_D13_v1.pickle"
        elif cfg["DATA_TYPE"] == "refine":
            train_baseline = "training_queries_refine_v1.pickle"
            test_baseline = "test_queries_baseline_v1.pickle"
        else:
            #print("DATA_TYPE is not support, only support: 'baseline' and 'refine'")
            logger.info("DATA_TYPE is not support, only support: 'baseline', 'refine', and 'lpd_baseline'")
            exit()

    cfg["TRAIN_FILE"] = os.path.join(cfg["TRAIN_FILE_ROOT"], train_baseline)
    cfg["TEST_FILE"] = os.path.join(cfg["TEST_FILE_ROOT"], test_baseline)

    # print("#"*20)
    # print("Parameters:")
    # for ky in cfg.keys():
    #     print('key: {} -> {}'.format(ky, cfg[ky]))
    # print("#"*20)
    return cfg

args = get_parser()
MODEL_teacher = importlib.import_module(args["ARCH_TEACHER"])  # import network module
logger.info("teacher ---> load {}.py success!".format(args["ARCH_TEACHER"]))

MODEL_student = importlib.import_module(args["ARCH_STUDENT"])  # import network module
logger.info("student ---> load {}.py success!".format(args["ARCH_STUDENT"]))
#exit()

cmd_str = "cp ./models/{}.py {}".format(args["ARCH_TEACHER"], os.path.join(args["save_path_student"], args["ARCH_TEACHER"]+"_teacher.py"))
# Note that the teacher model is copied to the student folder
print("cmd_str: {}".format(cmd_str))
os.system(cmd_str)

cmd_str = "cp ./models/{}.py {}".format(args["ARCH_STUDENT"], os.path.join(args["save_path_student"], args["ARCH_STUDENT"]+"_student.py"))
print("cmd_str: {}".format(cmd_str))
os.system(cmd_str)

#exit()

RESTORE_teacher = False
RESTORE_student = False

if args["resume_teacher"] is not None:
    RESTORE_teacher = True
    #RESTORE_student = True
    restore_teacher_epoch = int(args["resume_teacher"].split('_')[2].split("epoch")[1])
    print("===> restore teacher epoch: {}".format(restore_teacher_epoch))
else:
    print("===> You should use pretrained teacher model!!!")
    exit()

#RESTORE_student = False
#if args["resume_student"] is not None:
#    RESTORE_student = True
#    restore_student_epoch = int(args["resume_student"].split('_')[1].split("epoch")[1])
#    print("===> restore student epoch: {}".format(restore_student_epoch))
#else:
#    print("===> start a new training of student model!")
#exit()

DATASET_FOLDER = args["DATASET_FOLDER"]                 # default: '/test/dataset/benchmark_datasets'
INPUT_DIM = args["INPUT_DIM"]                           # default: 3
OUTPUT_DIM = args["FEATURE_OUTPUT_DIM"]                 # default: 256
BATCH_NUM_QUERIES = args["BATCH_NUM_QUERIES"]           # FLAGS.batch_num_queries   default: 1
EVAL_BATCH_SIZE = args["EVAL_BATCH_SIZE"]               # default: 1
NUM_POINTS = args["NUM_POINTS"]                         # default: 4096
POSITIVES_PER_QUERY = args["POSITIVES_PER_QUERY"]       # FLAGS.positives_per_query default: 2
NEGATIVES_PER_QUERY = args["NEGATIVES_PER_QUERY"]       # FLAGS.negatives_per_query default: 14
MAX_EPOCH = args["MAX_EPOCH"]                           # FLAGS.max_epoch
BASE_LEARNING_RATE = args["BASE_LEARNING_RATE"]         # FLAGS.learning_rate
# GPU_INDEX_teacher = args["GPU_INDEX_TEACHER"]           # FLAGS.gpu
GPU_INDEX_student = args["GPU_INDEX_STUDENT"]           # FLAGS.gpu
MOMENTUM = args["MOMENTUM"]                             # FLAGS.momentum
OPTIMIZER = args["OPTIMIZER"]                           # FLAGS.optimizer
DECAY_STEP = args["DECAY_STEP"]                         # FLAGS.decay_step
DECAY_RATE = args["DECAY_RATE"]                         # FLAGS.decay_rate
MARGIN1 = args["MARGIN_1"]                              # FLAGS.margin_1
MARGIN2 = args["MARGIN_2"]                              # FLAGS.margin_2
SAVE_PATH_teacher = args["save_path_teacher"] # 
SAVE_PATH_student = args["save_path_student"] # 
LOSS_TYPE = args["LOSS_TYPE"] 
T = args["TEMPERATURE"]
beta = args["BETA"]     # for quard
alpha = args["ALPHA"]   # for soft label
gamma = args["GAMMA"]   # for fea
# Generated training queies
#TRAIN_FILE = args["TRAIN_FILE"] # 'generating_queries/training_queries_baseline.pickle'
#TEST_FILE = args["TEST_FILE"]   # 'generating_queries/test_queries_baseline.pickle'

#os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_INDEX_teacher)+","+str(GPU_INDEX_student)
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_INDEX_student)

# Load dictionary of training queries
TRAINING_QUERIES = get_queries_dict(args["TRAIN_FILE"])
TEST_QUERIES = get_queries_dict(args["TEST_FILE"])

#DATABASE_SETS = get_sets_dict(args["EVAL_DATABASE_FILE"])
#QUERY_SETS = get_sets_dict(args["EVAL_QUERY_FILE"])

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

global HARD_NEGATIVES
HARD_NEGATIVES = {}

global TRAINING_LATENT_VECTORS
TRAINING_LATENT_VECTORS = []

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_NUM_QUERIES,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

#def log_string(out_str):
#    LOG_FOUT.write(out_str+'\n')
#    LOG_FOUT.flush()
#    print(out_str)

#learning rate halfed every 5 epoch
def get_learning_rate(epoch):
    learning_rate = BASE_LEARNING_RATE*((0.9)**(epoch//5))
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def load_pc_data(data, train=True):
    len_data = len(data.keys())
    if train:
        logger.info("train len: {}".format(len_data))
        logger.info("please wait about 14 min!...")
    else:
        logger.info("test len: {}".format(len_data))
        logger.info("please wait about some mins!...")
    pcs = []
    cnt_error = 0
    end = time.time()
    for i in range(len_data):
    # for i in tqdm(range(len_data)):
        # print(i, data[i]['query'])
        pc = load_pc_file(data[i]['query'], DATASET_FOLDER, INPUT_DIM)
        pc = pc.astype(np.float32)
        if pc.shape[0] != 4096:
            cnt_error += 1
            logger.info('error data! idx: {}'.format(i))
            continue
        pcs.append(pc)
        # if i == 100: break
    pcs = np.array(pcs)
    spd_time = (time.time() - end)/60.
    if train:
        logger.info('train data: {} load data spend: {:.6f}min'.format(pcs.shape, spd_time))
        logger.info('error train data rate: {}/{}'.format(cnt_error, len_data))
    else:
        logger.info('test data: {} load data spend: {:.6f}min'.format(pcs.shape, spd_time))
        logger.info('error test data rate: {}/{}'.format(cnt_error, len_data))
    # exit()
    return pcs

def load_pc_data_set(data_set):
    pc_set = []
    for i in range(len(data_set)):
        # print(''.format(len(data_set[i].keys())))
        pc = load_pc_data(data_set[i], train=False)
        # print(pc.shape)
        pc_set.append(pc)
        # if i == 2: break
    return pc_set

def count_params_all():
    n = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    #print("All model size: {}M".format(n/1000000.))
    return n/1000000.

def count_params(scope):
    n = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables(scope=scope)])
    #print(n)
    #print("{} model size: {}M".format(scope, n/1000000.))
    return n/1000000.

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print("FLOPs: {} Trainable params: {}".format(flops.total_float_ops, params.total_parameters))

def build_graph_teacher_v1():
    logger.info("Teacher model: In Graph")
    with tf.variable_scope("teacher") as tch:
        print("build graph teacher!")
           
        query = MODEL_teacher.placeholder_inputs(BATCH_NUM_QUERIES, 1, NUM_POINTS, INPUT_DIM)
        positives = MODEL_teacher.placeholder_inputs(BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, INPUT_DIM)
        negatives = MODEL_teacher.placeholder_inputs(BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, INPUT_DIM)
        other_negatives = MODEL_teacher.placeholder_inputs(BATCH_NUM_QUERIES,1, NUM_POINTS, INPUT_DIM)

        is_training_pl = tf.placeholder(tf.bool, shape=())
         
        batch = tf.Variable(0)
        epoch_num = tf.placeholder(tf.float32, shape=())
        bn_decay = get_bn_decay(batch)
        tf.summary.scalar('bn_decay', bn_decay)

        with tf.variable_scope("query_triplets") as scope:
            vecs = tf.concat([query, positives, negatives, other_negatives], 1)
            #print(vecs)                
            out_fea, out_vecs = MODEL_teacher.forward(vecs, is_training_pl, bn_decay=bn_decay, params=args)
            q_vec, pos_vecs, neg_vecs, other_neg_vec= tf.split(out_vecs, [1, POSITIVES_PER_QUERY, NEGATIVES_PER_QUERY, 1], 1)
            soft_label = tf.reshape(out_vecs, [-1, OUTPUT_DIM])
            

        #loss = MODEL_teacher.lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
        #tf.summary.scalar('loss_teacher', loss)

        #learning_rate = get_learning_rate(epoch_num)
        #tf.summary.scalar('learning_rate', learning_rate)
        #if OPTIMIZER == 'momentum':
        #    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
        #elif OPTIMIZER == 'adam':
        #    optimizer = tf.train.AdamOptimizer(learning_rate)

    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='teacher')
    #with tf.control_dependencies(update_ops):
    #    train_op = optimizer.minimize(loss, global_step=batch)
            
    postfix = "_teacher"
    print("postfix: {}".format(postfix))
    tf.add_to_collection("query"+postfix, query)
    tf.add_to_collection("positives"+postfix, positives)
    tf.add_to_collection("negatives"+postfix, negatives)
    tf.add_to_collection("other_negatives"+postfix, other_negatives)
    tf.add_to_collection("is_training_pl"+postfix, is_training_pl)
    #tf.add_to_collection("loss"+postfix, loss)
    #tf.add_to_collection("train_op"+postfix, train_op)
    #tf.add_to_collection("step"+postfix, batch)
    #tf.add_to_collection("epoch_num"+postfix, epoch_num)
    #tf.add_to_collection("lr"+postfix, learning_rate)
    tf.add_to_collection("q_vec"+postfix, q_vec)
    tf.add_to_collection("pos_vecs"+postfix, pos_vecs)
    tf.add_to_collection("neg_vecs"+postfix, neg_vecs)
    tf.add_to_collection("other_neg_vec"+postfix, other_neg_vec)
    tf.add_to_collection("soft_label"+postfix, soft_label)
    tf.add_to_collection("out_fea"+postfix, out_fea)

def kl_for_log_probs(log_p, log_q, reduction="mean"):
    p = tf.exp(log_p)
    neg_ent = tf.reduce_sum(p * log_p, axis=-1)
    neg_cross_ent = tf.reduce_sum(p*log_q, axis=-1)
    kl = neg_ent - neg_cross_ent
    if reduction == "mean":
        return tf.reduce_mean(kl)
    else:
        return tf.reduce_sum(kl)

def fn_mse(a, b):
    return tf.reduce_sum(tf.square(a-b))
    #return tf.reduce_mean(tf.square(a-b))
    #return tf.nn.l2_loss(a-b)
    #return tf.losses.mean_squared_error(a, b)

def square_error_sum(a, b):
    return tf.reduce_sum(tf.square(a-b))
    #return tf.reduce_mean(tf.square(a-b))
    #return tf.nn.l2_loss(a-b)
    #return tf.losses.mean_squared_error(a, b)

def square_error_mean(a, b):
    return tf.reduce_mean(tf.square(a-b))
    #return tf.reduce_mean(tf.square(a-b))
    #return tf.nn.l2_loss(a-b)
    #return tf.losses.mean_squared_error(a, b)

def build_graph_student_v1():
    logger.info("Student model: In Graph")
    with tf.variable_scope("student") as stu:
        print("build graph student!")
       
        query = MODEL_student.placeholder_inputs(BATCH_NUM_QUERIES, 1, NUM_POINTS, INPUT_DIM)
        positives = MODEL_student.placeholder_inputs(BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, INPUT_DIM)
        negatives = MODEL_student.placeholder_inputs(BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, INPUT_DIM)
        other_negatives = MODEL_student.placeholder_inputs(BATCH_NUM_QUERIES,1, NUM_POINTS, INPUT_DIM)

        soft_label = tf.placeholder(tf.float32, shape=(BATCH_NUM_QUERIES*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY+1), OUTPUT_DIM))
        #out_fea = tf.placeholder(tf.float32, shape=(BATCH_NUM_QUERIES*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY+1), NUM_POINTS, 1024))
        out_fea = tf.placeholder(tf.float32, shape=(BATCH_NUM_QUERIES*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY+1)*NUM_POINTS, 1024))
        is_training_pl = tf.placeholder(tf.bool, shape=())
            
        batch = tf.Variable(0)
        epoch_num = tf.placeholder(tf.float32, shape=())
        bn_decay = get_bn_decay(batch)
        tf.summary.scalar('bn_decay', bn_decay)

        with tf.variable_scope("query_triplets") as scope:
            vecs = tf.concat([query, positives, negatives, other_negatives], 1)
            out_fea_student, out_vecs = MODEL_student.forward(vecs, is_training_pl, bn_decay=bn_decay, params=args)
            q_vec, pos_vecs, neg_vecs, other_neg_vec = tf.split(out_vecs, [1,POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY,1],1)
            soft_label_student = tf.reshape(out_vecs, [-1, OUTPUT_DIM])

        #loss = MODEL_student.lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, MARGIN1)
        #loss = MODEL_student.softmargin_loss(q_vec, pos_vecs, neg_vecs)
        #loss = MODEL_student.quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
        loss_q = MODEL_student.lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
        if LOSS_TYPE == "kldloss":
            loss_soft = kl_for_log_probs(tf.nn.log_softmax(soft_label_student/T, axis=-1), tf.nn.softmax(soft_label/T, axis=-1)) * T * T
        elif LOSS_TYPE == "mse":
            loss_soft = fn_mse(soft_label_student, soft_label)
        elif LOSS_TYPE == "square_error_sum":
            #print('soft_label: {} soft_label_student: {}'.format(soft_label.shape, soft_label.shape))
            loss_soft = square_error_sum(soft_label_student, soft_label)
            #print('out_fea: {} out_fea_student: {}'.format(out_fea.shape, out_fea_student.shape))
            loss_fea = square_error_sum(out_fea_student, out_fea)
        elif LOSS_TYPE == "square_error_mean":
            loss_soft = square_error_mean(soft_label_student, soft_label)
            loss_fea = square_error_mean(out_fea_student, out_fea)
        elif LOSS_TYPE == "cross_entropy_loss":
            loss_soft = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=soft_label_student/T, labels=tf.nn.softmax(soft_label, axis=-1)))
            #print('loss_kl:', loss_kl)
        loss = loss_q * beta + loss_soft * alpha + loss_fea * gamma
        tf.summary.scalar('loss_student_q', loss_q)
        tf.summary.scalar('loss_student_soft', loss_soft)
        tf.summary.scalar('loss_student_fea', loss_fea)
        tf.summary.scalar('loss_student', loss)

        # Get training operator
        learning_rate = get_learning_rate(epoch_num)
        tf.summary.scalar('learning_rate', learning_rate)
        if OPTIMIZER == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
        elif OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='student')
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=batch)
  
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='student')
    #update_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='student/query_triplets/VLAD')
    #with tf.control_dependencies(update_ops):
    #    train_op = optimizer.minimize(loss, global_step=batch, var_list=update_list)
            
    postfix = "_student"
    print("postfix: {}".format(postfix))
    tf.add_to_collection("query"+postfix, query)
    tf.add_to_collection("positives"+postfix, positives)
    tf.add_to_collection("negatives"+postfix, negatives)
    tf.add_to_collection("other_negatives"+postfix, other_negatives)
    
    tf.add_to_collection("soft_label"+postfix, soft_label)                  # teacher
    tf.add_to_collection("out_fea"+postfix, out_fea)                        # teacher
    tf.add_to_collection("soft_label_self"+postfix, soft_label_student)
    tf.add_to_collection("out_fea_self"+postfix, out_fea_student) 
    
    tf.add_to_collection("is_training_pl"+postfix, is_training_pl)
    tf.add_to_collection("loss_q"+postfix, loss_q)
    tf.add_to_collection("loss_soft"+postfix, loss_soft)
    tf.add_to_collection("loss_fea"+postfix, loss_fea)
    tf.add_to_collection("loss"+postfix, loss)
    tf.add_to_collection("train_op"+postfix, train_op)
    tf.add_to_collection("step"+postfix, batch)
    tf.add_to_collection("epoch_num"+postfix, epoch_num)
    tf.add_to_collection("lr"+postfix, learning_rate)
    tf.add_to_collection("q_vec"+postfix, q_vec)
    tf.add_to_collection("pos_vecs"+postfix, pos_vecs)
    tf.add_to_collection("neg_vecs"+postfix, neg_vecs)
    tf.add_to_collection("other_neg_vec"+postfix, other_neg_vec)

def load_checkpoint(sess, restore_path, pre="teacher"):
    print("-"*30)
    from tensorflow.python import pywrap_tensorflow
    reader = pywrap_tensorflow.NewCheckpointReader(restore_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    num = 0
    for key in var_to_shape_map:
        if "BACKBONE" not in key: continue
        print(key)
        num += 1
        try:
            #with tf.variable_scope(pre, reuse=tf.AUTO_REUSE):
                #print("pre+key: {}".format(key))
                #print("pre/key: {}".format(pre+"/"+key))
            var = tf.get_variable(key)
            print(var)
            print("find {}\n".format(var))
        except ValueError as e:
            print(e)
            print("ignore {}\n".format(key))
    print("params in BACKBONE: num {}".format(num))

def train():
    global train_data, test_data
    train_data = load_pc_data(TRAINING_QUERIES, train=True)     #   train_len x 4096 x 3 
    logger.info("load train data success!")
    test_data = load_pc_data(TEST_QUERIES, train=False)         #   test_len x 4096 x 3
    logger.info("load test data success!")
    
    global HARD_NEGATIVES
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX_student)):
            # -----------------------------------------------------------
            # ---------------------build graph teacher-------------------
            # -----------------------------------------------------------
            build_graph_teacher_v1()
            ops_teacher = {}
            postfix = "_teacher"
            ops_teacher = {
                "query_teacher": tf.get_collection("query"+postfix)[0],
                "positives_teacher": tf.get_collection("positives"+postfix)[0],
                "negatives_teacher": tf.get_collection("negatives"+postfix)[0],
                "other_negatives_teacher": tf.get_collection("other_negatives"+postfix)[0],
                "is_training_pl_teacher": tf.get_collection("is_training_pl"+postfix)[0],
                #"loss_teacher": tf.get_collection("loss"+postfix)[0],
                #"train_op_teacher": tf.get_collection("train_op"+postfix)[0],
                #"step_teacher": tf.get_collection("step"+postfix)[0],
                #"epoch_num_teacher": tf.get_collection("epoch_num"+postfix)[0],
                #"lr_teacher": tf.get_collection("lr"+postfix)[0],
                "q_vec_teacher": tf.get_collection("q_vec"+postfix)[0],
                "pos_vecs_teacher": tf.get_collection("pos_vecs"+postfix)[0],
                "neg_vecs_teacher": tf.get_collection("neg_vecs"+postfix)[0],
                "other_neg_vec_teacher": tf.get_collection("other_neg_vec"+postfix)[0],
                "soft_label_teacher": tf.get_collection("soft_label"+postfix)[0],
                "out_fea_teacher": tf.get_collection("out_fea"+postfix)[0]
            }
            teacher_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="teacher")
            teacher_saver = tf.train.Saver(var_list=teacher_variables, max_to_keep=0)
            # -----------------------------------------------------------
            # ---------------------build graph student-------------------
            # -----------------------------------------------------------
            build_graph_student_v1()
            postfix = "_student"
            ops_student = {
                "query_student": tf.get_collection("query"+postfix)[0],
                "positives_student": tf.get_collection("positives"+postfix)[0],
                "negatives_student": tf.get_collection("negatives"+postfix)[0],
                "other_negatives_student": tf.get_collection("other_negatives"+postfix)[0],
                "soft_label_student": tf.get_collection("soft_label"+postfix)[0],
                "out_fea_student": tf.get_collection("out_fea"+postfix)[0],
                "soft_label_self_student": tf.get_collection("soft_label_self"+postfix)[0],
                "out_fea_self_student": tf.get_collection("out_fea_self"+postfix)[0],
                "is_training_pl_student": tf.get_collection("is_training_pl"+postfix)[0],
                "loss_q_student": tf.get_collection("loss_q"+postfix)[0],
                "loss_soft_student": tf.get_collection("loss_soft"+postfix)[0],
                "loss_fea_student": tf.get_collection("loss_fea"+postfix)[0],
                "loss_student": tf.get_collection("loss"+postfix)[0],
                "train_op_student": tf.get_collection("train_op"+postfix)[0],
                "step_student": tf.get_collection("step"+postfix)[0],
                "epoch_num_student": tf.get_collection("epoch_num"+postfix)[0],
                "lr_student": tf.get_collection("lr"+postfix)[0],
                "q_vec_student": tf.get_collection("q_vec"+postfix)[0],
                "pos_vecs_student": tf.get_collection("pos_vecs"+postfix)[0],
                "neg_vecs_student": tf.get_collection("neg_vecs"+postfix)[0],
                "other_neg_vec_student": tf.get_collection("other_neg_vec"+postfix)[0],
                #"w": tf.get_default_graph().get_tensor_by_name("student/query_triplets/BACKBONE/conv3/weights:0"),
                #"fcw": tf.get_default_graph().get_tensor_by_name("student/query_triplets/VLAD/fc1/weights:0")
            }
            # Note that we only train the fc part!
            student_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="student")
            student_saver = tf.train.Saver(var_list=student_variables, max_to_keep=0)
           
            #backbone_student_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="student/query_triplets/BACKBONE")
            #backbone_student_saver = tf.train.Saver(var_list=backbone_student_variables, max_to_keep=0)
            
            #student_variables_part = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="student")
            #for val in student_variables_part:
            #    print(val)
            #exit()
            #student_saver_part = tf.train.Saver(var_list=student_variables_part)

            # Create a session
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
            config = tf.ConfigProto(gpu_options=gpu_options)
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
        
            sess = tf.Session(config=config)

            # Add summary writers
            #merged = tf.summary.merge_all()
            #save_name = os.path.join(args["save_path"], "saved_model/train_epoch_{}_iter{}.pth".format(str(epoch+1), str(i)))
            train_writer = tf.summary.FileWriter(os.path.join(SAVE_PATH_student, 'train'), sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(SAVE_PATH_student, 'test'))

            # Initialize a new model
            init = tf.global_variables_initializer()
            sess.run(init)
            logger.info("Initialized success!!!")
            
            # --------------------------------------------------
            # --count parameters of teacher and student model!--
            # --------------------------------------------------
            tot_param_all = count_params_all()
            print("tot_param_all: {}M".format(tot_param_all))
            tot_param_teacher = count_params("teacher")
            print("tot_param_teacher: {}M".format(tot_param_teacher))
            tot_param_student = count_params("student")
            print("tot_param_student: {}M".format(tot_param_student))
            #tot_param_teacher = tot_param_all - tot_param_student
            #print("tot_param_teacher: {}M".format(tot_param_teacher)) 
            #stats_graph(sess.graph)    # Calculating FLOPs during training is not accurate.
        
            # --------------------------------------------------
            # ------Restore the teacher & student model---------
            # --------------------------------------------------
            start_epoch_teacher = 1
            #RESTORE_teacher = False
            #RESTORE_student = False

            if RESTORE_teacher:
                #resume_filename = os.path.join(args["save_path"], "saved_model", args["resume"])
                restore_path = os.path.join(SAVE_PATH_student, "transfer_teacher", args["resume_teacher"])
                print('teacher restore path: {}'.format(restore_path))
                #exit()
                #cnt = 0
                #for val in teacher_variables:
                #for val in tf.global_variables():
                #    if "BACKBONE" in val.name:
                #        print(val.name)
                #        #print(tf.get_variable(val.name))
                #        #print("\n")
                #        cnt += 1
                #print("params in BACKBONE: cnt {}".format(cnt))
                ##load_checkpoint(sess, restore_path) 
                #exit()
                teacher_saver.restore(sess, os.path.join(restore_path))
                #exit()
                logger.info("===> Teacher model restored: {}".format(restore_path))
                start_epoch_teacher = restore_teacher_epoch
            else:
                print("===> Teacher model needs load pretrained model!")
                exit()
            #exit()
            # Restore the student model
            start_epoch_student = 1
            if RESTORE_student:
                restore_path = os.path.join(SAVE_PATH_teacher, "transfer_student", args["resume_student"])
                print('student restore path: {}'.format(restore_path))
                #exit()
                #load_checkpoint(restore_path) 
                #exit()
                student_saver.restore(sess, os.path.join(restore_path))
                #backbone_student_saver.restore(sess, os.path.join(restore_path))
                logger.info("===> Student model restored pretrained teacher model!: {}".format(restore_path))
                start_epoch_student = 1     # restore_student_epoch
            #else:
            #    print("===> In kd_train_v2.py, we try to freeze the fast module and only train the fc layers! Thus, the student model load the teacher model!!!")
            #    exit()
        
            #exit()
            min_eval_loss = 100000000.
            for epoch in range(start_epoch_student, MAX_EPOCH+1):
                logger.info('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()

                #train_one_epoch(sess, ops_teacher, ops_student, train_writer, test_writer, epoch, teacher_saver, student_saver, merged) 
                train_one_epoch(sess, ops_teacher, ops_student, train_writer, test_writer, epoch, teacher_saver, student_saver, min_eval_loss) 


#def train_one_epoch(sess, ops_teacher, ops_student, train_writer, test_writer, epoch, teacher_saver, student_saver, merged):
def train_one_epoch(sess, ops_teacher, ops_student, train_writer, test_writer, epoch, teacher_saver, student_saver, min_eval_loss):
    global HARD_NEGATIVES
    global TRAINING_LATENT_VECTORS
    
    data_time = AverageMeter()
    batch_time = AverageMeter()
    batch_time_teacher = AverageMeter()
    batch_time_student = AverageMeter()
    loss_meter_teacher = AverageMeter()
    loss_q_meter_student = AverageMeter()
    loss_soft_meter_student = AverageMeter()
    loss_fea_meter_student = AverageMeter()
    loss_meter_student = AverageMeter()

    is_training = True
    sampled_neg = 4000
    #number of hard negatives in the training tuple
    #which are taken from the sampled negatives
    num_to_take = 10

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAINING_QUERIES.keys()))
    np.random.shuffle(train_file_idxs)
    
    # BATCH_NUM_QUERIES = 1
    iter_num = len(train_file_idxs)//BATCH_NUM_QUERIES
    max_iter = MAX_EPOCH * iter_num
    
    end = time.time()
    for i in range(iter_num):
        batch_keys = train_file_idxs[i*BATCH_NUM_QUERIES : (i+1)*BATCH_NUM_QUERIES]
        q_tuples = []

        faulty_tuple = False
        no_other_neg = False
        for j in range(BATCH_NUM_QUERIES):
            # positives_per_query = 2 is not satisfy
            if len(TRAINING_QUERIES[batch_keys[j]]["positives"]) < POSITIVES_PER_QUERY:     # positives_per_query = 2
                faulty_tuple = True
                break

            #no cached feature vectors               
            # training latent vectors default is []
            if len(TRAINING_LATENT_VECTORS) == 0:
                q_tuples.append(get_query_tuple(
                    batch_keys[j],  # idx
                    TRAINING_QUERIES[batch_keys[j]], POSITIVES_PER_QUERY, NEGATIVES_PER_QUERY, TRAINING_QUERIES, 
                    hard_neg=[], other_neg=True,
                    dataset_folder=DATASET_FOLDER, data=train_data))    # train_data: TxNx13
                # q_tuples.append(get_rotated_tuple(
                    #TRAINING_QUERIES[batch_keys[j]], POSITIVES_PER_QUERY, NEGATIVES_PER_QUERY, TRAINING_QUERIES, 
                    #hard_neg=[], other_neg=True))
                # q_tuples.append(get_jittered_tuple(
                    #TRAINING_QUERIES[batch_keys[j]], POSITIVES_PER_QUERY, NEGATIVES_PER_QUERY, TRAINING_QUERIES, 
                    #hard_neg=[], other_neg=True))

            elif len(HARD_NEGATIVES.keys()) == 0:
                query = get_feature_representation(batch_keys[j], TRAINING_QUERIES[batch_keys[j]]['query'], sess, ops_student)
                np.random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]]['negatives'][0 : sampled_neg]
                hard_negs = get_random_hard_negatives(query, negatives, num_to_take)
                #print(hard_negs)
                q_tuples.append(get_query_tuple(
                    batch_keys[j],
                    TRAINING_QUERIES[batch_keys[j]], POSITIVES_PER_QUERY, NEGATIVES_PER_QUERY, TRAINING_QUERIES, 
                    hard_negs, other_neg=True,
                    dataset_folder=DATASET_FOLDER, data=train_data))
                # q_tuples.append(get_rotated_tuple(
                    #TRAINING_QUERIES[batch_keys[j]], POSITIVES_PER_QUERY, NEGATIVES_PER_QUERY, TRAINING_QUERIES, 
                    #hard_negs, other_neg=True))
                # q_tuples.append(get_jittered_tuplej(
                    #TRAINING_QUERIES[batch_keys[j]], POSITIVES_PER_QUERY, NEGATIVES_PER_QUERY, TRAINING_QUERIES, 
                    #hard_negs, other_neg=True))
            else:
                query = get_feature_representation(batch_keys[j], TRAINING_QUERIES[batch_keys[j]]['query'], sess, ops_student)
                np.random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]]['negatives'][0 : sampled_neg]
                hard_negs = get_random_hard_negatives(query, negatives, num_to_take)
                hard_negs = list(set().union(HARD_NEGATIVES[batch_keys[j]], hard_negs))
                #print('hard', hard_negs)
                q_tuples.append(get_query_tuple(
                    batch_keys[j],
                    TRAINING_QUERIES[batch_keys[j]], POSITIVES_PER_QUERY, NEGATIVES_PER_QUERY, TRAINING_QUERIES, 
                    hard_negs, other_neg=True,
                    dataset_folder=DATASET_FOLDER, data=train_data))           
                # q_tuples.append(get_rotated_tuple(
                    #TRAINING_QUERIES[batch_keys[j]], POSITIVES_PER_QUERY, NEGATIVES_PER_QUERY, TRAINING_QUERIES, 
                    #hard_negs, other_neg=True))           
                # q_tuples.append(get_jittered_tuple(
                    #TRAINING_QUERIES[batch_keys[j]], POSITIVES_PER_QUERY, NEGATIVES_PER_QUERY, TRAINING_QUERIES, 
                    #hard_negs, other_neg=True))
            
            if q_tuples[j][3].shape[0] != NUM_POINTS:
                no_other_neg = True
                break

        #construct query array
        if faulty_tuple:
            logger.info('Epoch: [{}/{}][{}/{}] FAULTY TUPLE!!!'.format(epoch, MAX_EPOCH, i+1, iter_num))
            continue

        if no_other_neg:
            logger.info('Epoch: [{}/{}][{}/{}] NO OTHER NEG!!!'.format(epoch, MAX_EPOCH, i+1, iter_num))
            continue            

        queries = []
        positives = []
        negatives = []
        other_neg = []
        for k in range(len(q_tuples)):
            queries.append(q_tuples[k][0])
            positives.append(q_tuples[k][1])
            negatives.append(q_tuples[k][2])
            other_neg.append(q_tuples[k][3])

        queries = np.array(queries)
        queries = np.expand_dims(queries,axis=1)
        other_neg = np.array(other_neg)
        other_neg = np.expand_dims(other_neg,axis=1)
        positives = np.array(positives)
        negatives = np.array(negatives)
        
        if len(queries.shape) != 4:
            logger.info('Epoch: [{}/{}][{}/{}] FAULTY TUPLE!!!'.format(epoch, MAX_EPOCH, i+1, iter_num))
            continue
        
        data_time.update(time.time() - end)     # le
        
        # -----------------teacher model---------------------
        feed_dict_teacher = {
            ops_teacher['query_teacher']: queries,
            ops_teacher['positives_teacher']: positives,
            ops_teacher['negatives_teacher']: negatives,
            ops_teacher['other_negatives_teacher']: other_neg,
            ops_teacher['is_training_pl_teacher']: False        # always False
        }
        
        # ?x256
        #soft_label_teacher = sess.run(ops_teacher['soft_label_teacher'], feed_dict=feed_dict_teacher)
        
        # ?xNx1024, ?x256
        out_fea_teacher, soft_label_teacher = sess.run([ops_teacher['out_fea_teacher'], ops_teacher['soft_label_teacher']], feed_dict=feed_dict_teacher)
        #print()
        #print("out_fea_teacher: {}".format(out_fea_teacher.shape))
        #print("out_fea_teacher: {}".format(out_fea_teacher[0, 0, :10]))
        #print("out_fea_teacher: {}".format(out_fea_teacher[0, 1, :10]))
        #print("soft_label_teacher: {}".format(soft_label_teacher.shape))
        #print("soft_label_teacher: {}".format(soft_label_teacher[0, :10]))
        # -----------------student model---------------------
        #print("queries: {}".format(queries.shape))
        #print("positives: {}".format(positives.shape))
        #print("negatives: {}".format(negatives.shape))
        #print("other_negatives: {}".format(other_neg.shape))
        #print("soft_label_teacher: {}".format(soft_label_teacher.shape))

        feed_dict_student = {
            ops_student['query_student']: queries,
            ops_student['positives_student']: positives,
            ops_student['negatives_student']: negatives,
            ops_student['other_negatives_student']: other_neg,
            ops_student['is_training_pl_student']: is_training,
            ops_student['soft_label_student']: soft_label_teacher,      # teacher
            ops_student['out_fea_student']: out_fea_teacher,            # teacher
            ops_student['epoch_num_student']: epoch
        }
        step_student, train_student, soft_label_self_student, loss_q_student, loss_soft_student, loss_fea_student, loss_student, cur_lr_student = sess.run(
            [ops_student['step_student'], ops_student['train_op_student'], ops_student['soft_label_self_student'],
             ops_student['loss_q_student'], ops_student['loss_soft_student'], ops_student['loss_fea_student'], ops_student['loss_student'], ops_student['lr_student']], 
            feed_dict=feed_dict_student
        )
        #print("stu_w: {}".format(stu_w))
        #print("stu_fcw: {}".format(stu_w))
        #print("soft_label_self_student: {}".format(soft_label_self_student[0, :10]))
        #exit()
        #print('loss_q_student: {}'.format(loss_q_student)) 
        #print('loss_soft_student: {}'.format(loss_soft_student)) 
        loss_q_meter_student.update(loss_q_student)
        loss_soft_meter_student.update(loss_soft_student)
        loss_fea_meter_student.update(loss_fea_student)
        loss_meter_student.update(loss_student)     # le
        batch_time.update(time.time() - end)
        end = time.time()
        #print('lr: ', cur_lr_student)
        #exit()
        #train_writer.add_summary(summary_student, step_student)
        
        # calculate remain time
        current_iter = epoch * iter_num + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        logger.info('E: [{}/{}][{}/{}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'B {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Remain {remain_time} '
                    'Ls q {loss_q_meter_student.val:.4f} '
                    'Ls soft {loss_soft_meter_student.val:.4f} '
                    'Ls fea {loss_fea_meter_student.val:.4f} '
                    'Ls sum {loss_meter_student.val:.4f} '
                    'lr {lr:.8f} '.format(epoch, MAX_EPOCH, i+1, iter_num,
                                            batch_time=batch_time, data_time=data_time,
                                            remain_time=remain_time,
                                            loss_q_meter_student=loss_q_meter_student,
                                            loss_soft_meter_student=loss_soft_meter_student,
                                            loss_fea_meter_student=loss_fea_meter_student,
                                            loss_meter_student=loss_meter_student,
                                            lr=cur_lr_student))
        if i%200 == 7:
            test_file_idxs = np.arange(0, len(TEST_QUERIES.keys()))
            np.random.shuffle(test_file_idxs)

            eval_loss = 0
            eval_batches = 5
            eval_batches_counted = 0
            for eval_batch in range(eval_batches):
                eval_keys = test_file_idxs[eval_batch*BATCH_NUM_QUERIES : (eval_batch+1)*BATCH_NUM_QUERIES]
                eval_tuples = []

                faulty_eval_tuple = False
                no_other_neg = False
                for e_tup in range(BATCH_NUM_QUERIES):
                    if len(TEST_QUERIES[eval_keys[e_tup]]["positives"]) < POSITIVES_PER_QUERY:
                        faulty_eval_tuple = True
                        break
                    
                    eval_tuples.append(get_query_tuple(
                        eval_keys[e_tup],
                        TEST_QUERIES[eval_keys[e_tup]], POSITIVES_PER_QUERY, NEGATIVES_PER_QUERY, TEST_QUERIES, 
                        hard_neg=[], other_neg=True,
                        dataset_folder=DATASET_FOLDER, data=test_data))           

                    if eval_tuples[e_tup][3].shape[0] != NUM_POINTS:
                        no_other_neg = True
                        break

                if faulty_eval_tuple:
                    logger.info('Epoch: [{}/{}][{}/{}] FAULTY EVAL TUPLE!!!'.format(epoch, MAX_EPOCH, i+1, iter_num))
                    continue

                if no_other_neg:
                    logger.info('Epoch: [{}/{}][{}/{}] NO OTHER NEG EVAL!!!'.format(epoch, MAX_EPOCH, i+1, iter_num))
                    continue  

                eval_batches_counted += 1
                eval_queries = []
                eval_positives = []
                eval_negatives = []
                eval_other_neg = []

                for tup in range(len(eval_tuples)):
                    eval_queries.append(eval_tuples[tup][0])
                    eval_positives.append(eval_tuples[tup][1])
                    eval_negatives.append(eval_tuples[tup][2])
                    eval_other_neg.append(eval_tuples[tup][3])

                eval_queries = np.array(eval_queries)
                eval_queries = np.expand_dims(eval_queries,axis=1)                
                eval_other_neg = np.array(eval_other_neg)
                eval_other_neg = np.expand_dims(eval_other_neg,axis=1)
                eval_positives = np.array(eval_positives)
                eval_negatives = np.array(eval_negatives)
                feed_dict_student = {
                    ops_student['query_student']: eval_queries, 
                    ops_student['positives_student']: eval_positives, 
                    ops_student['negatives_student']: eval_negatives, 
                    ops_student['other_negatives_student']: eval_other_neg, 
                    ops_student['is_training_pl_student']: False, 
                    ops_student['epoch_num_student']: epoch
                }
                #e_summary, e_step, e_loss= sess.run(
                #    [merged, ops_student['step_student'], ops_student['loss_student']], 
                #    feed_dict=feed_dict_student
                #)
                e_step, e_q_loss= sess.run(
                    [ops_student['step_student'], ops_student['loss_q_student']], 
                    feed_dict=feed_dict_student
                )
                eval_loss += e_q_loss
                #if eval_batch == 4:
                #    test_writer.add_summary(e_summary, e_step)
            average_eval_loss = float(eval_loss) / eval_batches_counted
            logger.info('\t\t\tEVAL')
            logger.info('\t\t\teval_loss: %f' % average_eval_loss)
            if average_eval_loss < min_eval_loss:
                min_eval_loss = average_eval_loss
                save_path = student_saver.save(sess, os.path.join(
                    SAVE_PATH_student, "saved_model/min_eval_student_model_epoch{}_iter{}.ckpt".format(str(epoch), str(i))))
                logger.info("Student model saved in file: %s" % save_path)



        #if epoch > 5 and i%700 == 29:
        if epoch > 5 and i%(1400 // BATCH_NUM_QUERIES) == 29:
        #if epoch > 15 and i%(1400 // BATCH_NUM_QUERIES) == 29:
            TRAINING_LATENT_VECTORS = get_latent_vectors(sess, ops_student, TRAINING_QUERIES)
            logger.info("Updated cached feature vectors")

        #if i%1000 == 101:
        if i % (6000 // BATCH_NUM_QUERIES) == 101:
            save_path = student_saver.save(sess, os.path.join(
                SAVE_PATH_student, "saved_model/student_model_epoch{}_iter{}.ckpt".format(str(epoch), str(i))))
            logger.info("Student model saved in file: %s" % save_path)


def get_feature_representation(idx, filename, sess, ops):
    is_training = False
    
    #queries = load_pc_files([filename])
    idxs = [idx]
    queries = train_data[idxs] 
    queries = np.expand_dims(queries, axis=1)
    
    if BATCH_NUM_QUERIES-1 > 0:
        #fake_queries = np.zeros((BATCH_NUM_QUERIES-1, 1, NUM_POINTS, 13))
        fake_queries = np.zeros((BATCH_NUM_QUERIES-1, 1, NUM_POINTS, INPUT_DIM))
        queries = np.vstack((queries, fake_queries))
    #else:
    #    q = queries
    
    #fake_pos = np.zeros((BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, 13))
    fake_pos = np.zeros((BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, INPUT_DIM))
    #fake_neg = np.zeros((BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, 13))
    fake_neg = np.zeros((BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, INPUT_DIM))
    #fake_other_neg = np.zeros((BATCH_NUM_QUERIES, 1, NUM_POINTS, 13))
    fake_other_neg = np.zeros((BATCH_NUM_QUERIES, 1, NUM_POINTS, INPUT_DIM))
    feed_dict = {
        ops['query_student']: queries, 
        ops['positives_student']: fake_pos, 
        ops['negatives_student']: fake_neg, 
        ops['other_negatives_student']: fake_other_neg, 
        ops['is_training_pl_student']: is_training
    }
    output = sess.run(ops['q_vec_student'], feed_dict=feed_dict)
    output = output[0]
    output = np.squeeze(output)
    return output

def get_random_hard_negatives(query_vec, random_negs, num_to_take):
    global TRAINING_LATENT_VECTORS

    latent_vecs = []
    for j in range(len(random_negs)):
        latent_vecs.append(TRAINING_LATENT_VECTORS[random_negs[j]])
    
    latent_vecs = np.array(latent_vecs)
    nbrs = KDTree(latent_vecs)
    distances, indices = nbrs.query(np.array([query_vec]), k=num_to_take)
    hard_negs = np.squeeze(np.array(random_negs)[indices[0]])
    hard_negs = hard_negs.tolist()
    return hard_negs

def get_latent_vectors(sess, ops, dict_to_process):
    is_training = False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = BATCH_NUM_QUERIES * (1 + POSITIVES_PER_QUERY + NEGATIVES_PER_QUERY + 1)
    q_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index*batch_num : (q_index+1)*(batch_num)]
        #file_names = []
        #for index in file_indices:
        #    file_names.append(dict_to_process[index]["query"])
        #queries = load_pc_files(file_names)
        queries = train_data[file_indices]  # le

        q1 = queries[0 : BATCH_NUM_QUERIES]
        q1 = np.expand_dims(q1, axis=1)

        q2 = queries[BATCH_NUM_QUERIES : BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1)]
        #q2 = np.reshape(q2, (BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, 13))
        q2 = np.reshape(q2, (BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, INPUT_DIM))

        q3 = queries[BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1) : BATCH_NUM_QUERIES*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+1)]
        #q3 = np.reshape(q3, (BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, 13))
        q3 = np.reshape(q3, (BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, INPUT_DIM))

        q4 = queries[BATCH_NUM_QUERIES*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+1)
                     : BATCH_NUM_QUERIES*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+2)]
        q4 = np.expand_dims(q4, axis=1)

        feed_dict = {
            ops['query_student']: q1,
            ops['positives_student']: q2,
            ops['negatives_student']: q3,
            ops['other_negatives_student']: q4,
            ops['is_training_pl_student']: is_training
        }
        o1, o2, o3, o4 = sess.run(
            [ops['q_vec_student'], ops['pos_vecs_student'], ops['neg_vecs_student'], ops['other_neg_vec_student']], 
            feed_dict=feed_dict)
        
        o1 = np.reshape(o1, (-1, o1.shape[-1]))
        o2 = np.reshape(o2, (-1, o2.shape[-1]))
        o3 = np.reshape(o3, (-1, o3.shape[-1]))
        o4 = np.reshape(o4, (-1, o4.shape[-1]))        

        out = np.vstack((o1, o2, o3, o4))
        q_output.append(out)

    q_output = np.array(q_output)
    if len(q_output) != 0:  
        q_output = q_output.reshape(-1, q_output.shape[-1])

    #handle edge case
    #index_edge = len(train_file_idxs) // batch_num * batch_num          # le
    #while index_edge+BATCH_NUM_QUERIES < len(dict_to_process.keys()):   # le

    for q_index in range((len(train_file_idxs)//batch_num*batch_num), len(dict_to_process.keys())):
        #index = train_file_idxs[q_index]
        #queries = load_pc_files([dict_to_process[index]["query"]])
        index = [train_file_idxs[q_index]]  # le
        #index = [train_file_idxs[index_edge]]
        queries = train_data[index]         # le
        queries = np.expand_dims(queries,axis=1)

        if BATCH_NUM_QUERIES-1 > 0:
            #fake_queries = np.zeros((BATCH_NUM_QUERIES-1, 1, NUM_POINTS, 13))
            fake_queries = np.zeros((BATCH_NUM_QUERIES-1, 1, NUM_POINTS, INPUT_DIM))
            queries = np.vstack((queries, fake_queries))
        #else:
        #    q = queries

        #fake_pos = np.zeros((BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, 13))
        fake_pos = np.zeros((BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, INPUT_DIM))
        #fake_neg = np.zeros((BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, 13))
        fake_neg = np.zeros((BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, INPUT_DIM))
        #fake_other_neg = np.zeros((BATCH_NUM_QUERIES, 1, NUM_POINTS, 13))
        fake_other_neg = np.zeros((BATCH_NUM_QUERIES, 1, NUM_POINTS, INPUT_DIM))
        feed_dict = {
            ops['query_student']: queries, 
            ops['positives_student']: fake_pos, 
            ops['negatives_student']: fake_neg, 
            ops['other_negatives_student']: fake_other_neg, 
            ops['is_training_pl_student']: is_training
        }
        output = sess.run(ops['q_vec_student'], feed_dict = feed_dict)
        output = output[0]
        output = np.squeeze(output)
        if q_output.shape[0] != 0:
            q_output = np.vstack((q_output, output))
        else:
            q_output = output
        #index_edge = index_edge + BATCH_NUM_QUERIES

    #print(q_output.shape)
    return q_output

if __name__ == "__main__":
    train()
