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
    parser.add_argument('--save_path', type=str, default=None, required=True, help='evaluate')
    parser.add_argument('--weight', type=str, default=None, help='weight')
    parser.add_argument('--resume', type=str, default=None, help='resume')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r'))
    cfg["save_path"] = args.save_path
    cfg["weight"] = args.weight
    cfg["resume"] = args.resume
    
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
MODEL = importlib.import_module(args["ARCH"])  # import network module
logger.info("load {}.py success!".format(args["ARCH"]))
#exit()

cmd_str = "cp ./models/{}.py {}".format(args["ARCH"], args["save_path"])
print("cmd_str: {}".format(cmd_str))
os.system(cmd_str)
#exit()

RESTORE = False
if args["resume"] is not None:
    RESTORE = True
    restore_epoch = int(args["resume"].split('_')[1].split("epoch")[1])
    print("restore_epoch: {}".format(restore_epoch))
    #exit()

DATASET_FOLDER = args["DATASET_FOLDER"]                 # default: '/test/dataset/benchmark_datasets'
INPUT_DIM = args["INPUT_DIM"]                           # default: 3
BATCH_NUM_QUERIES = args["BATCH_NUM_QUERIES"]           # FLAGS.batch_num_queries   default: 1
EVAL_BATCH_SIZE = args["EVAL_BATCH_SIZE"]               # default: 1
NUM_POINTS = args["NUM_POINTS"]                         # default: 4096
POSITIVES_PER_QUERY = args["POSITIVES_PER_QUERY"]       # FLAGS.positives_per_query default: 2
NEGATIVES_PER_QUERY = args["NEGATIVES_PER_QUERY"]       # FLAGS.negatives_per_query default: 14
MAX_EPOCH = args["MAX_EPOCH"]                           # FLAGS.max_epoch
BASE_LEARNING_RATE = args["BASE_LEARNING_RATE"]         # FLAGS.learning_rate
GPU_INDEX = args["GPU_INDEX"]                           # FLAGS.gpu
MOMENTUM = args["MOMENTUM"]                             # FLAGS.momentum
OPTIMIZER = args["OPTIMIZER"]                           # FLAGS.optimizer
DECAY_STEP = args["DECAY_STEP"]                         # FLAGS.decay_step
DECAY_RATE = args["DECAY_RATE"]                         # FLAGS.decay_rate
MARGIN1 = args["MARGIN_1"]                              # FLAGS.margin_1
MARGIN2 = args["MARGIN_2"]                              # FLAGS.margin_2
SAVE_PATH = args["save_path"] # 
# Generated training queies
#TRAIN_FILE = args["TRAIN_FILE"] # 'generating_queries/training_queries_baseline.pickle'
#TEST_FILE = args["TEST_FILE"]   # 'generating_queries/test_queries_baseline.pickle'

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_INDEX)

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

def count_params():
    n = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    #print(n)
    print("Model size: {}M".format(n/1000000.))
    return n/1000000.

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print("FLOPs: {} Trainable params: {}".format(flops.total_float_ops, params.total_parameters))

def load_checkpoint(restore_path):
    print("-"*30)
    from tensorflow.python import pywrap_tensorflow
    reader = pywrap_tensorflow.NewCheckpointReader(restore_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    fout = open("params_std_pointnet.txt", "w")
    for key in var_to_shape_map:
        print(key)
        fout.write("{}\n".format(key))
    fout.close()
    #exit()

def train():
    
    global train_data, test_data
    train_data = load_pc_data(TRAINING_QUERIES, train=True)     #   train_len x 4096 x 3 
    logger.info("load train data success!")
    test_data = load_pc_data(TEST_QUERIES, train=False)         #   test_len x 4096 x 3
    logger.info("load test data success!")
    
    global HARD_NEGATIVES
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            #print("In Graph")
            logger.info("In Graph")
            query = MODEL.placeholder_inputs(BATCH_NUM_QUERIES, 1, NUM_POINTS, INPUT_DIM)
            positives = MODEL.placeholder_inputs(BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, INPUT_DIM)
            negatives = MODEL.placeholder_inputs(BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, INPUT_DIM)
            other_negatives = MODEL.placeholder_inputs(BATCH_NUM_QUERIES,1, NUM_POINTS, INPUT_DIM)

            is_training_pl = tf.placeholder(tf.bool, shape=())
            #print(is_training_pl)
            
            batch = tf.Variable(0)
            epoch_num = tf.placeholder(tf.float32, shape=())
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            with tf.variable_scope("query_triplets") as scope:
                vecs = tf.concat([query, positives, negatives, other_negatives], 1)
                #print(vecs)                
                out_vecs = MODEL.forward(vecs, is_training_pl, bn_decay=bn_decay, params=args)
                q_vec, pos_vecs, neg_vecs, other_neg_vec= tf.split(out_vecs, [1,POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY,1],1)
                #print(q_vec)
                #print(pos_vecs)
                #print(neg_vecs)
                #print(other_neg_vec)

            #loss = MODEL.lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, MARGIN1)
            #loss = MODEL.softmargin_loss(q_vec, pos_vecs, neg_vecs)
            #loss = MODEL.quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
            loss = MODEL.lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
            tf.summary.scalar('loss', loss)

            # Get training operator
            learning_rate = get_learning_rate(epoch_num)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=0)   # max_to_keep
            
        # Create a session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        #save_name = os.path.join(args["save_path"], "saved_model/train_epoch_{}_iter{}.pth".format(str(epoch+1), str(i)))
        train_writer = tf.summary.FileWriter(os.path.join(SAVE_PATH, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(SAVE_PATH, 'test'))

        # Initialize a new model
        init = tf.global_variables_initializer()
        sess.run(init)
        #print("Initialized success!!!")
        logger.info("Initialized success!!!")
        
        tot_param = count_params()
        print("tot_param: {}M".format(tot_param))
        stats_graph(sess.graph)

        # Restore a model
        start_epoch = 1
        if RESTORE:
            #resume_filename = os.path.join(args["save_path"], "saved_model", args["resume"])
            restore_path = os.path.join(SAVE_PATH, "saved_model", args["resume"])
            saver.restore(sess, os.path.join(restore_path))
            #print("Model restored: {}".format(restore_path))
            logger.info("Model restored: {}".format(restore_path))
            start_epoch = restore_epoch


        ops = {'query': query,
               'positives': positives,
               'negatives': negatives,
               'other_negatives': other_negatives,
               'is_training_pl': is_training_pl,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'epoch_num': epoch_num,
               'lr': learning_rate,
               'q_vec': q_vec,
               'pos_vecs': pos_vecs,
               'neg_vecs': neg_vecs,
               'other_neg_vec': other_neg_vec}


        for epoch in range(start_epoch, MAX_EPOCH+1):
            #print(epoch)
            #print()
            #log_string('**** EPOCH %03d ****' % (epoch))
            #print('**** EPOCH %03d ****' % (epoch))
            logger.info('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer, test_writer, epoch, saver)          
            #eval_one_epoch(sess, ops, train_writer, test_writer, epoch, saver)          


def train_one_epoch(sess, ops, train_writer, test_writer, epoch, saver):
    global HARD_NEGATIVES
    global TRAINING_LATENT_VECTORS
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

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
    #batch_size = BATCH_NUM_QUERIES
    #cur_lr = get_learning_rate(epoch)
    
    end = time.time()
    #for i in range(len(train_file_idxs)//BATCH_NUM_QUERIES):
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
                query = get_feature_representation(batch_keys[j], TRAINING_QUERIES[batch_keys[j]]['query'], sess, ops)
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
                query = get_feature_representation(batch_keys[j], TRAINING_QUERIES[batch_keys[j]]['query'], sess, ops)
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
            #log_string('----' + str(i) + '-----')
            #print('----' + str(i) + '-----')
            #log_string('----' + 'FAULTY TUPLE' + '-----')
            #print('----' + 'FAULTY TUPLE' + '-----')
            logger.info('Epoch: [{}/{}][{}/{}] FAULTY TUPLE!!!'.format(epoch, MAX_EPOCH, i+1, iter_num))
            continue

        if no_other_neg:
            #log_string('----' + str(i) + '-----')
            #print('----' + str(i) + '-----')
            #log_string('----' + 'NO OTHER NEG' + '-----')
            #print('----' + 'NO OTHER NEG' + '-----')
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
        #log_string('----' + str(i) + '-----')
        #print('----' + str(i) + '-----')
        
        if len(queries.shape) != 4:
            #log_string('----' + 'FAULTY QUERY' + '-----')
            #print('----' + 'FAULTY QUERY' + '-----')
            logger.info('Epoch: [{}/{}][{}/{}] FAULTY TUPLE!!!'.format(epoch, MAX_EPOCH, i+1, iter_num))
            continue
        
        data_time.update(time.time() - end)     # le

        feed_dict = {
            ops['query']: queries,
            ops['positives']: positives,
            ops['negatives']: negatives,
            ops['other_negatives']: other_neg,
            ops['is_training_pl']: is_training,
            ops['epoch_num']: epoch
        }
        summary, step, train, loss_val, cur_lr = sess.run(
            [ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['lr']], 
            feed_dict=feed_dict
        )
        
        loss_meter.update(loss_val)     # le
        batch_time.update(time.time() - end)
        end = time.time()

        train_writer.add_summary(summary, step)
        #log_string('batch loss: %f' % loss_val)
        #print('batch loss: %f' % loss_val)
        
        # calculate remain time
        current_iter = epoch * iter_num + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        logger.info('Epoch: [{}/{}][{}/{}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Remain {remain_time} '
                    'Loss {loss_meter.val:.4f} '
                    'lr {lr:.8f} '.format(epoch, MAX_EPOCH, i+1, iter_num,
                                            batch_time=batch_time, data_time=data_time,
                                            remain_time=remain_time,
                                            loss_meter=loss_meter,
                                            lr=cur_lr))
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
                    #log_string('----' + 'FAULTY EVAL TUPLE' + '-----')
                    #print('----' + 'FAULTY EVAL TUPLE' + '-----')
                    logger.info('Epoch: [{}/{}][{}/{}] FAULTY EVAL TUPLE!!!'.format(epoch, MAX_EPOCH, i+1, iter_num))
                    continue

                if no_other_neg:
                    #log_string('----' + str(i) + '-----')
                    #print('----' + str(i) + '-----')
                    #log_string('----' + 'NO OTHER NEG EVAL' + '-----')
                    #print('----' + 'NO OTHER NEG EVAL' + '-----')
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
                feed_dict = {
                    ops['query']: eval_queries, 
                    ops['positives']: eval_positives, 
                    ops['negatives']: eval_negatives, 
                    ops['other_negatives']: eval_other_neg, 
                    ops['is_training_pl']: False, 
                    ops['epoch_num']: epoch
                }
                e_summary, e_step, e_loss= sess.run(
                    [ops['merged'], ops['step'], ops['loss']], 
                    feed_dict=feed_dict
                )
                eval_loss += e_loss
                if eval_batch == 4:
                    test_writer.add_summary(e_summary, e_step)
            average_eval_loss = float(eval_loss) / eval_batches_counted
            logger.info('\t\t\tEVAL')
            logger.info('\t\t\teval_loss: %f' % average_eval_loss)


        #if epoch > 5 and i%700 == 29:
        if epoch > 5 and i%(1400 // BATCH_NUM_QUERIES) == 29:
            #update cached feature vectors
            TRAINING_LATENT_VECTORS = get_latent_vectors(sess, ops, TRAINING_QUERIES)
            #print("Updated cached feature vectors")
            logger.info("Updated cached feature vectors")

        #if i%1000 == 101:
        if i % (6000 // BATCH_NUM_QUERIES) == 101:
            #save_name = os.path.join(args["save_path"], "saved_model/train_epoch_{}_iter{}.pth".format(str(epoch+1), str(i)))
            save_path = saver.save(sess, os.path.join(
                SAVE_PATH, "saved_model/model_epoch{}_iter{}.ckpt".format(str(epoch), str(i))))
            #log_string("Model saved in file: %s" % save_path)
            #print("Model saved in file: %s" % save_path)
            logger.info("Model saved in file: %s" % save_path)


# ----------------------eval----------------------------------
def eval_one_epoch(sess, ops, train_writer, test_writer, epoch, saver):
    test_save_root = os.path.join(SAVE_PATH, "test/result_epoch_{}".format(str(epoch)))
    check_makedirs(test_save_root)

    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []
    
    DATABASE_VECTORS = []   # le
    QUERY_VECTORS = []      # le
    for i in range(len(DATABASE_SETS)):
        #DATABASE_VECTORS.append(get_latent_vectors_for_eval(sess, ops, DATABASE_SETS[i]))
        DATABASE_VECTORS.append(get_latent_vectors_for_eval(sess, ops, DATABASE_SETS[i], eval_database_set[i]))
    for j in range(len(QUERY_SETS)):
        QUERY_VECTORS.append(get_latent_vectors_for_eval(sess, ops, QUERY_SETS[j], eval_query_set[j]))

    for m in range(len(QUERY_SETS)):
        for n in range(len(QUERY_SETS)):
            if m == n:
                continue
            pair_recall, pair_similarity, pair_opr = get_recall(sess, ops, m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    #print()
    ave_recall = recall/count
    logger.info('ave_recall')
    logger.info(ave_recall)

    #print('similarity:')
    #print(similarity)
    average_similarity = np.mean(similarity)
    logger.info('average_similarity')
    logger.info(average_similarity)

    ave_one_percent_recall = np.mean(one_percent_recall)
    logger.info('ave_one_percent_recall')
    logger.info(ave_one_percent_recall)


    #filename=RESULTS_FOLDER +'average_recall_oxford_netmax_sg(finetune_conv5).txt'
    with open(output_file, "a") as output:
        #output.write(model)
        output.write("\n\n")
        logger.info("\n\n")
        output.write("Average Recall @N:\n")
        logger.info("Average Recall @N:\n")
        output.write(str(ave_recall))
        logger.info(str(ave_recall))
        output.write("\n\n")
        logger.info("\n\n")
        output.write("Average Similarity:\n")
        logger.info("Average Similarity:\n")
        output.write(str(average_similarity))
        logger.info(str(average_similarity))
        output.write("\n\n")
        logger.info("\n\n")
        output.write("Average Top 1% Recall:\n")
        logger.info("Average Top 1% Recall:\n")
        output.write(str(ave_one_percent_recall))
        logger.info(str(ave_one_percent_recall))
        output.write("\n\n")
        logger.info("\n\n")

def get_recall(sess, ops, m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS):
    #global DATABASE_VECTORS
    #global QUERY_VECTORS

    database_output = DATABASE_VECTORS[m]
    queries_output = QUERY_VECTORS[n]

    #print(len(queries_output))
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        true_neighbors = QUERY_SETS[n][i][m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i],database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break
                
        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors))))>0:
            one_percent_retrieved+=1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    #logger.info('recall')
    #logger.info(recall)
    #logger.info('top1_simlar_score')
    #logger.info(np.mean(top1_similarity_score))
    #logger.info('one_percent_recall')
    #logger.info(one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall 

def get_latent_vectors_for_eval(sess, ops, dict_to_process, data):
    is_training = False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))
    #print(len(train_file_idxs))

    batch_num = BATCH_NUM_QUERIES*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY)
    #batch_num = args["EVAL_BATCH_SIZE"] * (1 + args["EVAL_POSITIVES_PER_QUERY"] + args["EVAL_NEGATIVES_PER_QUERY"])
    #EVAL_BATCH_NUM_QUERIES = args["EVAL_BATCH_SIZE"] 
    EVAL_BATCH_NUM_QUERIES = BATCH_NUM_QUERIES

    q_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index*batch_num : (q_index+1)*(batch_num)]
        #file_names = []
        #for index in file_indices:
        #    file_names.append(dict_to_process[index]["query"])
        #queries=load_pc_files(file_names)
        queries = data[file_indices]

        # queries= np.expand_dims(queries,axis=1)
        q1 = queries[0 : EVAL_BATCH_NUM_QUERIES]
        q1 = np.expand_dims(q1, axis=1)
        #print(q1.shape)

        q2 = queries[EVAL_BATCH_NUM_QUERIES : EVAL_BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1)]
        #q2 = np.reshape(q2, (EVAL_BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, 13))
        q2 = np.reshape(q2, (EVAL_BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, INPUT_DIM))

        q3 = queries[EVAL_BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1) : EVAL_BATCH_NUM_QUERIES*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+1)]
        #q3 = np.reshape(q3, (EVAL_BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, 13))
        q3 = np.reshape(q3, (EVAL_BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, INPUT_DIM))
        feed_dict = {
            ops['query']: q1, 
            ops['positives']: q2, 
            ops['negatives']: q3, 
            ops['is_training_pl']: is_training
        }
        o1, o2, o3 = sess.run(
            [ops['q_vec'], ops['pos_vecs'], ops['neg_vecs']], 
            feed_dict=feed_dict
        )
        
        o1 = np.reshape(o1, (-1, o1.shape[-1]))
        o2 = np.reshape(o2, (-1, o2.shape[-1]))
        o3 = np.reshape(o3, (-1, o3.shape[-1]))

        out = np.vstack((o1, o2, o3))
        q_output.append(out)

    q_output = np.array(q_output)
    if len(q_output) != 0:  
        q_output = q_output.reshape(-1, q_output.shape[-1])
    #print(q_output.shape)

    #handle edge case
    for q_index in range((len(train_file_idxs)//batch_num*batch_num), len(dict_to_process.keys())):
        index = train_file_idxs[q_index]
        queries = load_pc_files([dict_to_process[index]["query"]])
        queries = np.expand_dims(queries,axis=1)
        #print(query.shape)
        #exit()
        #fake_queries = np.zeros((EVAL_BATCH_NUM_QUERIES-1, 1, NUM_POINTS, 13))
        fake_queries = np.zeros((EVAL_BATCH_NUM_QUERIES-1, 1, NUM_POINTS, INPUT_DIM))
        #fake_pos = np.zeros((EVAL_BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, 13))
        fake_pos = np.zeros((EVAL_BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, INPUT_DIM))
        #fake_neg = np.zeros((EVAL_BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, 13))
        fake_neg = np.zeros((EVAL_BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, INPUT_DIM))
        q = np.vstack((queries, fake_queries))
        #print(q.shape)
        feed_dict = {
            ops['query']: q, 
            ops['positives']: fake_pos, 
            ops['negatives']: fake_neg, 
            ops['is_training_pl']: is_training
        }
        output = sess.run(ops['q_vec'], feed_dict=feed_dict)
        #print(output.shape)
        output = output[0]
        output = np.squeeze(output)
        if q_output.shape[0] != 0:
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    #q_output=np.array(q_output)
    #q_output=q_output.reshape(-1,q_output.shape[-1])
    #print(q_output.shape)
    return q_output
# ----------------------eval----------------------------------

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
        ops['query']: queries, 
        ops['positives']: fake_pos, 
        ops['negatives']: fake_neg, 
        ops['other_negatives']: fake_other_neg, 
        ops['is_training_pl']: is_training
    }
    output = sess.run(ops['q_vec'], feed_dict=feed_dict)
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
            ops['query']: q1,
            ops['positives']: q2,
            ops['negatives']: q3,
            ops['other_negatives']: q4,
            ops['is_training_pl']: is_training
        }
        o1, o2, o3, o4 = sess.run(
            [ops['q_vec'], ops['pos_vecs'], ops['neg_vecs'], ops['other_neg_vec']], 
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
            ops['query']: queries, 
            ops['positives']: fake_pos, 
            ops['negatives']: fake_neg, 
            ops['other_negatives']: fake_other_neg, 
            ops['is_training_pl']: is_training
        }
        output = sess.run(ops['q_vec'], feed_dict = feed_dict)
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
