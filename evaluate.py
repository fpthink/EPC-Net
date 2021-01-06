import argparse
import os
import logging
import yaml
import sys
import importlib
import tensorflow as tf
import numpy as np
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
#from loading_pointclouds import get_sets_dict, load_pc_files
from loading_pointclouds import get_sets_dict, load_pc_file, load_pc_files
from sklearn.neighbors import KDTree

from tensorflow.python.client import timeline

import matplotlib
matplotlib.use("Agg")

print_log = False

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
    #parser.add_argument('--save_path', type=str, default=None, required=True, help='evaluate')
    parser.add_argument('--log_dir', type=str, default=None, help='dir for load trained model')
    parser.add_argument('--model_name', type=str, default=None, help='model_name')
    #parser.add_argument('--weight', type=str, default=None, help='weight')
    #parser.add_argument('--resume', type=str, default=None, help='resume')
    #parser.add_argument('--resume', type=str, default="train_epoch_5_iter101.pth", help='resume')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r'))
    cfg["log_dir"] = args.log_dir
    cfg["model_name"] = args.model_name
    
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

    return cfg

args = get_parser()

DATASET_FOLDER = args["DATASET_FOLDER"]                 # default: '/test/dataset/benchmark_datasets'
INPUT_DIM = args["INPUT_DIM"]
BATCH_NUM_QUERIES = 1 # 3 # args["BATCH_NUM_QUERIES"]           # FLAGS.batch_num_queries
EVAL_BATCH_SIZE = args["EVAL_BATCH_SIZE"]               # default: 1
NUM_POINTS = args["NUM_POINTS"]                         # default: 4096
POSITIVES_PER_QUERY= 0 # args["EVAL_POSITIVES_PER_QUERY"]   # default: 4
NEGATIVES_PER_QUERY= 0 # args["EVAL_NEGATIVES_PER_QUERY"]   # default: 10
GPU_INDEX = args["EVAL_GPU_INDEX"]
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_INDEX)

DECAY_STEP = args["DECAY_STEP"]                         # default: 200000
DECAY_RATE = args["DECAY_RATE"]                         # default: 0.7


DATABASE_FILE = args["EVAL_DATABASE_FILE"]  #'generating_queries/oxford_evaluation_database.pickle'
QUERY_FILE = args["EVAL_QUERY_FILE"]        #'generating_queries/oxford_evaluation_query.pickle'

LOG_DIR = args["log_dir"] #FLAGS.log_dir
#model = LOG_DIR.split('/')[1]
#RESULTS_FOLDER=os.path.join("results/", model)
strtime = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
tmp = args["model_name"].split('.')[0].split('_')


epoch_name = tmp[1]+'_'+tmp[2]
#print(strtime, epoch_name)
#exit()

dname = QUERY_FILE.split('/')[-1].split('_')[0]

RESULTS_FOLDER=os.path.join(LOG_DIR, 'results-{}'.format(dname), epoch_name+'_'+strtime)

#model = model.split('-')[0]
print(LOG_DIR)
#MODEL = importlib.import_module(model)
MODEL = importlib.import_module(args["ARCH"])  # import network module

mname = args["ARCH"]
print('model name {}'.format(mname))

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)
output_file= RESULTS_FOLDER +'/results.txt'

model_file= args["model_name"] # "model.ckpt"

DATABASE_SETS = get_sets_dict(DATABASE_FILE)
QUERY_SETS = get_sets_dict(QUERY_FILE)

global DATABASE_VECTORS
DATABASE_VECTORS = []

global QUERY_VECTORS
QUERY_VECTORS = []

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_NUM_QUERIES,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay     

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

#def load_checkpoint(restore_path):
#    print("-"*30)
#    from tensorflow.python import pywrap_tensorflow
#    reader = pywrap_tensorflow.NewCheckpointReader(restore_path)
#    var_to_shape_map = reader.get_variable_to_shape_map()
#    fout = open("params_std_pointnet.txt", "w")
#    for key in var_to_shape_map:
#        print(key)
#        fout.write("{}\n".format(key))
#    fout.close()
#    exit()

def load_checkpoint(restore_path):
    print("-"*30)
    from tensorflow.python import pywrap_tensorflow
    reader = pywrap_tensorflow.NewCheckpointReader(restore_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    fout = open("params_ptv_pointnetvlad.txt", "w")
    for key in var_to_shape_map:
        print(key)
        fout.write("{}\n".format(key))
    fout.close()
    exit()

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def evaluate():
    global DATABASE_VECTORS
    global QUERY_VECTORS
    
    global eval_database_set, eval_query_set
    eval_database_set = load_pc_data_set(DATABASE_SETS)
    eval_query_set = load_pc_data_set(QUERY_SETS)

    with tf.Graph().as_default() as graph:
        with tf.device('/gpu:'+str(GPU_INDEX)):
            print("In Graph")
            query = MODEL.placeholder_inputs(BATCH_NUM_QUERIES, 1, NUM_POINTS, INPUT_DIM)
            positives = MODEL.placeholder_inputs(BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, INPUT_DIM)
            negatives = MODEL.placeholder_inputs(BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, INPUT_DIM)
            eval_queries = MODEL.placeholder_inputs(EVAL_BATCH_SIZE, 1, NUM_POINTS, INPUT_DIM)

            is_training_pl = tf.placeholder(tf.bool, shape=())
            #print(is_training_pl)

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)

            with tf.variable_scope("query_triplets") as scope:
                vecs = tf.concat([query, positives, negatives],1)
                out_vecs = MODEL.forward(vecs, is_training_pl, bn_decay=bn_decay, params=args)
                q_vec, pos_vecs, neg_vecs = tf.split(out_vecs, [1,POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY],1)
                
            saver = tf.train.Saver()

        # Create a session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        
        model_dir = os.path.join(LOG_DIR, 'saved_model', model_file)
        if not os.path.exists(model_dir+'.meta') or not os.path.exists(model_dir+'.index') or not os.path.exists(model_dir+'.data-00000-of-00001'):
            print("Not exist model_dir: {}".format(model_dir))
            exit()
        #load_checkpoint(model_dir)  # for extract params from pretrained model
        saver.restore(sess, model_dir)
        print("Model restored:{}".format(model_dir))
        
        #stats_graph(graph)
        #exit()
        
        # timeline to trace model execute time
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        ops = {
            'query': query,
            'positives': positives,
            'negatives': negatives,
            'is_training_pl': is_training_pl,
            'eval_queries': eval_queries,
            'q_vec':q_vec,
            'pos_vecs': pos_vecs,
            'neg_vecs': neg_vecs,
            'options': options,
            'run_metadata': run_metadata,
        }
        


        recall= np.zeros(25)
        count=0
        similarity=[]
        one_percent_recall=[]
        for i in range(len(DATABASE_SETS)):
            #DATABASE_VECTORS.append(get_latent_vectors(sess, ops, DATABASE_SETS[i]))
            DATABASE_VECTORS.append(get_latent_vectors(sess, ops, DATABASE_SETS[i], eval_database_set[i]))

        for j in range(len(QUERY_SETS)):
            #QUERY_VECTORS.append(get_latent_vectors(sess, ops, QUERY_SETS[j]))
            QUERY_VECTORS.append(get_latent_vectors(sess, ops, QUERY_SETS[j], eval_query_set[j]))

        for m in range(len(QUERY_SETS)):
            for n in range(len(QUERY_SETS)):
                if (m==n):
                    continue
                
                #pair_recall, pair_similarity, pair_opr = get_recall(sess, ops, m, n)
                pair_recall, pair_similarity, pair_opr, for_plot = get_recall(sess, ops, m, n, fout=None)    # le: add fout
                
                recall+=np.array(pair_recall)
                count+=1
                one_percent_recall.append(pair_opr)
                for x in pair_similarity:
                    similarity.append(x)

        #print()
        ave_recall=recall/count
        print('ave_recallrecall')
        print(ave_recall)

        print('similarity:')
        #print(similarity)
        average_similarity= np.mean(similarity)
        print('average_similarity')
        print(average_similarity)

        ave_one_percent_recall= np.mean(one_percent_recall)
        print('ave_one_percent_recall')
        print(ave_one_percent_recall)


        #filename=RESULTS_FOLDER +'average_recall_oxford_netmax_sg(finetune_conv5).txt'
        with open(output_file, "a") as output:
            #output.write(model)
            output.write(args["ARCH"])
            output.write("\n\n")
            output.write("Average Recall @N:\n")
            output.write(str(ave_recall))
            output.write("\n\n")
            output.write("Average Similarity:\n")
            output.write(str(average_similarity))
            output.write("\n\n")
            output.write("Average Top 1% Recall:\n")
            output.write(str(ave_one_percent_recall))
            output.write("\n\n")


def get_latent_vectors(sess, ops, dict_to_process, data):
    is_training = False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))
    #print(len(train_file_idxs))
    batch_num = BATCH_NUM_QUERIES*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY)
    q_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index*batch_num: (q_index+1)*(batch_num)]
        #file_names=[]
        #for index in file_indices:
        #    file_names.append(dict_to_process[index]["query"])
        #queries=load_pc_files(file_names, DATABASE_VECTORS)
        queries = data[file_indices]    # le

        # queries= np.expand_dims(queries,axis=1)
        q1 = queries[0:BATCH_NUM_QUERIES]
        q1 = np.expand_dims(q1,axis=1)
        #print(q1.shape)

        q2 = queries[BATCH_NUM_QUERIES: BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1)]
        #q2=np.reshape(q2,(BATCH_NUM_QUERIES,POSITIVES_PER_QUERY,NUM_POINTS,13))
        q2 = np.reshape(q2, (BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, INPUT_DIM))

        q3 = queries[BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1): BATCH_NUM_QUERIES*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+1)]
        #q3=np.reshape(q3,(BATCH_NUM_QUERIES,NEGATIVES_PER_QUERY,NUM_POINTS,13))
        q3 = np.reshape(q3, (BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, INPUT_DIM))
        
        feed_dict = {
            ops['query']: q1, 
            ops['positives']: q2, 
            ops['negatives']: q3, 
            ops['is_training_pl']: is_training
        }
        #start_time = time.time()
        o1, o2, o3 = sess.run([ops['q_vec'], 
                               ops['pos_vecs'],
                               ops['neg_vecs']],
                               feed_dict=feed_dict,
                               options=ops['options'],
                               run_metadata=ops['run_metadata'])
        
        #o1, o2, o3=sess.run([ops['q_vec'], ops['pos_vecs'], ops['neg_vecs']], feed_dict=feed_dict,options=ops['options'], run_metadata=ops['run_metadata'])
        #print('use_time: ', time.time()-start_time)
        #print('o1: ', o1.shape)
       
        #fetched_timeline = timeline.Timeline(ops['run_metadata'].step_stats)
        #chrome_trace = fetched_timeline.generate_chrome_trace_format()
        #with open("calculate_time/timeline_{}_{}.json".format(q_index, mname), 'w') as f:
        #    f.write(chrome_trace)


        o1 = np.reshape(o1, (-1,o1.shape[-1]))
        o2 = np.reshape(o2, (-1,o2.shape[-1]))
        o3 = np.reshape(o3, (-1,o3.shape[-1]))

        out = np.vstack((o1, o2, o3))
        q_output.append(out)

    q_output = np.array(q_output)
    if (len(q_output) != 0):  
        q_output = q_output.reshape(-1, q_output.shape[-1])
    #print(q_output.shape)

    #handle edge case
    for q_index in range((len(train_file_idxs)//batch_num*batch_num),len(dict_to_process.keys())):
        index = train_file_idxs[q_index]
        #queries = load_pc_files([dict_to_process[index]["query"]], DATASET_FOLDER, input_dim=INPUT_DIM) # 1x4096x3
        #queries = np.expand_dims(queries, axis=1)
        queries = data[index]   # le 4096 x 3
        queries = np.expand_dims(queries, axis=0)
        queries = np.expand_dims(queries, axis=1)
        #print('queries: {}'.format(queries.shape))
        #exit()
        #fake_queries = np.zeros((BATCH_NUM_QUERIES-1, 1, NUM_POINTS, 13))
        fake_queries = np.zeros((BATCH_NUM_QUERIES-1, 1, NUM_POINTS, INPUT_DIM))
        #print('fake_queries: {}'.format(fake_queries.shape))
        #fake_pos = np.zeros((BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, 13))
        fake_pos = np.zeros((BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, INPUT_DIM))
        #fake_neg = np.zeros((BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, 13))
        fake_neg = np.zeros((BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, INPUT_DIM))
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
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    #exit(0)
    #q_output=np.array(q_output)
    #q_output=q_output.reshape(-1,q_output.shape[-1])
    #print(q_output.shape)
    return q_output

#def get_recall(sess, ops, m, n):
def get_recall(sess, ops, m, n, fout=None):
    global DATABASE_VECTORS
    global QUERY_VECTORS

    database_output= DATABASE_VECTORS[m]
    queries_output= QUERY_VECTORS[n]

    #print(len(queries_output))
    database_nbrs = KDTree(database_output)

    num_neighbors=25
    recall=[0]*num_neighbors

    top1_similarity_score=[]
    one_percent_retrieved=0
    threshold=max(int(round(len(database_output)/100.0)),1)

    num_evaluated=0
    
    for_plot = []

    for i in range(len(queries_output)):
        true_neighbors= QUERY_SETS[n][i][m]
        if(len(true_neighbors)==0):
            continue
        num_evaluated+=1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]),k=num_neighbors)
        
        # save string
        if print_log:
            st = ""
            st = st + QUERY_SETS[n][i]['query']
            st = st + '|'
            for idx in range(len(indices[0])):
                st = st + str(indices[0][idx])
                if idx != len(indices[0])-1:
                    st = st + ' '
            st = st + '|'
            for idx in range(len(true_neighbors)):
                st = st + str(true_neighbors[idx])
                if idx != len(true_neighbors)-1:
                    st = st + ' '
            st = st + '|'
            for j in range(len(indices[0])):
                #if j < threshold:
                #    print("DATABASE_SETS[m][indices[0][j]]", DATABASE_SETS[m][indices[0][j]]['query'])
                #if j < threshold-1:
                if j < len(indices[0])-1:
                    st = st + DATABASE_SETS[m][indices[0][j]]['query'] + ' '
                #if j == threshold-1:
                if j == len(indices[0])-1:
                    st = st + DATABASE_SETS[m][indices[0][j]]['query']
            print('st: ', st)
            if fout is not None:
                fout.write("{}\n".format(st))
        for_plot.append(QUERY_SETS[n][i]['easting'])
        for_plot.append(QUERY_SETS[n][i]['northing'])
        flag = False
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if(j==0):
                    similarity= np.dot(queries_output[i],database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j]+=1
                for_plot.append(j)
                flag = True
                break

        if flag is False:
            for_plot.append(25)

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors))))>0:
            one_percent_retrieved+=1

    one_percent_recall=(one_percent_retrieved/float(num_evaluated))*100
    recall=(np.cumsum(recall)/float(num_evaluated))*100
    #print('recall')
    #print(recall)
    #print('top1_simlar_score')
    #print(np.mean(top1_similarity_score))
    #print('one_percent_recall')
    #print(one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall, for_plot

def get_similarity(sess, ops, m, n):
    global DATABASE_VECTORS
    global QUERY_VECTORS

    database_output= DATABASE_VECTORS[m]
    queries_output= QUERY_VECTORS[n]

    threshold= len(queries_output)
    print(len(queries_output))
    database_nbrs = KDTree(database_output)

    similarity=[]
    for i in range(len(queries_output)):
        distances, indices = database_nbrs.query(np.array([queries_output[i]]),k=1)
        for j in range(len(indices[0])):
            q_sim= np.dot(q_output[i], database_output[indices[0][j]])
            similarity.append(q_sim)
    average_similarity=np.mean(similarity)
    #print(average_similarity)
    return average_similarity 


if __name__ == "__main__":
    evaluate()
