import os
import pickle
import numpy as np
import random

#UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
#DATASET_PATH=os.path.join(UTILS_DIR, '../../benchmark_datasets/')
#print(DATASET_PATH)
#BASE_PATH = "/media/deep-three/Deep_Store/CVPR2018/benchmark_datasets/"

def get_queries_dict(filename):
    #key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
    with open(filename, 'rb') as handle:
        queries = pickle.load(handle)
        print("Queries Loaded.")
        return queries

def get_sets_dict(filename):
    #[key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},
    # key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
    with open(filename, 'rb') as handle:
        trajectories = pickle.load(handle)
        print("Trajectories Loaded.")
        return trajectories

def load_pc_file(filename, dataset_folder, input_dim=3):
    #returns Nx13 matrix (3 pose 10 handcraft features)
    #pc = np.fromfile(os.path.join(DATASET_PATH, filename), dtype=np.float64)
    pc = np.fromfile(os.path.join(dataset_folder, filename), dtype=np.float64)
    
    if input_dim == 3:
        if pc.shape[0] != 4096*3:
            print("Error in pointcloud shape")
            print(pc.shape)
            print(filename)
            #return np.array([])
            return np.zeros([4096, 3])
        pc = np.reshape(pc,(pc.shape[0]//3, 3))
    else:
        if pc.shape[0]!= 4096*13:
            print("Error in pointcloud shape")
            print(pc.shape)
            print(filename)
            #return np.array([])
            return np.zeros([4096, 13])
        pc = np.reshape(pc, (pc.shape[0]//13, 13))
        # preprocessing data
        # Normalization
        pc[:,3:12] = ((pc-pc.min(axis=0))/(pc.max(axis=0)-pc.min(axis=0)))[:,3:12]
        pc[np.isnan(pc)] = 0.0 # some pcs are NAN
        pc[np.isinf(pc)] = 1.0 # some pcs are INF

    return pc

#def load_pc_files(filenames):
def load_pc_files(filenames, dataset_folder, input_dim=3):
    pcs = []
    for filename in filenames:
        #print(filename)
        pc = load_pc_file(filename, dataset_folder, input_dim)
        if pc.shape[0] != 4096:
            continue
        pcs.append(pc)
    pcs=np.array(pcs)
    return pcs

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        #-90 to 90
        rotation_angle = (np.random.uniform()*np.pi)- np.pi/2.0
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def jitter_point_cloud(batch_data, sigma=0.005, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def get_query_tuple(idx, dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False,
                   dataset_folder=None, data=None):
    #get query tuple for dictionary entry
    #return list [query,positives,negatives]

    #query=load_pc_file(dict_value["query"]) #Nx13
    query = data[idx]   # TxNx13 -> Nx13

    random.shuffle(dict_value["positives"])
    #pos_files=[]
    pos_files_idx = []

    for i in range(num_pos):
        #pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
        pos_files_idx.append(dict_value["positives"][i])
    
    #positives= load_pc_files(dict_value["positives"][0:num_pos])
    #positives=load_pc_files(pos_files)
    positives = data[pos_files_idx]

    #neg_files=[]
    neg_indices = []
    if len(hard_neg) == 0:
        random.shuffle(dict_value["negatives"])	
        for i in range(num_neg):
            #neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            #neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j = 0
        #while len(neg_files) < num_neg:
        while len(neg_indices) < num_neg:
            if not dict_value["negatives"][j] in hard_neg:
                #neg_files.append(QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    
    #negatives=load_pc_files(neg_files)
    negatives = data[neg_indices]

    if other_neg == False:
        return [query, positives, negatives]
    #For Quadruplet Loss
    else:
        #get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)

        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)

        # erase the whole possible idx, which is the potential areas 
        possible_negs= list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)

        if len(possible_negs) == 0:
            return [query, positives, negatives, np.array([])]

        #neg2= load_pc_file(QUERY_DICT[possible_negs[0]]["query"])
        neg2 = data[possible_negs[0]]

        return [query, positives, negatives, neg2]


def get_rotated_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False, dataset_folder=None):
    query=load_pc_file(dict_value["query"], dataset_folder) #Nx3 le
    q_rot= rotate_point_cloud(np.expand_dims(query, axis=0))
    q_rot= np.squeeze(q_rot)

    random.shuffle(dict_value["positives"])
    pos_files=[]
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    #positives= load_pc_files(dict_value["positives"][0:num_pos])
    positives=load_pc_files(pos_files, dataset_folder) # le
    p_rot= rotate_point_cloud(positives)

    neg_files=[]
    neg_indices=[]
    if (len(hard_neg)==0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j=0
        while(len(neg_files)<num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j+=1	
    negatives=load_pc_files(neg_files, dataset_folder)  # le
    n_rot=rotate_point_cloud(negatives)

    if(other_neg==False):
        return [q_rot,p_rot,n_rot]

    #For Quadruplet Loss
    else:
        #get neighbors of negatives and query
        neighbors=[]
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs= list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)

        if(len(possible_negs)==0):
            return [q_jit, p_jit, n_jit, np.array([])]

        neg2= load_pc_file(QUERY_DICT[possible_negs[0]]["query"], dataset_folder) # le
        n2_rot= rotate_point_cloud(np.expand_dims(neg2, axis=0))
        n2_rot= np.squeeze(n2_rot)

        return [q_rot,p_rot,n_rot,n2_rot]

def get_jittered_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False, dataset_folder=None):
    query=load_pc_file(dict_value["query"], dataset_folder) #Nx3 le
    #q_rot= rotate_point_cloud(np.expand_dims(query, axis=0))
    q_jit= jitter_point_cloud(np.expand_dims(query, axis=0))
    q_jit= np.squeeze(q_jit)

    random.shuffle(dict_value["positives"])
    pos_files=[]
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    #positives= load_pc_files(dict_value["positives"][0:num_pos])
    positives=load_pc_files(pos_files, dataset_folder) # le
    p_jit= jitter_point_cloud(positives)

    neg_files=[]
    neg_indices=[]
    if(len(hard_neg)==0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j=0
        while(len(neg_files)<num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j+=1	
    negatives=load_pc_files(neg_files, dataset_folder) # le
    n_jit=jitter_point_cloud(negatives)

    if(other_neg==False):
        return [q_jit,p_jit,n_jit]
    #For Quadruplet Loss
    else:
        #get neighbors of negatives and query
        neighbors=[]
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs= list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)

        if(len(possible_negs)==0):
            return [q_jit, p_jit, n_jit, np.array([])]

        neg2= load_pc_file(QUERY_DICT[possible_negs[0]]["query"], dataset_folder) # le
        n2_jit= jitter_point_cloud(np.expand_dims(neg2, axis=0))
        n2_jit= np.squeeze(n2_jit)

        return [q_jit,p_jit,n_jit,n2_jit]
