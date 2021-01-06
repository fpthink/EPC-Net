import numpy as np
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def count_files():
    root = '/test/dataset/benchmark_datasets/oxford'
    files = os.listdir(root)
    cnt = 0
    for i in range(len(files)):
        data_path = os.path.join(root, files[i], 'pointcloud_20m_10overlap')
        cnt += len(os.listdir(data_path))
    print('data files: {}'.format(cnt))
    return cnt
