#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import argparse
import os
 
parser = argparse.ArgumentParser(description='')
 
# .../xxx.ckpt
parser.add_argument("--old_ckpt", required=True, help="old ckpt name")  # old ckpt path
parser.add_argument("--new_path", required=True, help="path to ckpt")  # old ckpt path
parser.add_argument("--prefix", default='teacher', help="prefix for addition")                          # new prefix
 
args = parser.parse_args()
 

def transfer_teacher():
    if not os.path.exists(args.old_ckpt+'.meta'):
        print("ckpt: {} not exist!".format(args.old_ckpt))
        exit()
    if not os.path.exists(args.new_path):
        print("new_path is not exist!")
        os.makedirs(args.new_path)
        print("make new_path")
        #exit()
    with tf.Session() as sess:
        new_var_list = []
        for var_name, _ in tf.contrib.framework.list_variables(args.old_ckpt):
            var = tf.contrib.framework.load_variable(args.old_ckpt, var_name)
 
            new_name = var_name
            new_name = new_name.replace("query_triplets", args.prefix+"/query_triplets")
            if new_name in ["Variable", "beta1_power", "beta2_power"]:
                new_name = args.prefix + '/' + new_name
            #new_name = args.prefix + '/' + new_name   # add new prefix
  
            print('Renaming %s \n   ==> %s.' % (var_name, new_name))
            renamed_var = tf.Variable(var, name=new_name)
            new_var_list.append(renamed_var)
 
        print('starting to write new checkpoint !')
        saver = tf.train.Saver(var_list=new_var_list)   # create a new saver
        sess.run(tf.global_variables_initializer())     # initial variable, very important!!!
        model_name = args.prefix + '_' + args.old_ckpt.split('/')[-1]       # new_name
        print('model_name: {}'.format(model_name))
        #exit()
        new_ckpt = os.path.join(args.new_path, model_name)   # create new_ckpt path
        print('new_ckpt: {}'.format(new_ckpt))
        saver.save(sess, new_ckpt)   # save new ckpt
        print("done !")

def transfer_student():
    if not os.path.exists(args.old_ckpt+'.meta'):
        print("ckpt: {} not exist!".format(args.old_ckpt))
        exit()
    if not os.path.exists(args.new_path):
        print("new_path is not exist!")
        os.makedirs(args.new_path)
        print("make new_path")
        #exit()
    with tf.Session() as sess:
        new_var_list = []
        for var_name, _ in tf.contrib.framework.list_variables(args.old_ckpt):
            var = tf.contrib.framework.load_variable(args.old_ckpt, var_name)
            if "VLAD" in var_name: continue
            new_name = var_name
            new_name = new_name.replace("query_triplets", args.prefix+"/query_triplets")
            new_name = new_name.replace("fastdgcnn", "BACKBONE")
            if new_name in ["Variable", "beta1_power", "beta2_power"]:
                new_name = args.prefix + '/' + new_name
            #new_name = args.prefix + '/' + new_name   # add new prefix
  
            print('Renaming %s \n   ==> %s.' % (var_name, new_name))
            renamed_var = tf.Variable(var, name=new_name)
            new_var_list.append(renamed_var)
 
        print('starting to write new checkpoint !')
        saver = tf.train.Saver(var_list=new_var_list)   # create a new saver
        sess.run(tf.global_variables_initializer())     # initial variable, very important!!!
        model_name = args.prefix + '_' + args.old_ckpt.split('/')[-1]       # new_name
        print('model_name: {}'.format(model_name))
        #exit()
        new_ckpt = os.path.join(args.new_path, model_name)   # create new_ckpt path
        print('new_ckpt: {}'.format(new_ckpt))
        saver.save(sess, new_ckpt)   # save new ckpt
        print("done !")

def transfer_fastdgcnn():
    if not os.path.exists(args.old_ckpt+'.meta'):
        print("ckpt: {} not exist!".format(args.old_ckpt))
        exit()
    if not os.path.exists(args.new_path):
        print("new_path is not exist!")
        os.makedirs(args.new_path)
        print("make new_path")
        #exit()
    with tf.Session() as sess:
        new_var_list = []
        for var_name, _ in tf.contrib.framework.list_variables(args.old_ckpt):
            var = tf.contrib.framework.load_variable(args.old_ckpt, var_name)
            if "VLAD" in var_name: continue
            new_name = var_name
            #new_name = new_name.replace("query_triplets", args.prefix+"/query_triplets")
            #if new_name in ["Variable", "beta1_power", "beta2_power"]:
            #    new_name = args.prefix + '/' + new_name
            #new_name = args.prefix + '/' + new_name   # add new prefix
  
            print('Renaming %s \n   ==> %s' % (var_name, new_name))
            renamed_var = tf.Variable(var, name=new_name)
            new_var_list.append(renamed_var)
 
        print('starting to write new checkpoint !')
        saver = tf.train.Saver(var_list=new_var_list)   # create a new saver
        sess.run(tf.global_variables_initializer())     # initial variable, very important!!!
        model_name = args.prefix + '_' + args.old_ckpt.split('/')[-1]       # new_name
        print('model_name: {}'.format(model_name))
        #exit()
        new_ckpt = os.path.join(args.new_path, model_name)   # create new_ckpt path
        print('new_ckpt: {}'.format(new_ckpt))
        saver.save(sess, new_ckpt)   # save new ckpt
        print("done !")

if __name__ == '__main__':
    if args.prefix == "teacher":
        transfer_teacher()
    elif args.prefix == "student":
        transfer_student()
    elif args.prefix == "fastdgcnn":
        transfer_fastdgcnn()
    else:
        print("prefix error! prefix only supports ['teacher', 'studnet']")

