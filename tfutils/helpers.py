# -*- coding: utf-8 -*-
"""
Tensorflow helper functionality 
Partially from: https://github.com/openai/iaf/blob/master/tf_utils/common.py
Copyright (c) 2016 openai
"""
import tensorflow as tf
import numpy as np

def anneal_linear(t, n, e_final,e_init=1):
    ''' Linear anneals between e_init and e_final '''
    if t >= n:
        return e_final
    else:
        return e_init - ( (t/n) * (e_init - e_final) )
        
class HParams(object):
    ''' Hyperparameter object, from https://github.com/openai/iaf/blob/master/tf_utils '''

    def __init__(self, **kwargs):
        self._items = {}
        for k, v in kwargs.items():
            self._set(k, v)

    def _set(self, k, v):
        self._items[k] = v
        setattr(self, k, v)

    def parse(self, str_value):
        hps = HParams(**self._items)
        for entry in str_value.strip().split(","):
            entry = entry.strip()
            if not entry:
                continue
            key, sep, value = entry.partition("=")
            if not sep:
                raise ValueError("Unable to parse: %s" % entry)
            default_value = hps._items[key]
            if isinstance(default_value, bool):
                hps._set(key, value.lower() == "true")
            elif isinstance(default_value, int):
                hps._set(key, int(value))
            elif isinstance(default_value, float):
                hps._set(key, float(value))
            elif isinstance(default_value, list):
                value = value.split('-')
                default_inlist = hps._items[key][0]
                if key == 'seq1':
                    default_inlist = hps._items[hps._items['item1']]
                if key == 'seq2':
                   default_inlist = hps._items[hps._items['item2']]
                if isinstance(default_inlist, bool):
                    hps._set(key, [i.lower() == "true" for i in value])
                elif isinstance(default_inlist, int):
                    hps._set(key, [int(i) for i in value])
                elif isinstance(default_inlist, float):
                    hps._set(key, [float(i) for i in value])
                else:
                    hps._set(key,value) # string
            else:
                hps._set(key, value)
        return hps

def split(x, split_dim, split_sizes):
    n = len(list(x.get_shape()))
    dim_size = np.sum(split_sizes)
    assert int(x.get_shape()[split_dim]) == dim_size
    ids = np.cumsum([0] + split_sizes)
    ids[-1] = -1
    begin_ids = ids[:-1]

    ret = []
    for i in range(len(split_sizes)):
        cur_begin = np.zeros([n], dtype=np.int32)
        cur_begin[split_dim] = begin_ids[i]
        cur_end = np.zeros([n], dtype=np.int32) - 1
        cur_end[split_dim] = split_sizes[i]
        ret += [tf.slice(x, cur_begin, cur_end)]
    return ret

def repeat_v2(x,k):
    ''' repeat k times along first dimension '''
    def change(x,k):    
        shape = x.get_shape().as_list()[1:]
        x_1 = tf.expand_dims(x,1)
        tile_shape = tf.concat([tf.ones(1,dtype='int32'),[k],tf.ones([tf.rank(x)-1],dtype='int32')],axis=0)
        x_rep = tf.tile(x_1,tile_shape)
        new_shape = np.insert(shape,0,-1)
        x_out = tf.reshape(x_rep,new_shape)    
        return x_out
        
    return tf.cond(tf.equal(k,1),
                   lambda: x,
                   lambda: change(x,k))    
