# -*- coding: utf-8 -*-
"""
Distribution functions 
From IAF (OpenAI github, https://github.com/openai/iaf/blob/master/tf_utils) and Gumbell-softmax (Jiang github)
"""
import numpy as np
import tensorflow as tf

##### Probabilistic functions (partially from OpenAI github)
class DiagonalGaussian(object):
    def __init__(self, mean, logvar, sample=None):
        self.mean = mean
        self.logvar = logvar

        if sample is None:
            noise = tf.random_normal(tf.shape(mean))
            sample = mean + tf.exp(0.5 * logvar) * noise
        self.sample = sample

    def logps(self, sample):
        return gaussian_diag_logps(self.mean, self.logvar, sample)

def gaussian_diag_logps(mean, logvar, sample=None):
    if sample is None:
        noise = tf.random_normal(tf.shape(mean))
        sample = mean + tf.exp(0.5 * logvar) * noise
    return tf.clip_by_value(-0.5 * (np.log(2 * np.pi) + logvar + tf.square(sample - mean) / tf.exp(logvar)),-(10e10),10e10)

def discretized_logistic(mean, logscale=None, binsize=1/6.0, sample=None):
    if logscale is None:
        logscale = tf.zeros(tf.shape(mean))
    logscale = tf.clip_by_value(logscale,-10,10)
    scale = tf.exp(logscale)
    x = (sample - mean) * (1/binsize) # stretch back
    logp = tf.log(tf.sigmoid((x + 0.5)/scale) - tf.sigmoid((x - 0.5)/scale) + 1e-20)
    return logp 

def logsumexp(x):
    x_max = tf.reduce_max(x, [1], keep_dims=True)
    return tf.reshape(x_max, [-1]) + tf.log(tf.reduce_sum(tf.exp(x - x_max), [1]))

def compute_lowerbound(log_pxz, sum_kl_costs, k=1, alpha=0.5):
    if k == 1:
        return sum_kl_costs - log_pxz
    # log 1/k \sum p(x | z) * p(z) / q(z | x) = -log(k) + logsumexp(log p(x|z) + log p(z) - log q(z|x))
    log_pxz = tf.reshape(log_pxz, [-1, k])
    sum_kl_costs = tf.reshape(sum_kl_costs, [-1, k])
    diff = (log_pxz - sum_kl_costs)*(1-alpha)    
    elbo = tf.reduce_mean(- tf.log(float(k)) + logsumexp(diff)) / (1-alpha)
    return -elbo
    
    
###### Gumbell-sotfmax ####### (from E. Jiang)
def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = tf.add(logits,sample_gumbel(tf.shape(logits)))
  return tf.nn.softmax( tf.div(y, temperature))

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  #if hard:
  #  k = tf.shape(logits)[-1]
  #  #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
  #  y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
  #  y = tf.stop_gradient(y_hard - y) + y
  return y

