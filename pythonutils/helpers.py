# -*- coding: utf-8 -*-
"""
Python helpers
@author: thomas
"""
import os
import matplotlib.pyplot as plt

def save(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.
    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.
    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    plt.savefig(savepath)
    
    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")
    
def nested_list(n1,n2,n3):
    results=[]
    for i in range(n1):
        results.append([])
        for j in range(n2):
            results[-1].append([])
            for k in range(n3):
                results[-1][-1].append([])
    return results
    
def make_name(hps):
    ''' structures output folders based on hps '''
    name = ''
    if hasattr(hps,'artificial_data'):
        if not hps.artificial_data:
            if hps.use_target_net:
                name += 'RL_target_network/'
            else:
                name += 'RL/'
        else:
            name +='decorrelated/'
    if hps.loop_hyper:
        name += 'hyper_{}_{}/'.format(hps.item1,hps.item2)
    elif hps.network == 2:
        if hps.deterministic:
            name += 'deterministic'
        else:
            name += 'mlp_z{}'.format(hps.z_size)
    elif len(hps.var_type) > 1:
        name += '{}_z{}_nf{}_n{}_K{}'.format('_'.join(hps.var_type),hps.z_size,hps.n_flow,hps.N,hps.K)
    elif hps.var_type == ['discrete']:
        name += '{}_n{}_K{}'.format(hps.var_type[0],hps.N,hps.K)
    elif hps.var_type == ['continuous']:
        name += '{}_z{}_nf{}{}'.format(hps.var_type[0],hps.z_size,hps.n_flow,'ar' if hps.ar else '')
    return name