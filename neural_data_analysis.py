import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import defaultdict

import tools
from network import Model

from std.train_std import do_eval
from std.task_std import rule_name, rules_dict, generate_trials


def make_raw_data(model_dir, task, root_dir='result', **kwargs):
    """
    Args:
        model_dir: directory of the model
        task: task name in ruleset
        root_dir: root directory
        **kwargs:
            std: standard deviation of the input stimulus
    """
    save_dir = os.path.join(root_dir, rule_name[task])
    if len(kwargs.keys()):
        kwargs_list = ['{0}: {1}'.format(key, value) for key, value in kwargs.items()]
        kwargs_list.sort()
        tmp = '\n'.join(kwargs_list)
        save_dir = os.path.join(save_dir, tmp)
    model_name = model_dir.split('/')[-1]
    save_dir = os.path.join(save_dir, model_name)

    if not (os.path.isdir(save_dir)):
        os.makedirs(os.path.join(save_dir))
    elif len(os.listdir(save_dir)) != 0:
        print('Already done')
        return 0

    model = Model(model_dir)
    hp = model.hp

    with tf.Session() as sess:
        model.restore(model_dir)
        trial = generate_trials(task, hp, mode='test', noise_on=False, **kwargs)
        feed_dict = {model.x: trial.x,
                     model.y: trial.y,
                     model.c_mask: trial.c_mask}
        input, target = trial.x, trial.y
        output, hidden = sess.run([model.y_hat, model.h], feed_dict=feed_dict)

        h_fix_off = hidden[trial.epochs['fix1'][1]:, :, :]
        task_variance = h_fix_off.var(axis=1).mean(axis=0)

        save_list = ['input', 'target', 'output', 'hidden', 'task_variance']
        for file in save_list:
            fname = os.path.join(save_dir, file)
            with open(fname, 'wb') as f:
                pickle.dump(locals()[file], f)
                print(file, 'Saved')


def load_raw_data(model_dir, task, type, root_dir='result', **kwargs):
    """
    Args:
        model_dir: directory of the model, string type path
        task: task name in ruleset, string type
        type: type of data, string type, 'input', output', 'target', 'hidden', 'task_variance', 'title'
        root_dir: root directory
        **kwargs:
                std: standard deviation of the input stimulus

    Returns: title or data of the selected type
    """
    saved_dir = os.path.join(root_dir, rule_name[task])
    if len(kwargs.keys()):
        kwargs_list = ['{0}: {1}'.format(key, value) for key, value in kwargs.items()]
        kwargs_list.sort()
        tmp = '\n'.join(kwargs_list)
        saved_dir = os.path.join(saved_dir, tmp)
    model_name = model_dir.split('/')[-1]
    saved_dir = os.path.join(saved_dir, model_name)
    if type == 'title':
        title = saved_dir.replace('/', '  ')
        title = title.replace('test', '')
        title = title[:-2] + '  model' + title[-1]
        return title
    file = os.path.join(saved_dir, type)
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def load_dataset(model_dir, task, root_dir='result', **kwargs):
    input = load_raw_data(model_dir, task, 'input', root_dir=root_dir, **kwargs)
    target = load_raw_data(model_dir, task, 'target', root_dir=root_dir, **kwargs)
    output = load_raw_data(model_dir, task, 'output', root_dir=root_dir, **kwargs)
    hidden = load_raw_data(model_dir, task, 'hidden', root_dir=root_dir, **kwargs)
    task_variance = load_raw_data(model_dir, task, 'task_variance', root_dir=root_dir, **kwargs)
    title = load_raw_data(model_dir, task, 'title', root_dir=root_dir, **kwargs)

    return input, target, output, hidden, task_variance, title


def array_plot(raw_data,
               task_variance=None,
               disp_num=None,
               batch_num=0,
               vline=None,
               zoom=None,
               fixtime=None,
               ylim=None,
               title=None,
               std=None,
               save=False):

    # array: Unit x Time
    if raw_data.ndim == 3:
        array = raw_data[:, batch_num, :].T
    elif raw_data.ndim == 2:
        array = raw_data
    else:
        raise ValueError('Input array dimension error')

    unit, time = array.shape
    if disp_num is None:
        disp_num = unit

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_axes([0.15, 0.15, 0.73, 0.73])
    # x = np.arange(time) * 0.02
    x = np.arange(time)
    colors = [plt.cm.jet(c) for c in np.linspace(0, 1.0, unit)]

    if task_variance is not None:
        unit_order = task_variance.argsort()[::-1]
    else:
        unit_order = np.arange(unit)

    for i, u in enumerate(unit_order):
        if i == disp_num:
            break
        ax.plot(x, array[u, :], lw=0.5, color=colors[u], label=u)

    if ylim is None:
        plt.ylim(np.array([-0.1, 1.2 * np.max(array)]))
    else:
        plt.ylim(np.array([-0.1, ylim]))

    if vline is not None:
        ax.axvline(x=vline, lw=1, c='k')

    if zoom and fixtime is not None:
        plt.xlim(fixtime - 0.02 * zoom, fixtime + 0.02 * zoom)
        # plt.xlim(fixtime - zoom, fixtime + zoom)

    if title is not None:
        plt.title(title)

    fig.tight_layout()
    if save:
        plt.savefig('FIg/' + str(std) + '.png', transparent=True)
    plt.show()


def fix_time(output, threshold, batch_num=0):
    # Range of  threshold: (0.05, 0.85)
    fix_output = output[:, batch_num, 0]
    time = np.where(fix_output < threshold)[0][0] * 0.02
    return time


def model_hp_compare(model_dir1, model_dir2):
    hp1 = tools.load_hp(model_dir1)
    hp2 = tools.load_hp(model_dir2)

    same_list = list()
    different_list = list()
    only1_list = list()

    for key, value in hp1.items():
        if key in hp2.keys():
            if value == hp2[key]:
                same_list.append(key)
            else:
                different_list.append(key)
        else:
            only1_list.append(key)

    only2_list = hp2.keys() - hp1.keys()

    print('- Same hp')
    for key in same_list:
        print(key, ":", hp1[key])
    print('\n- Different hp')
    for key in different_list:
        print(key, "hp1 :", hp1[key])
        print(key, "hp2 :", hp2[key])

    print('\n- Only hp1')
    for key in only1_list:
        print(key)
    print('\n- Only hp2')
    for key in only2_list:
        print(key, ":", hp2[key])


def model_test(model_dir, rule_set=None, std=1.5):
    hp = tools.load_hp(model_dir)
    model = Model(model_dir, hp=hp)

    with tf.Session() as sess:
        model.restore(model_dir)
        if rule_set is not None:
            model.hp['rules'] = rule_set
        log = defaultdict(list)
        log['model_dir'] = model_dir
        log['trials'] = [0]
        log['times'] = [0]
        do_eval(sess, model, log, hp['rules'], std=std)


if __name__ == '__main__':
    root_dir = 'test'
    model_dir = './debug/test2'

    for task in rules_dict['all']:
        make_raw_data(model_dir, task, root_dir=root_dir)
    for std in np.arange(1, 3, 0.5):
        make_raw_data(model_dir, 'reactgo_v2', root_dir=root_dir, std=std)

    task = 'reactgo_v2'
    std = 1.5

    input = load_raw_data(model_dir, task, 'input', root_dir=root_dir, std=std)
    target = load_raw_data(model_dir, task, 'target', root_dir=root_dir, std=std)
    output = load_raw_data(model_dir, task, 'output', root_dir=root_dir, std=std)
    hidden = load_raw_data(model_dir, task, 'hidden', root_dir=root_dir, std=std)
    tv = load_raw_data(model_dir, task, 'task_variance', root_dir=root_dir, std=std)

    array_plot(output)
    array_plot(output, zoom=15, fixtime=1)

    print("Done")







