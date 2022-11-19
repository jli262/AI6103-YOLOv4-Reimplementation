# coding:utf-8
import json
import numpy as np
import matplotlib.pyplot as plt
from .log_utils import LossLogger


def plot_loss(loss_file_path):
    """ 
        Para: loss file path
    """
    logg = LossLogger(loss_file_path)
    epoch_all = np.arange(1, len(logg.losses)+1)

    plot_image, plots = plt.subplots(1, 1, num='Loss Curve')
    plots.plot(epoch_all, logg.losses)
    plots.set(xlabel='epoch', title='Total Loss', ylabel="loss")

    return plot_image, plots


def plot_PR(pr_file_path, classes_name):
    """ 
        Para: PR file path, class name
    """
    with open(pr_file_path, encoding='utf-8') as file:
        results = json.load(file)[classes_name]

    plot_image, plots = plt.subplots(1, 1, num='PR Curve')
    plots.plot(results['recall'], results['precision'])
    plots.set(xlabel='recall', ylabel='precision', title='PR curve')
    return plot_image, plots


def plot_AP(file_path: str):
    """ 
        Para: ap file path
    """
    with open(file_path, encoding='utf-8') as file:
        results = json.load(file)

    AP_list = []
    class_list = []
    for i, j in results.items():
        AP_list.append(j['AP'])
        class_list.append(i)

    plot_image, plots = plt.subplots(1, 1, num='AP Column')
    plots.barh(range(len(AP_list)), AP_list, height=0.6, tick_label=class_list)
    plots.set(xlabel='AP')

    return plot_image, plots
