# coding:utf-8
import json
import numpy as np
import matplotlib.pyplot as plt
from .cgd_log_utils import LossLogger


def plot_loss(loss_file_path):
    epoch_all, logg = lossFile(loss_file_path)
    plot_image, plots = ploting(epoch_all, logg)

    return plot_image, plots


def ploting(epoch_all, logg):
    plot_image, plots = plt.subplots(1, 1, num='Loss Curve')
    plots.set(xlabel='epoch', title='Total Loss', ylabel="loss")
    plots.plot(epoch_all, logg.losses)
    return plot_image, plots


def lossFile(loss_file_path):
    logg = LossLogger(loss_file_path)
    epoch_all = np.arange(1, len(logg.losses) + 1)
    return epoch_all, logg


def plot_PR(pr_file_path, classes_name):
    results = openPRfile(classes_name, pr_file_path)

    plot_image, plots = plottingP(results)
    return plot_image, plots


def plottingP(results):
    plot_image, plots = plt.subplots(1, 1, num='PR Curve')
    plots.set(xlabel='recall', ylabel='precision', title='PR curve')
    plots.plot(results['recall'], results['precision'])
    return plot_image, plots


def openPRfile(classes_name, pr_file_path):
    with open(pr_file_path, encoding='utf-8') as file:
        results = json.load(file)[classes_name]
    return results


def plot_AP(file_path: str):
    results = openfile(file_path)
    AP_list = []
    class_list = []
    append(AP_list, class_list, results)

    plot_image, plots = plotingAP(AP_list, class_list)

    return plot_image, plots


def plotingAP(AP_list, class_list):
    plot_image, plots = plt.subplots(1, 1, num='AP Column')
    plots.set(xlabel='AP')
    plots.barh(range(len(AP_list)), AP_list, height=0.6, tick_label=class_list)
    return plot_image, plots


def append(AP_list, class_list, results):
    for i, j in results.items():
        AP_list.append(j['AP'])
        class_list.append(i)


def openfile(file_path):
    with open(file_path, encoding='utf-8') as file:
        results = json.load(file)
    return results
