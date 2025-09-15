from django.db import models
from django.contrib.auth.models import User
# Create your models here.
from datetime import datetime
import pandas as pd
import pickle
from .Algorithms.algorithm import *


class Experiments(models.Model):
    EXPERIMENT_ID = models.AutoField(primary_key=True)
    USER_ID = models.ForeignKey(User, on_delete=models.CASCADE)
    RAW_DF = models.BinaryField()
    NAME = models.CharField(max_length=255)
    LAST_EDIT = models.DateTimeField(default=datetime.now)


class Blocks(models.Model):
    BLOCKS_ID = models.AutoField(primary_key=True)
    EXPERIMENTS_ID = models.ForeignKey(Experiments, on_delete=models.CASCADE)
    X_TRANS = models.CharField(max_length=255, default='x')
    Y_TRANS = models.CharField(max_length=255, default='y')
    ER_MODE = models.CharField(max_length=255, default='range')
    DEGREE = models.FloatField(default=0)
    L_X_LIM = models.FloatField(null=True, default=None)
    R_X_LIM = models.FloatField(null=True, default=None)
    L_Y_LIM = models.FloatField(null=True, default=None)
    R_Y_LIM = models.FloatField(null=True, default=None)
    FIT_TYPE = models.CharField(max_length=255, default='auto')
    X_MAJOR_TICK = models.FloatField(null=True, default=None)
    Y_MAJOR_TICK = models.FloatField(null=True, default=None)
    X_MINOR_TICK = models.FloatField(null=True, default=None)
    Y_MINOR_TICK = models.FloatField(null=True, default=None)
    X_ERR = models.FloatField(null=True)
    Y_ERR = models.FloatField(null=True)

    def process(self):
        """
        :param attributes: the new attributes that needs processing, sort in order:
        X_TRANS
        Y_TRANS
        ER_MODE
        DEGREE
        L_X_LIM
        R_X_LIM
        L_Y_LIM
        R_Y_LIM
        FIT_TYPE
        X_MAJOR_TICK
        Y_MAJOR_TICK
        X_MINOR_TICK
        Y_MINOR_TICK
        :return: Process the block
        """
        # blocks_id_copy = self.BLOCKS_ID
        # experiments_id_copy = self.EXPERIMENTS_ID
        # x_trans_copy = self.X_TRANS
        # y_trans_copy = self.Y_TRANS
        # er_mode_copy = self.ER_MODE
        # degree_copy = self.DEGREE
        # l_x_lim_copy = self.L_X_LIM
        # r_x_lim_copy = self.R_X_LIM
        # l_y_lim_copy = self.L_Y_LIM
        # r_y_lim_copy = self.R_Y_LIM
        # fit_type_copy = self.FIT_TYPE
        # x_major_tick_copy = self.X_MAJOR_TICK
        # y_major_tick_copy = self.Y_MAJOR_TICK
        # x_minor_tick_copy = self.X_MINOR_TICK
        # y_minor_tick_copy = self.Y_MINOR_TICK
        # x_err_copy = self.X_ERR
        # y_err_copy = self.Y_ERR

        experiment = self.EXPERIMENTS_ID
        df = pickle.loads(experiment.RAW_DF)
        df = pd.DataFrame(df)
        xerr, yerr, xv, yv = decode_input(df)
        y_bar = yv.mean(axis=1)

        if self.ER_MODE == "Range":
            yerr = cal_error(yv)
        else:
            yerr = cal_error(yv, "sd")

        X, Y, xerr, yerr = transform(xv, y_bar, xerr, yerr, self.X_TRANS, self.Y_TRANS)
        print('in processing, xerr and yerr')
        print(xerr)
        print(yerr)
        print('------')
        self.X_ERR = xerr.mean().round(1)
        self.Y_ERR = yerr.mean().round(1)
        print('self.XERR and YERR')
        print(self.X_ERR)
        print(self.Y_ERR)
        print('------')
        print(self.FIT_TYPE, self.DEGREE)
        result = raw_fit(X, Y, self.FIT_TYPE, self.DEGREE)
        print(result)
        f, relationship, fit_type, deg = result
        plot = plot_graph(X, Y, f, relationship,
                          fit_type, deg, xerr, yerr,
                          [self.L_X_LIM, self.R_X_LIM],
                          [self.L_Y_LIM, self.R_Y_LIM],
                          [self.X_MAJOR_TICK, self.X_MINOR_TICK],
                          [self.Y_MAJOR_TICK, self.Y_MINOR_TICK])

        plot = pickle.loads(plot)
        name = str(self.BLOCKS_ID) + '.png'
        save_path = 'my_app/static/PLOTS/' + name
        print(save_path)
        # import os
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        plot.savefig(save_path)
