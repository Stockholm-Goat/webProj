# Create your views here.
import base64

import sympy
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.http import JsonResponse
from .Algorithms.algorithm import *
from django.urls import reverse
import pandas as pd
from my_app.models import Experiments, Blocks
import pickle
from django.contrib.auth.decorators import login_required
from django.utils.datastructures import MultiValueDictKeyError
from django.contrib import messages


def main_page(request):
    return render(request, "default_mainpage.html")


def log_in(request):
    find_user = True
    if request.method == "POST":
        username1 = request.POST.get('username')
        password1 = request.POST.get('password')
        remember_me = request.POST.get('remember_me')

        user = authenticate(request, username=username1, password=password1)
        if user is not None:
            login(request, user)
            if not remember_me:
                request.session.set_expiry(0)
            return redirect('/home')
        else:
            find_user = False
            return render(request, "login.html", {'find_user': find_user})
    return render(request, 'login.html', {'find_user': find_user})


@login_required
def home(request):
    print('access home')
    print(request.method)
    user = request.user
    print(user.username)
    experiments = Experiments.objects.filter(USER_ID=user)
    print(user.username)

    if request.method == 'POST':
        print('In POST!')
        if request.POST.get('delete') == '1':  # if you want to delete
            ex_id = request.POST.get('experiment_id')
            print(request.POST)

            ex = Experiments.objects.get(EXPERIMENT_ID=ex_id)
            import os
            blocks = Blocks.objects.filter(EXPERIMENTS_ID=ex)
            for block in blocks:
                path = 'my_app/static/PLOTS/' + str(block.BLOCKS_ID) + '.png'

                if os.path.isfile(path):
                    os.remove(path)
                else:
                    print('nofile')

            ex.delete()
            url = reverse('home')
            return redirect(url)
    print('line before render')
    return render(request, 'home.html', {'username': user.username, 'experiments': experiments})


def register(request):
    success = True  # if can create
    if request.method == 'POST':
        username1 = request.POST['username']
        password1 = request.POST['password']
        auth_password = request.POST['auth_password']
        digit = False  # password only digit
        short = False  # password too short
        if password1.isdigit():
            digit = True
        if len(password1) < 8:
            short = True
        if short or digit:
            return render(request, 'register.html', {'not_long': short, 'no_letters': digit})  # return to webpage
        if password1 != auth_password:
            txt = "Passwords do not match"
            success = False # check password failed
            return render(request, "register.html", {'txt': txt, 'success': success})
        Query = User.objects.filter(username=username1).exists()
        if Query:
            txt = "User already exists"
            success = False
            return render(request, 'register.html', {'txt': txt, 'success': success})
        if success:
            NewUser = User.objects.create_user(username=username1, email='', password=password1)
            login(request, NewUser)
            return redirect('/home')
    return render(request, 'register.html')


@login_required
def new_experiment(request):
    error = False
    checked = False
    if request.method == 'POST':
        try:
            file = request.FILES['csv']
        except MultiValueDictKeyError:
            error = True
            return render(request, 'new_experiments.html', {'error': error})
        name = request.POST['experiment_name']
        if len(name) == 0:
            error = True
            return render(request, 'new_experiments.html', {'error': error})

        user = request.user
        df = pd.read_csv(file)
        from datetime import datetime
        serialisedDF = pickle.dumps(df)
        newExperiment = Experiments(USER_ID=user, RAW_DF=serialisedDF, LAST_EDIT=datetime.now(), NAME=name)
        newExperiment.save()
        url = reverse('experiment_page', args=[newExperiment.EXPERIMENT_ID])
        print("url=", url)
        return redirect(url)
    return render(request, 'new_experiments.html', {'error': error})


@login_required
def experiment_page(request, experiment_id):
    # retrieve the experiment from the database
    experiment = Experiments.objects.get(EXPERIMENT_ID=experiment_id)
    print('accessed in experiment page')
    print(request.POST)
    # Create new block
    if request.method == "POST":
        print(request.POST)
        if request.POST.get('redirect') == '1':
            print('in redirect')
            # url = reverse('home')
            # print(url)
            return redirect('/home')
        if request.POST.get('create') == '1':
            print("accessed create")
            new_block = Blocks(EXPERIMENTS_ID=experiment)
            new_block.save()
            new_block.process()
            url = reverse('experiment_page', args=[experiment.EXPERIMENT_ID])  # go to GET
            return redirect(url)
        if request.POST.get('delete') == '1':
            print('delete' + str(request.POST['block_id']))
            delete_block = Blocks(BLOCKS_ID=request.POST['block_id'])
            delete_block.delete()
            import os
            path = 'my_app/static/PLOTS/' + str(request.POST['block_id']) + '.png'

            if os.path.isfile(path):
                os.remove(path)
            else:
                print('nofile')  # usually does not happen...
            url = reverse('experiment_page', args=[experiment.EXPERIMENT_ID])
            return redirect(url)
        else:
            print('hi')  # usually does not happen...
            # block, created = Blocks.objects.get_or_create(id=block_id, experiment=experiment)
            # block.process()
            # block.save()
            # return redirect('experiment_page', experiment_id=experiment_id)
    df = pickle.loads(experiment.RAW_DF)
    df = pd.DataFrame(df)
    df_html = df.to_html(border=1, classes=['table', 'table-bordered', 'table-striped', 'mystyle'])
    blocks = Blocks.objects.filter(EXPERIMENTS_ID=experiment_id)
    return render(request, 'experiment_page.html', {'df_html': df_html, 'blocks': blocks})


def process_block(request, block_id, ex_id):
    if request.method == 'POST':
        print(request.POST)
        print(type(request.POST.get('DEGREE_' + str(block_id))))
        block = Blocks.objects.get(BLOCKS_ID=block_id)

        x_trans_copy = block.X_TRANS
        y_trans_copy = block.Y_TRANS
        er_mode_copy = block.ER_MODE
        degree_copy = block.DEGREE
        l_x_lim_copy = block.L_X_LIM
        r_x_lim_copy = block.R_X_LIM
        l_y_lim_copy = block.L_Y_LIM
        r_y_lim_copy = block.R_Y_LIM
        fit_type_copy = block.FIT_TYPE
        x_major_tick_copy = block.X_MAJOR_TICK
        y_major_tick_copy = block.Y_MAJOR_TICK
        x_minor_tick_copy = block.X_MINOR_TICK
        y_minor_tick_copy = block.Y_MINOR_TICK
        x_err_copy = block.X_ERR
        y_err_copy = block.Y_ERR

        # process any changes in parameters
        er_mode_field = 'ER_MODE_' + str(block_id)
        er_mode_value = request.POST.get(er_mode_field)
        if er_mode_value:
            er_mode = er_mode_value
        else:
            er_mode = block.ER_MODE

        degree_field = 'DEGREE_' + str(block_id)
        print(degree_field)
        degree_value = request.POST.get(degree_field)
        if degree_value:
            degree = float(degree_value)
        else:
            degree = 0

        l_x_lim_field = 'X_LIM_L_' + str(block_id)
        l_x_lim_value = request.POST.get(l_x_lim_field)
        if l_x_lim_value:
            l_x_lim = float(l_x_lim_value)
        else:
            l_x_lim = None

        r_x_lim_field = 'X_LIM_R_' + str(block_id)
        r_x_lim_value = request.POST.get(r_x_lim_field)
        if r_x_lim_value:
            r_x_lim = float(r_x_lim_value)
        else:
            r_x_lim = None

        l_y_lim_field = 'Y_LIM_L_' + str(block_id)
        l_y_lim_value = request.POST.get(l_y_lim_field)
        if l_y_lim_value:
            l_y_lim = float(l_y_lim_value)
        else:
            l_y_lim = None

        r_y_lim_field = 'Y_LIM_R_' + str(block_id)
        r_y_lim_value = request.POST.get(r_y_lim_field)
        if r_y_lim_value:
            r_y_lim = float(r_y_lim_value)
        else:
            r_y_lim = None

        fit_type_field = 'fit_type_' + str(block_id)
        fit_type_value = request.POST.get(fit_type_field)
        if fit_type_value:
            fit_type = fit_type_value
        else:
            fit_type = block.FIT_TYPE

        x_major_tick_field = 'X_MAJOR_TICK_' + str(block_id)
        x_major_tick_value = request.POST.get(x_major_tick_field)
        if x_major_tick_value:
            x_major_tick = float(x_major_tick_value)
        else:
            x_major_tick = None

        y_major_tick_field = 'Y_MAJOR_TICK_' + str(block_id)
        y_major_tick_value = request.POST.get(y_major_tick_field)
        if y_major_tick_value:
            y_major_tick = float(y_major_tick_value)
        else:
            y_major_tick = None

        x_minor_tick_field = 'X_MINOR_TICK_' + str(block_id)
        x_minor_tick_value = request.POST.get(x_minor_tick_field)
        if x_minor_tick_value:
            x_minor_tick = float(x_minor_tick_value)
        else:
            x_minor_tick = None

        y_minor_tick_field = 'Y_MINOR_TICK_' + str(block_id)
        y_minor_tick_value = request.POST.get(y_minor_tick_field)
        if y_minor_tick_value:
            y_minor_tick = float(y_minor_tick_value)
        else:
            y_minor_tick = None

        x_trans_field = 'X_TRANS_' + str(block_id)
        x_trans_value = request.POST.get(x_trans_field)
        if x_trans_value:
            x_trans = x_trans_value
        else:
            x_trans = 'x'

        y_trans_field = 'Y_TRANS_' + str(block_id)
        y_trans_value = request.POST.get(y_trans_field)
        if y_trans_value:
            y_trans = y_trans_value
        else:
            y_trans = 'y'

        print('now in process block!')
        if fit_type == 'poly':
            degree = int(degree)

        block.DEGREE = degree
        block.Y_TRANS = y_trans
        block.ER_MODE = er_mode
        block.X_TRANS = x_trans
        block.Y_MINOR_TICK = y_minor_tick
        block.Y_MAJOR_TICK = y_major_tick
        block.X_MAJOR_TICK = x_major_tick
        block.X_MINOR_TICK = x_minor_tick
        block.L_X_LIM = l_x_lim
        block.L_Y_LIM = l_y_lim
        block.R_X_LIM = r_x_lim
        block.R_Y_LIM = r_y_lim
        block.FIT_TYPE = fit_type
        print(block.FIT_TYPE)
        print('can enter!')
        try:
            block.process()
        except:
            block.DEGREE = degree_copy
            block.Y_TRANS = y_trans_copy
            block.ER_MODE = er_mode_copy
            block.X_TRANS = x_trans_copy
            block.Y_MINOR_TICK = y_minor_tick_copy
            block.Y_MAJOR_TICK = y_major_tick_copy
            block.X_MAJOR_TICK = x_major_tick_copy
            block.X_MINOR_TICK = x_minor_tick_copy
            block.L_X_LIM = l_x_lim_copy
            block.L_Y_LIM = l_y_lim_copy
            block.R_X_LIM = r_x_lim_copy
            block.R_Y_LIM = r_y_lim_copy
            block.FIT_TYPE = fit_type_copy
            block.process()
            raise ValueError
        block.save()
        print('can save!')
        return redirect('experiment_page', experiment_id=ex_id)
    return redirect('experiment_page', experiment_id=ex_id)
