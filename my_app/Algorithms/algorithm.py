# def edit_distance(A: str, B: str):  # for search projects
#     n = len(A)
#     m = len(B)
#     f = [[0 for i in range(m)] for j in range(n)]
#     for i in range(n):
#         f[i][0] = i
#     for j in range(m):
#         f[0][j] = j
#     for i in range(n):
#         for j in range(m):
#             if A[i] == B[j]:
#                 f[i][j] = f[i - 1][j - 1]
#             else:
#                 f[i][j] = min(f[i - 1][j], f[i][j - 1], f[i - 1][j - 1]) + 1
#     return f[n - 1][m - 1]

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import pickle
import sympy


class InvalidValueAfterTran(Exception):
    # in transformation when log has neg values
    pass


class InvalidSympify(Exception):
    # in checking input
    pass


class TooManyVariables(Exception):
    # too many variables in sympify expression
    pass


def get_vandermonde(X: pd.Series, degree: int):  # get vandermonde matrix for polynomial fitting
    # print("get vandermonde " + str(degree))
    vandermonde = []  # store the final matrix
    # vandermonde matrix is needed for fitting polynomial
    # for a degree n polynomial, polynomial fitting can be transformed into n-dimension linear regression
    # vandermonde matrix is the design matrix for the n-dimension regression
    for i in range(degree):
        vandermonde.append(np.power(X, i + 1))
        # every data to the power from 1-degree
    return pd.DataFrame(vandermonde).transpose()  # keep the shape commensurate


def cross_val(k: int, X: pd.Series, Y: pd.Series, max_deg: int):
    '''
    Use KFold to find the best degree of polynomial\n
    k = (k) in kfold\n
    X = independent\n
    Y = dependent\n
    max_deg = maximum degree in interation
    '''

    # mechanism: split the dataset into k (k>=2) segments, use one to train, others to test, repeat k times
    # for each split, store the MSE of using different degrees, and finally take the average for each degree
    # select the one with minimum MSE
    test_mse = []
    kf = KFold(n_splits=k)
    for train, test in kf.split(X, Y):  # random split
        split_mse = []  # MSE for this split
        for deg in range(1, max_deg + 1):
            model = LinearRegression()
            vandermonde_train = get_vandermonde(X.iloc[train], deg)
            vandermonde_test = get_vandermonde(X.iloc[test], deg)
            model.fit(vandermonde_train, Y.iloc[train])  # train the model
            mse = mean_squared_error(Y.iloc[test], model.predict(vandermonde_test))  # see whether this degree is good
            split_mse.append(mse)

        test_mse.append(split_mse)
    test_mse = np.mean(np.array(test_mse), axis=0)  # mean MSE of every degree
    opt_deg = np.argmin(test_mse) + 1
    return opt_deg


def decode_input(data: pd.DataFrame):
    '''
    Input: Experiment DF
    Output: xerr, yerr, xval, yval
    '''
    # X_Name = data.iloc[0, 0]  # first row and first col
    # Y_Name = data.iloc[0, 1:]  # first row except first col
    xerr = data.iloc[0, 0]
    yerr = data.iloc[0, 1:]
    X_Val = data.iloc[1:, 0]  # first col of every row starting from the third row
    Y_Val = data.iloc[1:, 1:]  # The rest
    return xerr, yerr, X_Val, Y_Val


def loss(b, X: np.array, Y: np.array):  # for ax^b+c
    model = LinearRegression(fit_intercept=False)
    if b > 0:
        X = np.power(X, b)
    elif b < 0:
        X = 1 / (np.power(X, -b) + 1e-9)
    else:
        b += 1e-6
    model.fit(X, Y)
    a = model.coef_[0][0]
    SE = 0
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        SE += (a * x - y) ** 2
    return SE


def power_fit(X: pd.Series, Y: pd.Series, degree: float = 0):  # ax^b=y
    """
    :param X: X array
    :param Y: Y array
    :param degree: degree of ax^b=y
    :return: model of ax^b=y
    """
    # ax^b=y
    X = X.values.reshape(-1, 1)
    Y = Y.values.reshape(-1, 1)
    model = LinearRegression(fit_intercept=False)
    if degree == 0:
        left = -100
        right = 100

        while abs(right - left) > 1e-9:
            # since right and left are floats, use a-b>eps, where eps is small to see whether a=b
            third = (right - left) / 3  # "ter"nary search
            mid1 = left + third
            mid2 = right - third

            if loss(mid1, X, Y) < loss(mid2, X, Y):  # if min not in range [mid2, right]
                right = mid2
            else:
                left = mid1

        b = (left + right) / 2  # best deg
        if b > 0:
            X_pow = np.power(X, b)
        elif b < 0:
            X_pow = 1 / (np.power(X, -b) + 1e-9)
        else:
            raise ValueError
        model.fit(X_pow, Y)
        return model, b

    X_pow = np.power(X, degree)
    model.fit(X_pow, Y)
    return model, degree


def lin_fit(X: pd.Series, Y: pd.Series):
    model = LinearRegression()  # linear regression model
    model.fit(X.values.reshape(-1, 1), Y.values.reshape(-1, 1))
    return model


def poly_fit(X: pd.Series, Y: pd.Series, degree=0):  # polynomial fit y=a0+a1*x+a2*x^2+...
    '''
    :param X: independent variable
    :param Y: dependent variable
    :param degree: if degree=0, user do not have a degree, else the degree of polynomial
    :return: if degree=0, (model, degree of polynomial), else the model
    '''
    max_deg = 20  # maximum degree possible in IB level, the polynomial with degree>20 is not practical...
    if degree == 0:  # if user does not choose a degree
        K = int(len(X) / 4) + 1  # number of k in kfold validation
        opt_deg = cross_val(K, X, Y, max_deg)  # cross validation
        vandermonde = get_vandermonde(X, opt_deg)  # vandermonde matrix for polynomial fitting
        # vandermonde matrix is needed for fitting polynomial
        # for a degree n polynomial, polynomial fitting can be transformed into n-dimension linear regression
        # vandermonde matrix is the design matrix for the n-dimension regression
        model = LinearRegression()
        model.fit(vandermonde, Y.values.reshape(-1, 1))  # a opt_deg dimensions linear regression:
        # y=a*x^n+b*x^(n-1)+... <==> y=a*x1+b*x2+..., where x1=x^n, x2=x^(n-1) etc.
        return model, opt_deg

    vandermonde = get_vandermonde(X, degree)
    model = LinearRegression()
    model.fit(vandermonde, Y)
    return model


def log_fit(X: pd.Series, Y: pd.Series):
    """
    :param X: independent variable
    :param Y: dependent variable
    :return: a log model, sklearn LinearRegression object
    """
    X_copy = X
    for i in range(len(X_copy)):  # check no neg values
        if X_copy.iloc[i] <= 0:
            X = X.drop(labels=i)  # drop all negative values record
            Y = Y.drop(labels=i)

    X = X.values.reshape(-1, 1)  # convert pd.Series to np.array, and put it into n*1 shape (convention)
    Y = Y.values.reshape(-1, 1)  #
    ln_X = np.log(X.reshape(-1, 1))
    model = LinearRegression()
    # for model y=a*log(x)+b, one can see it as y=a*t+b, where t=log(x), which is a linear model.
    # the "model" variable is a model that contains a and b
    # similar logic for exp_fit
    model.fit(ln_X, Y.reshape(-1, 1))
    return model


def exp_fit(X: pd.Series, Y: pd.Series):  # exp fit that fits y=ae^x+b
    X = X.values.reshape(-1, 1)
    Y = Y.values.reshape(-1, 1)
    print(X)
    t = np.exp(X.reshape(-1, 1))
    model = LinearRegression()
    model.fit(t, Y.reshape(-1, 1))
    return model


def select_best(X: pd.Series, Y: pd.Series):  # auto select the model that best fit with the data
    SE = {"poly": 0, "exp": 0, "log": 0, "lin": 0, "power": 0}  # square error, the best model has the smallest error

    polyfit, opt_deg = poly_fit(X, Y)  # model and degree for polynomial
    linfit = lin_fit(X, Y)  # linear model
    expfit = exp_fit(X, Y)  # e^x model
    try:  # log has a domain, check whether it is good to say X and Y is a log model
        logfit = log_fit(X, Y)
        X_log = np.log(X)
        SE["log"] = np.sum(np.square(logfit.predict(X_log.values.reshape(-1, 1)) - Y.values.reshape(-1, 1)))
    except ValueError:
        pass  # domain issue, do not consider log model
    powerfit, power_deg = power_fit(X, Y)  # power_deg is the optimum degree
    X_poly = get_vandermonde(X, opt_deg)  # vandermonde matrix, polynomial model only eat this
    X_exp = np.exp(X)  # transformation
    SE['poly'] = 5 * (np.sum(np.square(polyfit.predict(X_poly) - Y.values.reshape(-1, 1))) + 2)
    # polynomial is not good (it is not simple for a physics model), add a *5 penalty
    SE["exp"] = np.sum(np.square(expfit.predict(X_exp.values.reshape(-1, 1)) - Y.values.reshape(-1, 1)))
    SE['power'] = np.sum(
        np.square(powerfit.predict(np.power(X, power_deg).values.reshape(-1, 1)) - Y.values.reshape(-1, 1)))
    SE["lin"] = np.sum(np.square(linfit.predict(X.values.reshape(-1, 1)) - Y.values.reshape(-1, 1)))

    best = min(SE, key=SE.get)
    if best == "poly":
        return polyfit, "poly", opt_deg
    elif best == "lin":
        return linfit, "lin"
    elif best == "exp":
        return expfit, "exp"
    elif best == "log":
        return logfit, "log"
    elif best == "power":
        return powerfit, 'pow', power_deg


def cal_error(Y: pd.DataFrame, er_mode="range"):
    Y_avg = Y.mean(axis=1)
    if er_mode == "range":
        max_spread = -1e9
        for i in range(1, len(Y_avg)):
            max_spread = max(max(Y.iloc[i]) - min(Y.iloc[i]), max_spread)
        yerr = max_spread / 2
        return round(yerr, 1)
    elif er_mode == "sd":
        sd = np.std(Y, axis=1)
        return round(sd, 2)


def raw_fit(X, Y, fit_type: str, degree: float = 0):  # prepare function for plot, text, fit_type
    """
    ## fit types:
    - linear = lin
    - log = log
    - exp = exp
    - polynomial = poly
    - power = pow
    - auto = auto
    ## returns:
    f, relationship, fit_type, deg_out
    """
    deg_out = -1
    Y_avg = Y
    relationship = "Y="
    if fit_type == "poly" and degree == 0:
        f, opt = poly_fit(X, Y_avg)
        deg_out = opt
        intercept = f.intercept_
        coefficients = f.coef_
        coefficients = np.concatenate((intercept.flatten(), coefficients.flatten()))
        terms = [f"{coeff:.3f}*x^{i}" if i > 0 else f"{coeff:.2f}" for i, coeff in enumerate(coefficients)]
        relationship = relationship + "+".join(terms[::-1])

    elif fit_type == 'poly':
        deg_out = degree
        f = poly_fit(X, Y_avg, degree)
        intercept = f.intercept_
        coefficients = f.coef_
        coefficients = np.concatenate((intercept.flatten(), coefficients.flatten()))
        terms = [f"{coeff:.3f}*x^{i}" if i > 0 else f"{coeff:.2f}" for i, coeff in enumerate(coefficients)]
        relationship = relationship + "+".join(terms[::-1])

    elif fit_type == "lin":
        f = lin_fit(X, Y_avg)
        coefficients = f.coef_[0][0]
        intercept = f.intercept_[0]
        relationship = relationship + f"{coefficients:.3f}*x+" + f"{intercept:.3f}"

    elif fit_type == "log":
        f = log_fit(X, Y_avg)
        coefficients = f.coef_[0][0]
        intercept = f.intercept_[0]
        relationship = relationship + f"{coefficients:.3f}*lnx+" + f"{intercept:.3f}"

    elif fit_type == "exp":
        f = exp_fit(X, Y_avg)
        coefficients = f.coef_[0][0]
        intercept = f.intercept_[0]
        relationship = relationship + f"{coefficients:.3f}*exp(x)+" + f"{intercept:.3f}"

    elif fit_type == "pow" and degree == 0:
        f, deg = power_fit(X, Y_avg)
        deg_out = deg
        coefficients = f.coef_[0][0]
        relationship = relationship + f"{coefficients:.3f}*x^" + f"{deg:.3f}"

    elif fit_type == "pow" and degree != 0:
        deg_out = degree
        f, deg = power_fit(X, Y_avg, degree)
        coefficients = f.coef_[0][0]
        relationship = relationship + f"{coefficients:.3f}*x^" + f"{deg:.3f}"

    # --------------------auto----------------------------------#
    else:  # auto
        result = select_best(X, Y_avg)
        deg_out = 0
        if len(result) == 2:
            f, fit_type = result
        else:
            f, fit_type, d = result
            deg_out = d

        print('in auto')
        # if fit_type == "poly" and degree == 0:
        #     f, opt = poly_fit(X, Y_avg)
        #     print(f)
        #     deg_out = opt
        #     intercept = f.intercept_
        #     coefficients = f.coef_
        #     coefficients = np.concatenate((intercept.flatten(), coefficients.flatten()))
        #     terms = [f"{coeff:.3f}*x^{i}" if i > 0 else f"{coeff:.2f}" for i, coeff in enumerate(coefficients)]
        #     relationship = relationship + "+".join(terms[::-1])

        if fit_type == 'poly':
            # print(d, type(d))
            deg_out = d
            f = poly_fit(X, Y_avg, d)
            intercept = f.intercept_
            coefficients = f.coef_
            coefficients = np.concatenate((intercept.flatten(), coefficients.flatten()))
            terms = [f"{coeff:.3f}*x^{i}" if i > 0 else f"{coeff:.2f}" for i, coeff in enumerate(coefficients)]
            relationship = relationship + "+".join(terms[::-1])

        elif fit_type == "lin":
            f = lin_fit(X, Y_avg)
            coefficients = f.coef_[0][0]
            intercept = f.intercept_[0]
            relationship = relationship + f"{coefficients:.3f}*x+" + f"{intercept:.3f}"

        elif fit_type == "log":
            f = log_fit(X, Y_avg)
            coefficients = f.coef_[0][0]
            intercept = f.intercept_[0]
            relationship = relationship + f"{coefficients:.3f}*lnx+" + f"{intercept:.3f}"

        elif fit_type == "exp":
            f = exp_fit(X, Y_avg)
            coefficients = f.coef_[0][0]
            intercept = f.intercept_[0]
            relationship = relationship + f"{coefficients:.3f}*exp(x)+" + f"{intercept:.3f}"

        elif fit_type == "pow":
            f, deg = power_fit(X, Y_avg, d)
            deg_out = d
            coefficients = f.coef_[0][0]
            relationship = relationship + f"{coefficients:.3f}*x^" + f"{deg:.3f}"

    return f, relationship, fit_type, deg_out


def transform(X: pd.Series, Y: pd.Series, xerr: pd.Series, yerr: pd.Series, x_trans: str, y_trans: str):
    """
    Function process transformation on data by user-defined functions, given by string
    :param X: independent variable
    :param Y: dependent variable
    :param xerr: error in X
    :param yerr: error in Y
    :param x_trans transformation function for X
    :param y_Trans transformation function for Y
    :return transformed X array, transformed y array, transformed x error, transformed y error
    ## Use example:
    - y=sqrt(exp(x))
    - y^2=exp(x)
    - y_trans="y^2", x_trans=exp(x)
    - return value=transformedX, transformedY, transformedXerr, transformedYerr
    # NOTICE: ADD BRACKETS WHEN USING FUNCTION, LOG ONLY SUPPORT POSITIVE INTEGERS AND NATUAL LOG(LOG means LN, not LOG10),
      MULTIPLICATION SIGN NEEDED
    """
    try:
        x_expr = sympy.sympify(x_trans)  # parse user input to sympy expression
        y_expr = sympy.sympify(y_trans)
        x_free_symbols = np.array(list(x_expr.free_symbols))  # get a list of variables
        y_free_symbols = np.array(list(y_expr.free_symbols))
        # need to check only ONE VARIABLE!!!
        varx = x_free_symbols[0]
        vary = y_free_symbols[0]
    except:  # sympy does not understand input
        raise InvalidSympify
    if len(x_free_symbols) != 1 or len(y_free_symbols) != 1:
        raise TooManyVariables

    x_tr = sympy.lambdify(varx, x_expr)  # transform function
    y_tr = sympy.lambdify(vary, y_expr)  # transform function

    trans_X = x_tr(X)
    trans_Y = y_tr(Y)
    delta_x = sympy.symbols('deltax')  # error
    delta_y = sympy.symbols('deltay')
    fx_prime = sympy.diff(x_expr, varx)  # df/dx
    fy_prime = sympy.diff(y_expr, vary)  # df/dy
    dx = fx_prime * delta_x  # this is a function of x and dx
    dy = fy_prime * delta_y

    dx_tr = sympy.lambdify([varx, delta_x], dx)  # transform error function
    dy_tr = sympy.lambdify([vary, delta_y], dy)
    xerr = np.abs(dx_tr(X, xerr))
    yerr = np.abs(dy_tr(Y, yerr))

    for i in trans_X:
        if np.isnan(i):
            raise InvalidValueAfterTran
    for i in trans_Y:
        if np.isnan(i):
            raise InvalidValueAfterTran

    return trans_X, trans_Y, xerr, yerr


def plot_graph(X: pd.Series, Y: pd.Series, fit_func: LinearRegression, relationship: str, fit_type: str, deg, xerr,
               yerr,
               x_range: list, y_range: list, xtick: list, ytick: list):
    """deg if not use = 0"""
    # if x_range is None:
    #     x_range = list()
    #     x_range.append(min(X) * (1 - 0.1))
    #     x_range.append(max(X) * (1 + 0.1))
    # if y_range is None:
    #     y_range = list()
    #     y_range.append(min(Y) * (1 - 0.1))
    #     y_range.append(max(Y) * (1 + 0.1))
    if x_range[0] is None:
        x_range[0] = min(X) * (1 - 0.1)
    if x_range[1] is None:
        x_range[1] = max(X) * (1 + 0.1)
    if y_range[0] is None:
        y_range[0] = min(Y) * (1 - 0.1)
    if y_range[1] is None:
        y_range[1] = max(Y) * (1 + 0.1)

    fig, ax = plt.subplots()
    print('in plot_graph, xtick, ytick')
    print(xtick)
    print(ytick)
    print('---------')
    if xtick[0] is None:
        ax.set_xticks(np.linspace(x_range[0], x_range[1], 5))
    else:
        ax.set_xticks(np.arange(x_range[0], x_range[1], step=xtick[0]))
    if xtick[1] is None:
        ax.set_xticks(np.linspace(x_range[0], x_range[1], 25), minor=True)
    else:
        ax.set_xticks(np.arange(x_range[0], x_range[1], step=xtick[1]), minor=True)
    if ytick[0] is None:
        ax.set_yticks(np.linspace(y_range[0], y_range[1], 5))
    else:
        ax.set_yticks(np.arange(y_range[0], y_range[1], step=ytick[0]))
    if ytick[1] is None:
        ax.set_yticks(np.linspace(y_range[0], y_range[1], 25), minor=True)
    else:
        ax.set_yticks(np.arange(y_range[0], y_range[1], step=ytick[1]), minor=True)


    if fit_type == 'pow':
        x_plot = np.linspace(x_range[0], x_range[1], 1000)
        x_plot = x_plot[x_plot != 0]
        y_plot = fit_func.predict(np.power(x_plot, deg).reshape(-1, 1))
    if fit_type == 'lin':
        x_plot = np.linspace(x_range[0], x_range[1], 1000)
        y_plot = fit_func.predict(x_plot.reshape(-1, 1))
    if fit_type == 'exp':
        x_plot = np.linspace(x_range[0], x_range[1], 1000)
        y_plot = fit_func.predict(np.exp(x_plot).reshape(-1, 1))
    if fit_type == 'poly':
        x_plot = np.linspace(x_range[0], x_range[1], 1000)
        vandermonde = get_vandermonde(pd.Series(x_plot), deg)
        y_plot = fit_func.predict(vandermonde)
    if fit_type == 'log':
        x_plot = np.linspace(x_range[0], x_range[1], 1000)
        y_plot = fit_func.predict(np.log(x_plot).reshape(-1, 1))

    ax.plot(x_plot, y_plot)
    ax.text(x_range[0], 1.001 * y_range[1], relationship)
    ax.scatter(X, Y, marker='x')
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.errorbar(x=X, y=Y, xerr=xerr, yerr=yerr, ls='none')
    ax.grid(True, which='both')
    fig = plt.gcf()
    plot_data = pickle.dumps(fig)
    return plot_data
