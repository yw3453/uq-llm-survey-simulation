import random, copy
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# given data, calculate the CLT confidence interval
# data: a list or numpy array of data
# gamma: the confidence level (e.g., 0.95)
# return: a list of three elements, [mean, lower bound, upper bound]
def CI(data, gamma):
    n = len(data)
    m, se = np.mean(data), np.std(data)
    h = norm.ppf(1/2 + gamma/2)
    me = h * se / np.sqrt(n)
    return [m, m-me, m+me]

# given synthetic answers, calculate the synthetic confidence interval based on the first k answers
# answers: a list or numpy array of answers
# k: the number of answers to use
# alpha: the significance level (e.g., 0.05)
# C: a scaling constant for the confidence interval half-width
# CI_type: 'Hoeffding' or 'CLT'; 'Hoeffding' for Hoeffding's inequality, 'CLT' for the Central Limit Theorem
# rv_bound: the range of the random variable (e.g., 1 for Bernoulli)
# return: a list of three elements, [mean, lower bound, upper bound]
def synthetic_CI(answers, k, alpha, C = 2, CI_type = 'Hoeffding', rv_bound = 1):
    data = answers[:k]
    m = np.mean(data)
    if CI_type == 'Hoeffding':
        me = C * rv_bound * np.sqrt(np.log(2/alpha) / (2 * k))
    elif CI_type == 'CLT':
        se = np.std(data)
        h = norm.ppf(1-alpha/2)
        me = C * h * se / np.sqrt(k)
    lower, upper = m - me, m + me
    return [m, lower, upper]

# calculate the miscoverage rate of the synthetic confidence interval
# real_CIs: a dictionary of real confidence intervals, key is the question id, value is a list of three elements [mean, lower bound, upper bound]
# synthetic_data: a dictionary of synthetic answers, key is the question id, value is a list of answers
# indices: a list of question ids to use
# k: the number of synthetic answers to use
# alpha: the significance level
# C: a scaling constant for the confidence interval half-width
# type: 'interval' or 'point'; 'interval' for confidence set inclusion, 'point' for empirical mean inclusion
# synth_CI_type: 'Hoeffding' or 'CLT'; 'Hoeffding' for CI based on Hoeffding's inequality, 'CLT' for CI based on the Central Limit Theorem
# rv_bound: the range of the random variable (e.g., 1 for Bernoulli)
# return: the miscoverage rate
def G(real_CIs, synthetic_data, indices, k, alpha, C = 2, type = 'interval', synth_CI_type = 'Hoeffding', rv_bound = 1):
    m = len(indices)
    coverage = 0
    for i in indices:
        real_CI = real_CIs[i]
        synth_data_i = synthetic_data[i]
        synth_CI = synthetic_CI(synth_data_i, k, alpha, C, synth_CI_type, rv_bound)
        if type == 'interval' and real_CI[1] >= synth_CI[1] and real_CI[2] <= synth_CI[2]: # the real CI is within the synthetic CI
            coverage += 1
        elif type == 'point' and real_CI[0] >= synth_CI[1] and real_CI[0] <= synth_CI[2]: # the empirical mean from the real data is within the synthetic CI
            coverage += 1
    return (m - coverage) / m

# find the smallest k such that the miscoverage rate is below the threshold
# ks: a list of k values
# Gks: a list of miscoverage rates
# threshold: the threshold for the miscoverage rate
# return: the smallest k such that the miscoverage rate is below the threshold or 0 if no such k exists
def find_k_hat(ks, Gks, threshold):
    ind_list = np.where(Gks > threshold)[0]
    if len(ind_list) == 0:
        return 0
    else:
        return ks[ind_list[0] - 1]
    
# train-test split the data once and find the k hat, synthetic CI width, and test miscoverage rate
# real_answers: a dictionary of real answers, key is the question id, value is a list of answers
# synthetic_answers: a dictionary of synthetic answers, key is the question id, value is a list of answers
# questions_id: a list of question ids to use
# alpha: the significance level for the synthetic CI
# gamma: the confidence level for the real CI
# k_max: the maximum number of synthetic answers to use
# C: a scaling constant for the synthetic CI half-width
# train_proportion: the proportion of questions to use for training
# type: 'interval' or 'point'; 'interval' for confidence set inclusion, 'point' for empirical mean inclusion
# synth_CI_type: 'Hoeffding' or 'CLT'; 'Hoeffding' for CI based on Hoeffding's inequality, 'CLT' for CI based on the Central Limit Theorem
# rv_bound: the range of the random variable (e.g., 1 for Bernoulli)
# seed: the random seed for the train-test split
# report: whether to print the results and plot the upcrossing graph
# k_report: the frequency of printing the progress
# return: k_hat, synthetic CI width, and test miscoverage rate
def one_split_run(real_answers, synthetic_answers, questions_id, alpha, gamma, k_max, C, train_proportion, type, 
                  synth_CI_type = 'Hoeffding', rv_bound = 1, seed = 0, report = True, k_report = 100):

    ks = np.arange(1, k_max+1)

    # calculate the real confidence intervals
    real_CIs = {}
    for question_id in questions_id:
        real_CIs[question_id] = CI(real_answers[question_id], gamma)
    
    # randomly select training and test questions
    random.seed(seed)
    random_questions_id = copy.deepcopy(questions_id)
    random.shuffle(random_questions_id)
    train_test_split_index = int(train_proportion * len(random_questions_id))
    train_questions, test_questions = random_questions_id[:train_test_split_index], random_questions_id[train_test_split_index:]

    # find the smallest k such that the miscoverage rate is below the threshold
    Gks = np.zeros(len(ks))
    for i in range(len(ks)):
        k = ks[i]
        Gks[i] = G(real_CIs, synthetic_answers, train_questions, k, alpha, C, type, synth_CI_type, rv_bound)
        if report and (k+1) % k_report == 0:
            print('k =', k, end = '\r')
    if type == 'interval':
        threshold = alpha * gamma
    elif type == 'point':
        threshold = alpha / 2
    k_hat = find_k_hat(ks, Gks, threshold)
    if k_hat == 0:
        k_hat = ks[-1]
    synth_CI_width = C * rv_bound * np.sqrt(np.log(2/alpha) / (2 * k_hat))
    test_G = G(real_CIs, synthetic_answers, test_questions, k_hat, alpha, C, type, synth_CI_type, rv_bound)
    
    if report:
        plt.plot(ks, Gks, label = 'train miscoverage rate')
        plt.xlabel('k')
        plt.ylabel('train miscoverage rate')
        plt.hlines(threshold, ks[0], ks[-1], colors = 'r', linestyles = 'dashed', label = 'threshold')
        plt.legend(framealpha=0.3)
        print('threshold: ', threshold)
        print('k_hat: ', k_hat)
        print('synthetic CI width: ', synth_CI_width)
        print('test miscoverage rate: ', test_G)
    
    return k_hat, synth_CI_width, test_G

# train-test split the data multiple times and assess the statistics of k hat, synthetic CI width, and test miscoverage rate
# see one_split_run for most of the parameter descriptions
# num_splits: the number of train-test splits
# split_report: the frequency of printing the progress over the splits
# report: whether to calculate the statistics of k hat, synthetic CI width, and test miscoverage rate
# return: the results (and the statistics)
def multiple_split_run(real_answers, synthetic_answers, questions_id, alpha, gamma, k_max, C, train_proportion, type, 
                       synth_CI_type = 'Hoeffding', rv_bound = 1, seed = 0, num_splits = 100, split_report = 1, report = True):
    
    np.random.seed(seed)
    split_seeds = np.random.randint(0, 10000, num_splits)
    split_seeds = [int(seed) for seed in split_seeds]

    results = pd.DataFrame(columns = ['k_hat', 'synth_CI_width', 'test_miscov_rate'])
    for i in range(num_splits):
        k_hat, synth_CI_width, test_G = one_split_run(real_answers, synthetic_answers, questions_id, alpha, gamma, k_max, C, train_proportion, type, 
                                                      synth_CI_type, rv_bound, split_seeds[i], False)
        results.loc[i] = [k_hat, synth_CI_width, test_G]
        if (i+1) % split_report == 0:
            print('split', i, 'done', end = '\r')
    
    if report:
        statistics = [[np.mean(results['k_hat']), np.std(results['k_hat'])], [np.mean(results['synth_CI_width']), np.std(results['synth_CI_width'])], 
                      [np.mean(results['test_miscov_rate']), np.std(results['test_miscov_rate'])]]
        df_report = pd.DataFrame(statistics, index = ['k_hat', 'synth_CI_width', 'test_miscov_rate'], columns = ['mean', 'std'])
        return results, df_report
    else:
        return results

# assess the statistics of k hat, synthetic CI width, and test miscoverage rate over multiple train-test splits over multiple alphas
# see multiple_split_run for most of the parameter descriptions
# alphas: a list of significance levels for the synthetic CI
# report_alpha: whether to print the progress over the alphas
# return: the results for both interval and point types
def get_reports(real_answers, synthetic_answers, questions_id, alphas, gamma, k_max, C, train_proportion, synth_CI_type = 'Hoeffding', rv_bound = 1, seed = 0, num_splits = 100, report_alpha = False):
    reports_interval = []
    reports_point = []
    for alpha in alphas:
        _, report_interval = multiple_split_run(real_answers, synthetic_answers, questions_id, alpha, gamma, k_max, C, train_proportion,
                                                type = 'interval', synth_CI_type = synth_CI_type, rv_bound = rv_bound, seed = seed, num_splits = num_splits, split_report = 100000, report = True)
        _, report_point = multiple_split_run(real_answers, synthetic_answers, questions_id, alpha, gamma, k_max, C, train_proportion,
                                             type = 'point', synth_CI_type = synth_CI_type, rv_bound = rv_bound, seed = seed, num_splits = num_splits, split_report = 100000, report = True)
        reports_interval.append(report_interval)
        reports_point.append(report_point)
        if report_alpha:
            print(alpha, end = '\r')
    return reports_interval, reports_point

# assess the statistics of k hat, synthetic CI width, and test miscoverage rate over multiple train-test splits over multiple alphas and multiple synthetic answers (from different models)
# see get_reports for most of the parameter descriptions
# synthetic_answers_all: a dictionary of synthetic answers, key is the model name, value is a dictionary of synthetic answers
# return: the results for both interval and point types
def get_reports_multiple(real_answers, synthetic_answers_all, questions_id, alphas, gamma, k_max, C, train_proportion, synth_CI_type = 'Hoeffding', rv_bound = 1, seed = 0, num_splits = 100):
    keys = list(synthetic_answers_all.keys())
    reports_interval_all = {}
    reports_point_all = {}
    for key in keys:
        synthetic_answers = synthetic_answers_all[key]
        reports_interval, reports_point = get_reports(real_answers, synthetic_answers, questions_id, alphas, gamma, k_max, C, train_proportion, synth_CI_type = synth_CI_type, rv_bound = rv_bound, seed = seed, num_splits = num_splits, report_alpha = False)
        reports_interval_all[key] = reports_interval
        reports_point_all[key] = reports_point
        print(key)
    return reports_interval_all, reports_point_all

# plot the results of the reports
# reports_all: a dictionary of reports (either the interval type or the point type), key is the model name, value is a list of reports
# num_splits: the number of train-test splits
# alphas: a list of significance levels for the synthetic CI
# gamma: the confidence level for the real CI
# metric: 'k_hat', 'synth_CI_width', 'test_miscov_rate' (two-sided 95% CI), or 'test_miscov_rate_lower' (one-sided 95% CI lower bound)
# type: should follow the type of the reports, 'interval' or 'point'
# return nothing
def plot_reports(reports_all, num_splits, alphas, gamma, metric = 'test_miscov_rate', type = 'interval'):
    means_all, errs_all = {}, {}
    for key in reports_all.keys():
        reports = reports_all[key]
        # here, errs are 95% CI half-widths except for test_miscov_rate_lower
        if metric == 'k_hat':
            means = np.array([reports[i]['mean'].values[0] for i in range(len(reports))])
            errs = np.array([1.96 / np.sqrt(num_splits) * reports[i]['std'].values[0] for i in range(len(reports))])
        elif metric == 'synth_CI_width':
            means = np.array([reports[i]['mean'].values[1] for i in range(len(reports))])
            errs = np.array([1.96 / np.sqrt(num_splits) * reports[i]['std'].values[1] for i in range(len(reports))])
        elif metric == 'test_miscov_rate':
            means = np.array([reports[i]['mean'].values[2] for i in range(len(reports))]) / gamma
            errs = np.array([1.96 / np.sqrt(num_splits) * reports[i]['std'].values[2] for i in range(len(reports))]) / gamma
        elif metric == 'test_miscov_rate_lower':
            means = np.array([reports[i]['mean'].values[2] for i in range(len(reports))]) / gamma
            errs = np.array([1.645 / np.sqrt(num_splits) * reports[i]['std'].values[2] for i in range(len(reports))]) / gamma
        means_all[key] = means
        errs_all[key] = errs
    
    #plt.figure(figsize = (10, 6))
    plt.figure(figsize = (6, 6))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    if metric == 'test_miscov_rate_lower':
        for i, key in enumerate(reports_all.keys()):
            plt.plot(alphas, means_all[key] - errs_all[key], marker='o', linestyle='-', label = key, color = colors[i])
    else:
        for i, key in enumerate(reports_all.keys()):
            means = means_all[key]
            errs = errs_all[key]
            plt.errorbar(alphas, means, errs, capsize = 3, label = key, color = colors[i])
    if metric == 'test_miscov_rate' or metric == 'test_miscov_rate_lower':
        if type == 'interval':
            thresholds = alphas
        elif type == 'point':
            thresholds = alphas / 2
        plt.plot(alphas, thresholds, color = colors[-1], label = 'threshold')
    plt.xlabel(r'$\alpha$', fontsize = 16)
    if metric == 'k_hat':
        plt.ylabel(r'$\hat{k}$', fontsize = 16)
    elif metric == 'synth_CI_width':
        plt.ylabel('synthetic CI width', fontsize = 16)
    elif metric == 'test_miscov_rate':
        plt.ylabel('proxy for test miscoverage probability', fontsize = 16)
    elif metric == 'test_miscov_rate_lower':
        plt.ylabel('proxy for test miscoverage probability \n (CI lower bound)', fontsize = 16, multialignment='center')
    plt.legend(framealpha=0.3)

# create a table of the results of the reports
# see plot_reports for the parameter descriptions
# return: a table of the results
def table_reports(reports_all, num_splits, alphas, gamma, metric = 'test_miscov_rate'):
    keys = list(reports_all.keys())
    table = pd.DataFrame(index = keys, columns = alphas)
    for key in keys:
        reports = reports_all[key]
        if metric == 'k_hat':
            means = np.array([reports[i]['mean'].values[0] for i in range(len(reports))])
            errs = np.array([1.96 / np.sqrt(num_splits) * reports[i]['std'].values[0] for i in range(len(reports))])
        elif metric == 'synth_CI_width':
            means = np.array([reports[i]['mean'].values[1] for i in range(len(reports))])
            errs = np.array([1.96 / np.sqrt(num_splits) * reports[i]['std'].values[1] for i in range(len(reports))])
        elif metric == 'test_miscov_rate':
            means = np.array([reports[i]['mean'].values[2] for i in range(len(reports))]) / gamma
            errs = np.array([reports[i]['std'].values[2] / np.sqrt(num_splits) for i in range(len(reports))]) / gamma
        to_record = [str(np.round(means[i], 4)) + u"\u00B1" + str(np.round(errs[i], 4)) for i in range(len(means))]
        table.loc[key] = to_record
    return table

# calculate the p-values of the null hypothesis that L_tilde(k_hat) <= alpha
# see table_reports for the parameter descriptions
# return: a table of the p-values
def test_miscoverage(reports_all, num_splits, alphas, gamma):
    p_vals = pd.DataFrame(columns = alphas)
    for key in reports_all.keys():
        reports = reports_all[key]
        means = np.array([reports[i]['mean'].values[2] for i in range(len(reports))]) / gamma
        std_errs = np.array([reports[i]['std'].values[2] / np.sqrt(num_splits) for i in range(len(reports))]) / gamma
        p_vals_row = norm.cdf(-(means - alphas) / std_errs)
        p_vals.loc[key] = np.round(p_vals_row, 4)
    p_vals.index = reports_all.keys()
    return p_vals

