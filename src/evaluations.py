"""
Evaluation module for assessing the quality of synthetic survey responses.

This module provides functionality to:
1. Calculate confidence intervals for real survey data using the Central Limit Theorem (CLT)
2. Compute synthetic confidence intervals from LLM-generated responses
3. Evaluate miscoverage rates to assess how well synthetic CIs capture real CIs
4. Find optimal sample sizes (k_hat) for synthetic responses
5. Perform train-test splits and cross-validation to assess model performance
6. Generate reports and visualizations comparing different LLM models

The module implements statistical methods for uncertainty quantification, including
confidence set inclusion tests and empirical mean inclusion tests.
"""

import os
import random
import copy
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt

# ========================================================================
# VISUALIZATION CONSTANTS
# ========================================================================
# Color for threshold lines in plots
THRESHOLD_COLOR = 'black'

LLM_PLOT_INFO = {
    'claude-3.5-haiku': {
        'label': 'Claude 3.5 Haiku',
        'color': '#FFA500',
        'marker': 'o'
    },
    'deepseek-v3': {
        'label': 'Deepseek V3',
        'color': '#0096FF',
        'marker': 's'
    },
    'gpt-3.5-turbo': {
        'label': 'GPT-3.5 Turbo',
        'color': 'green',
        'marker': 'D'
    },
    'gpt-4o-mini': {
        'label': 'GPT-4o mini',
        'color': '#F0E442',
        'marker': 'X'
    },
    'gpt-4o': {
        'label': 'GPT-4o',
        'color': '#4B67B3',
        'marker': 'P'
    },
    'gpt-5-mini': {
        'label': 'GPT-5 mini',
        'color': '#E75480',
        'marker': 'v'
    },
    'llama-3.3-70B-instruct-turbo': {
        'label': 'Llama 3.3 70B',
        'color': 'purple',
        'marker': '^'
    },
    'mistral-7B-instruct-v0.3': {
        'label': 'Mistral 7B',
        'color': 'brown',
        'marker': '<'
    },
    'random': {
        'label': 'Random',
        'color': 'gray',
        'marker': '>'
    }
}


def _precompute_real_CIs(real_answers: dict, questions_id: list, gamma: float) -> dict:
    """
    Precompute real confidence intervals once so downstream routines can reuse them.
    """
    return {
        question_id: CI(np.asarray(real_answers[question_id]), gamma)
        for question_id in questions_id
    }


def _precompute_synthetic_stats(synthetic_answers: dict, questions_id: list) -> dict:
    """
    Precompute prefix sums for synthetic answers to allow O(1) mean/variance queries
    for any prefix length k.
    """
    stats = {}
    for question_id in questions_id:
        values = np.asarray(synthetic_answers[question_id], dtype=float)
        if values.size == 0:
            stats[question_id] = {
                'values': values,
                'prefix_sum': np.array([]),
                'prefix_sq_sum': np.array([]),
                'length': 0
            }
            continue
        prefix_sum = np.cumsum(values)
        prefix_sq_sum = np.cumsum(values ** 2)
        stats[question_id] = {
            'values': values,
            'prefix_sum': prefix_sum,
            'prefix_sq_sum': prefix_sq_sum,
            'length': values.size
        }
    return stats

def CI(data: np.ndarray, gamma: float) -> list:
    """
    Calculate confidence interval using the Central Limit Theorem (CLT) 
    with gamma coverage probability for the population mean.
    
    Args:
        data (np.ndarray): Array-like object containing the data points.
            Can be a list, numpy array, or pandas Series.
        gamma (float): Coverage probability (between 0 and 1).
            For example, gamma=0.95 gives a 95% confidence interval (95% coverage probability).
    
    Returns:
        list: A list of three elements:
            - [0]: Sample mean (m)
            - [1]: Lower bound of confidence interval (m - margin_of_error)
            - [2]: Upper bound of confidence interval (m + margin_of_error)
        
        Format: [mean, lower_bound, upper_bound]
    
    Formula:
        - Sample mean: m = (1/n) * Σx_i
        - Sample standard deviation: s = sqrt((1/(n-1)) * Σ(x_i - m)²)
        - Margin of error: ME = z_{1-gamma/2} * s / sqrt(n)
        - Confidence interval: [m - ME, m + ME]
    
    Note:
        - Uses sample standard deviation (ddof=1) for unbiased estimation
        - Assumes data is approximately normally distributed (CLT approximation)
    """
    # ========================================================================
    # COMPUTE SAMPLE STATISTICS
    # ========================================================================
    n = len(data)  # Sample size
    # Sample mean
    m = np.mean(data)
    # Sample standard deviation (ddof=1 for unbiased estimate)
    # ddof=1 means divide by (n-1) instead of n
    se = np.std(data, ddof=1)
    
    # ========================================================================
    # CALCULATE CRITICAL VALUE AND MARGIN OF ERROR
    # ========================================================================
    # Critical value for two-sided confidence interval
    # norm.ppf(1/2 + gamma/2) gives the z-score for gamma coverage probability or (1-gamma) confidence level
    # For gamma=0.95: norm.ppf(0.975) ≈ 1.96 (95% coverage)
    h = norm.ppf(1/2 + gamma/2)
    # Margin of error using CLT formula
    me = h * se / np.sqrt(n)
    
    # ========================================================================
    # RETURN CONFIDENCE INTERVAL
    # ========================================================================
    return [m, m - me, m + me]

def synthetic_CI(
    answers: np.ndarray,
    k: int,
    alpha: float,
    C: float = 2,
    k_min: int = 2,
    full_param_CI: list = [0, 1],
    CI_type: str = 'clt',
    precomputed_stats: dict | None = None
) -> list:
    """
    Calculate synthetic confidence interval from the first k synthetic answers.
    
    If k > len(answers), uses all available answers and the actual sample size n (not k)
    in all CI calculations. If n <= k_min, returns a conservative interval covering the
    full parameter space. The scaling factor C controls the width of the interval for
    conservative uncertainty quantification.
    
    Args:
        answers (np.ndarray): Array of synthetic answers/responses.
            Typically numeric values (e.g., correctness scores 0/1, or opinion scores).
        k (int): Number of answers to use for calculating the CI.
            Only the first k elements of answers will be used.
            If k > len(answers), uses all available answers and the actual sample size n
            (where n = len(answers)) in all CI formulas instead of k.
            Must be positive.
        alpha (float): Significance level (between 0 and 1).
            The confidence level is (1-alpha).
        C (float, optional): Scaling constant for the confidence interval half-width.
            Defaults to 2. The margin of error is multiplied by sqrt(C).
        k_min (int, optional): Minimum number of answers required to compute a valid CI.
            Defaults to 2. Must be at least 2. If n <= k_min (where n is the actual number of data points used),
            returns the full parameter CI instead.
            This prevents unreliable intervals from very small samples.
        full_param_CI (list, optional): Full parameter space bounds.
            Defaults to [0, 1] (appropriate for proportions/probabilities).
            Format: [lower_bound, upper_bound].
            Used when n <= k_min to return a conservative interval.
        CI_type (str, optional): Type of confidence interval to compute.
            Defaults to 'clt'.
            - 'clt': Central Limit Theorem-based confidence interval.
            - 'hoeffding': Hoeffding's inequality-based confidence interval (modified with sqrt(C) factor).
            - 'bernstein': Bernstein's inequality-based confidence interval (modified with sqrt(C) and C factors).
        precomputed_stats (dict, optional): Cached prefix statistics for answers.
            Expected keys: {'prefix_sum', 'prefix_sq_sum', 'length'}. Provides large
            speedups when computing CIs repeatedly over many k values. If None, stats
            are computed directly from the sliced data.
    
    Returns:
        list: A list of three elements:
            - [0]: Sample mean of the data points used (first k answers, or all if k > len(answers))
            - [1]: Lower bound of synthetic CI
            - [2]: Upper bound of synthetic CI
        
        Format: [mean, lower_bound, upper_bound]
        
        If n <= k_min: Returns [mean, full_param_CI[0], full_param_CI[1]]
        Otherwise: Returns CI computed from data using actual sample size n with scaling factor C
    
    Note:
        - Uses sample standard deviation (ddof=1) for unbiased estimation
        - When k > len(answers), all formulas use n = len(answers) instead of k
    """
    # ========================================================================
    # EXTRACT FIRST K ANSWERS AND COMPUTE MEAN
    # ========================================================================
    # Handle edge case: k=0 (should not occur in normal operation)
    if k <= 0:
        # Return conservative interval covering full parameter space
        # Use midpoint of full parameter space as mean estimate
        m = (full_param_CI[0] + full_param_CI[1]) / 2
        return [m, full_param_CI[0], full_param_CI[1]]
    
    if precomputed_stats is not None:
        n_available = precomputed_stats['length']
        if n_available == 0:
            m = (full_param_CI[0] + full_param_CI[1]) / 2
            return [m, full_param_CI[0], full_param_CI[1]]
        n = min(k, n_available)
        prefix_sum = precomputed_stats['prefix_sum'][n - 1]
        m = prefix_sum / n
    else:
        # Use only the first k answers for computing the synthetic CI
        # If k > len(answers), use all available answers instead
        data = answers[:k]
        # Actual number of data points used (may be less than k if k > len(answers))
        n = len(data)
        if n == 0:
            m = (full_param_CI[0] + full_param_CI[1]) / 2
            return [m, full_param_CI[0], full_param_CI[1]]
        # Sample mean of the data points used
        m = np.mean(data)
    
    # ========================================================================
    # CHECK IF WE HAVE ENOUGH DATA
    # ========================================================================
    # If we don't have enough data points (n <= k_min), return conservative interval
    if n <= k_min or n <= 1: # If n <= 1, return conservative interval
        return [m, full_param_CI[0], full_param_CI[1]]
    
    if CI_type == 'clt':
        if precomputed_stats is not None:
            prefix_sq_sum = precomputed_stats['prefix_sq_sum'][n - 1]
            variance = (prefix_sq_sum - n * m ** 2) / (n - 1)
            sd = np.sqrt(max(variance, 0.0))
        else:
            sd = np.std(data, ddof=1)
        # Critical value for (1-alpha) confidence level
        h = norm.ppf(1 - alpha/2)
        # Scaled margin of error: sqrt(C) multiplies the standard error
        # This creates a more conservative (wider) interval
        # Use n (actual number of data points) instead of k
        me = np.sqrt(C) * h * sd / np.sqrt(n)
    elif CI_type == 'hoeffding':
        # Hoeffding's inequality: P(|X̄ - μ| ≥ t) ≤ 2 * exp(-2 * n * t² / (b - a)²)
        # For (1-α) confidence: t = (b - a) * sqrt(log(2/α) / (2 * n))
        # Note: We include sqrt(C) factor for consistency with CLT-based approach
        # Use n (actual number of data points) instead of k
        me = (full_param_CI[1] - full_param_CI[0]) * np.sqrt(C * np.log(2/alpha) / (2 * n))
    elif CI_type == 'bernstein':
        # Bernstein's inequality requires standard deviation
        if precomputed_stats is not None:
            prefix_sq_sum = precomputed_stats['prefix_sq_sum'][n - 1]
            variance = (prefix_sq_sum - n * m ** 2) / (n - 1)
            sd = np.sqrt(max(variance, 0.0))
        else:
            sd = np.std(data, ddof=1)
        # Bernstein's inequality with variance term and range term
        # Use n (actual number of data points) instead of k
        me = sd * np.sqrt(2 * C * np.log(4/alpha) / n) + 7 / 3 * C * (full_param_CI[1] - full_param_CI[0]) * np.log(4/alpha) / (n - 1)
    else:
        raise ValueError(f"Invalid CI_type: {CI_type}. Must be 'clt', 'hoeffding', or 'bernstein'.")
    
    return [m, m - me, m + me]

def G(
    real_CIs: dict,
    synthetic_data: dict,
    indices: list,
    k: int,
    alpha: float,
    C: float = 2,
    type: str = 'general',
    k_min: int = 2,
    full_param_CI: list = [0, 1],
    CI_type: str = 'clt',
    synthetic_stats: dict | None = None
) -> tuple:
    """
    Calculate the miscoverage rate of synthetic confidence intervals, defined as the proportion of
    questions where the synthetic CI fails to cover the real CI.
    
    Coverage types:
    - 'general': Entire real CI must be within synthetic CI
    - 'simple': Only the empirical mean from real data must be within synthetic CI
    
    Args:
        real_CIs (dict): Dictionary mapping question IDs to real confidence intervals.
            Keys: Question identifiers (strings or integers)
            Values: Lists of three elements [mean, lower_bound, upper_bound]
        synthetic_data (dict): Dictionary mapping question IDs to synthetic answers.
            Keys: Question identifiers (must match real_CIs keys)
            Values: Lists of synthetic answers/responses (numeric values)
        indices (list): List of question IDs to evaluate.
            Only questions in this list will be included in the miscoverage calculation.
            All IDs must exist in both real_CIs and synthetic_data.
        k (int): Number of synthetic answers to use for computing synthetic CI.
            Only the first k answers from each question will be used.
            If k > len(synthetic_data[i]) for any question, uses all available answers
            and the actual sample size in calculations (see synthetic_CI for details).
        alpha (float): Significance level for synthetic CI (between 0 and 1).
            The synthetic CI will have confidence level (1-alpha).
        C (float, optional): Scaling constant for synthetic CI half-width.
            Defaults to 2. The margin of error is multiplied by sqrt(C).
        type (str, optional): Type of coverage test. Defaults to 'general'.
            - 'general': Confidence set inclusion test.
                Coverage if: real_CI_lower >= synth_CI_lower AND real_CI_upper <= synth_CI_upper
                This ensures the entire real CI is contained within the synthetic CI.
            - 'simple': Empirical mean inclusion test.
                Coverage if: synth_CI_lower <= empirical_mean <= synth_CI_upper
                This only requires the empirical mean from the real data to be within the synthetic CI.
        k_min (int, optional): Minimum k for valid synthetic CI.
            Defaults to 2. Must be at least 2. Passed to synthetic_CI function.
        full_param_CI (list, optional): Full parameter space bounds.
            Defaults to [0, 1]. Passed to synthetic_CI function.
        CI_type (str, optional): Type of confidence interval to compute.
            Defaults to 'clt'.
            - 'clt': Central Limit Theorem-based confidence interval.
            - 'hoeffding': Hoeffding's inequality-based confidence interval.
            - 'bernstein': Bernstein's inequality-based confidence interval.
        synthetic_stats (dict, optional): Precomputed prefix statistics keyed by question ID.
            Allows reusing synthetic CI calculations across many k values without repeated slicing.
    Returns:
        tuple: A tuple of two elements:
            - miscoverage_rate (float): Proportion of questions with miscoverage.
                Range: [0, 1]. Lower values indicate better coverage.
                Formula: (number of questions with miscoverage) / (total number of questions)
            - synth_CI_widths (list): List of synthetic CI half-widths for each question.
                Length equals len(indices). Each element is (upper - lower) / 2.
                Used for analyzing the precision of synthetic CIs.
    
    Note:
        - Miscoverage rate = 1 - coverage rate
        - Synthetic CI half-widths are computed as half-widths (radius) of the intervals
    """
    # ========================================================================
    # INITIALIZE VARIABLES
    # ========================================================================
    m = len(indices)  # Number of questions to evaluate
    # Handle edge case: empty indices list
    if m == 0:
        raise ValueError("indices list cannot be empty. At least one question must be provided for evaluation.")
    coverage = 0  # Counter for questions with successful coverage
    synth_CI_widths = []  # List to store CI half-widths for each question
    
    # ========================================================================
    # EVALUATE COVERAGE FOR EACH QUESTION
    # ========================================================================
    for i in indices:
        # Get real CI for this question: [mean, lower_bound, upper_bound]
        real_CI = real_CIs[i]
        # Get synthetic answers for this question
        synth_data_i = synthetic_data[i]
        stats_i = None
        if synthetic_stats is not None:
            stats_i = synthetic_stats.get(i)
        # Compute synthetic CI using first k answers
        synth_CI = synthetic_CI(
            synth_data_i,
            k,
            alpha,
            C,
            k_min,
            full_param_CI,
            CI_type,
            precomputed_stats=stats_i
        )
        # synth_CI format: [mean, lower_bound, upper_bound]
        
        # ====================================================================
        # CHECK COVERAGE BASED ON TYPE
        # ====================================================================
        if type == 'general':
            # General type: Real CI must be completely within synthetic CI
            # Coverage if: real_CI_lower >= synth_CI_lower AND real_CI_upper <= synth_CI_upper
            # This ensures the entire real confidence interval is contained
            if real_CI[1] >= synth_CI[1] and real_CI[2] <= synth_CI[2]:
                coverage += 1
        elif type == 'simple':
            # Simple type: Only real mean must be within synthetic CI
            # Coverage if: synth_CI_lower <= empirical_mean <= synth_CI_upper
            if synth_CI[1] <= real_CI[0] <= synth_CI[2]:
                coverage += 1
        
        # ====================================================================
        # STORE SYNTHETIC CI HALF-WIDTH
        # ====================================================================
        # Compute half-width (radius) of synthetic CI
        # Half-width = (upper_bound - lower_bound) / 2
        synth_CI_widths.append((synth_CI[2] - synth_CI[1]) / 2)
    
    # ========================================================================
    # COMPUTE MISCOVERAGE RATE
    # ========================================================================
    # Miscoverage rate = proportion of questions with miscoverage
    # = (total questions - questions with coverage) / total questions
    miscoverage_rate = (m - coverage) / m
    
    return miscoverage_rate, synth_CI_widths

def find_k_hat(
    ks: np.ndarray,
    Gks: np.ndarray,
    threshold: float
) -> int:
    """
    Find the largest k such that the miscoverage rate is below the threshold.
    
    Performs threshold crossing analysis to find the optimal sample size k. Returns
    the largest k where Gks[k] <= threshold. If all k values meet threshold, returns
    the largest available k. If no k meets threshold, returns 0.
    
    Args:
        ks (np.ndarray): Array of k values (sample sizes) to evaluate.
            Typically an array like [1, 2, 3, ..., k_max].
            Must be the same length as Gks.
        Gks (np.ndarray): Array of miscoverage rates corresponding to each k.
            Gks[i] is the miscoverage rate when using ks[i] synthetic answers.
            Must be the same length as ks. Should be roughly increasing.
        threshold (float): Threshold for acceptable miscoverage rate.
            Must be between 0 and 1. Typically alpha*gamma (for 'general' type)
            or alpha/2 (for 'simple' type).
    
    Returns:
        int: The largest k value where Gks <= threshold.
            - If all Gks <= threshold: Returns ks[-1] (largest available k)
            - If first element > threshold: Returns 0 (no valid k exists)
            - Otherwise: Returns ks[ind_list[0] - 1], where ind_list[0] is the
              first index where Gks > threshold
    
    Algorithm:
        1. Find all indices where Gks > threshold
        2. If none exist: All k values meet threshold → return ks[-1] (largest available k)
        3. If first index is 0: Even k=1 exceeds threshold → return 0 (no valid k)
        4. Otherwise: Return the k value just before the first crossing
    """
    # ========================================================================
    # FIND INDICES WHERE MISCOVERAGE RATE EXCEEDS THRESHOLD
    # ========================================================================
    # Find all indices where Gks > threshold (upcrossing points)
    ind_list = np.where(Gks > threshold)[0]
    
    # ========================================================================
    # HANDLE SPECIAL CASES
    # ========================================================================
    if len(ind_list) == 0:
        # All Gks <= threshold: all k values meet the requirement
        # Return ks[-1] (largest available k)
        return ks[-1]
    elif ind_list[0] == 0:
        # First element already exceeds threshold: even k=1 doesn't meet requirement
        # Return 0 to signal that no valid k exists
        return 0
    else:
        # ====================================================================
        # FIND LARGEST K BEFORE THRESHOLD CROSSING
        # ====================================================================
        # ind_list[0] is the first index where Gks > threshold (upcrossing point)
        # ind_list[0] - 1 is the last index where Gks <= threshold
        # Return the k value at that index (largest k that still meets threshold)
        return ks[ind_list[0] - 1]

def one_split_run(
    real_answers: dict,
    synthetic_answers: dict,
    questions_id: list,
    alpha: float,
    gamma: float,
    k_max: int,
    C: float,
    train_proportion: float,
    type: str,
    k_min: int = 2,
    full_param_CI: list = [0, 1],
    CI_type: str = 'clt',
    seed: int = 0,
    report: bool = True,
    real_CIs: dict | None = None,
    synthetic_stats: dict | None = None
) -> dict:
    """
    Perform a single train-test split and find optimal k_hat for synthetic responses.
    
    Pipeline: (1) Compute real CIs, (2) Split into train/test, (3) Evaluate miscoverage
    rates for k=1..k_max, (4) Find optimal k_hat on training set, (5) Evaluate test
    performance using k_hat from training set, (6) Optionally plot and report results.
    
    Args:
        real_answers (dict): Dictionary mapping question IDs to real answer lists.
            Keys: Question identifiers (strings or integers)
            Values: Lists of real answers/responses (numeric values)
        synthetic_answers (dict): Dictionary mapping question IDs to synthetic answer lists.
            Keys: Question identifiers (must match real_answers keys)
            Values: Lists of synthetic answers/responses (numeric values)
            Must have the same structure as real_answers.
        questions_id (list): List of question IDs to include in the evaluation.
            All IDs must exist in both real_answers and synthetic_answers.
        alpha (float): Significance level for synthetic CI (between 0 and 1).
            The synthetic CI will have confidence level (1-alpha).
        gamma (float): Coverage probability for real CI (between 0 and 1).
        k_max (int): Maximum number of synthetic answers to evaluate.
            Evaluates k values from 1 to k_max (inclusive).
            If k_max > len(synthetic_answers[i]) for any question, uses all available
            answers and the actual sample size in calculations (see synthetic_CI for details).
        C (float): Scaling constant for synthetic CI half-width.
        train_proportion (float): Proportion of questions to use for training.
            Must be between 0 and 1. Remaining questions are used for testing.
        type (str): Type of coverage test. Must be 'general' or 'simple'.
            - 'general': Confidence set inclusion test
            - 'simple': Empirical mean inclusion test
        k_min (int, optional): Minimum k for valid synthetic CI.
            Defaults to 2. Must be at least 2. Passed to synthetic_CI and G functions.
        full_param_CI (list, optional): Full parameter space bounds.
            Defaults to [0, 1]. Passed to synthetic_CI and G functions.
        CI_type (str, optional): Type of confidence interval to compute.
            Defaults to 'clt'.
            - 'clt': Central Limit Theorem-based confidence interval.
            - 'hoeffding': Hoeffding's inequality-based confidence interval.
            - 'bernstein': Bernstein's inequality-based confidence interval.
        seed (int, optional): Random seed for train-test split.
            Defaults to 0. Ensures reproducibility of the split.
        report (bool, optional): Whether to print results and plot graph.
            Defaults to True. If False, suppresses output and plotting.
        real_CIs (dict, optional): Precomputed real confidence intervals keyed by question_id.
            Saves redundant CI computations when running many splits.
        synthetic_stats (dict, optional): Precomputed prefix statistics for synthetic answers.
            Allows O(1) computation of means/variances for any prefix length.
    
    Returns:
        dict: A dictionary with the following keys:
            - 'k_hat_train' (int): Optimal k found on training set.
                Largest k where train miscoverage rate <= threshold.
            - 'synth_CI_widths' (np.ndarray): Array of mean synthetic CI half-widths for each k.
                Length: k_max. Element i is the mean CI half-width when using k=i+1.
            - 'k_hat_test' (int): Optimal k found on test set (for comparison).
                Largest k where test miscoverage rate <= threshold.
            - 'test_G' (float): Test miscoverage rate when using k_hat_train.
            - 'test_synth_CI_widths' (list): List of test synthetic CI half-widths.
                Length equals number of test questions.
            - 'Gks_train' (list): List of train miscoverage rates for each k (k=1 to k_max).
                If k > len(synthetic_answers[i]) for any question, uses all available answers and the actual sample size in calculations,
                and Gks_train after the actual sample size is reached becomes constant.
            - 'Gks_test' (list): List of test miscoverage rates for each k (k=1 to k_max).
                If k > len(synthetic_answers[i]) for any question, uses all available answers and the actual sample size in calculations,
                and Gks_test after the actual sample size is reached becomes constant.
    
    Threshold Calculation:
        - For 'general' type: threshold = alpha * gamma
        - For 'simple' type: threshold = alpha / 2
    """
    # ========================================================================
    # INITIALIZE K VALUES
    # ========================================================================
    # Create array of k values to evaluate: [1, 2, 3, ..., k_max]
    ks = np.arange(1, k_max + 1)
    
    # ========================================================================
    # COMPUTE REAL CONFIDENCE INTERVALS
    # ========================================================================
    # Calculate real CI for each question using real answers
    # Real CI format: [mean, lower_bound, upper_bound]
    if real_CIs is None:
        real_CIs = _precompute_real_CIs(real_answers, questions_id, gamma)
    if synthetic_stats is None:
        synthetic_stats = _precompute_synthetic_stats(synthetic_answers, questions_id)

    
    # ========================================================================
    # SPLIT QUESTIONS INTO TRAIN AND TEST SETS
    # ========================================================================
    # Set random seed for reproducibility
    random.seed(seed)
    # Create a deep copy to avoid modifying the original list
    random_questions_id = copy.deepcopy(questions_id)
    # Shuffle questions randomly
    random.shuffle(random_questions_id)
    # Calculate split index
    train_test_split_index = int(train_proportion * len(random_questions_id))
    # Split into train and test sets
    train_questions = random_questions_id[:train_test_split_index]
    test_questions = random_questions_id[train_test_split_index:]
    
    # ========================================================================
    # EVALUATE MISCOVERAGE RATES FOR DIFFERENT K VALUES
    # ========================================================================
    # Initialize arrays to store miscoverage rates and CI half-widths
    Gks_train = np.zeros(len(ks))  # Train miscoverage rates for each k
    Gks_test = np.zeros(len(ks))   # Test miscoverage rates for each k
    synth_CI_widths = np.zeros(len(ks))  # Mean synthetic CI half-widths for each k
    
    # Evaluate miscoverage rate for each k value
    for i in range(len(ks)):
        k = ks[i]
        # Compute train miscoverage rate and CI half-widths
        # G() returns (miscoverage_rate, list_of_CI_half_widths_per_question)
        Gks_train[i], widths_list_train = G(
            real_CIs, synthetic_answers, train_questions, k, alpha, C, type, 
            k_min, full_param_CI, CI_type, synthetic_stats=synthetic_stats
        )
        # Store mean CI half-width across train questions for this k
        synth_CI_widths[i] = np.mean(widths_list_train)
        # Compute test miscoverage rate (CI half-widths not needed for test set here)
        Gks_test[i], _ = G(
            real_CIs, synthetic_answers, test_questions, k, alpha, C, type, 
            k_min, full_param_CI, CI_type, synthetic_stats=synthetic_stats
        )
    
    # ========================================================================
    # DETERMINE THRESHOLD AND FIND OPTIMAL K_HAT
    # ========================================================================
    # Calculate threshold based on coverage type
    if type == 'general':
        # General type: threshold = alpha * gamma
        threshold = alpha * gamma
    elif type == 'simple':
        # Simple type: threshold = alpha / 2
        threshold = alpha / 2
    
    # Find optimal k_hat on training set (used for final evaluation)
    k_hat_train = find_k_hat(ks, Gks_train, threshold)
    # Find optimal k_hat on test set (for comparison only)
    k_hat_test = find_k_hat(ks, Gks_test, threshold)
    
    # ========================================================================
    # EVALUATE TEST PERFORMANCE USING K_HAT_TRAIN
    # ========================================================================
    # Compute test miscoverage rate using k_hat_train (not k_hat_test)
    test_G, test_synth_CI_widths = G(
        real_CIs, synthetic_answers, test_questions, k_hat_train, alpha, C, type, 
        k_min, full_param_CI, CI_type, synthetic_stats=synthetic_stats
    )
    
    # ========================================================================
    # REPORT RESULTS (IF REQUESTED)
    # ========================================================================
    if report:
        # Plot train miscoverage rate vs k
        plt.plot(ks, Gks_train, label='train miscoverage rate')
        plt.xlabel('$k$')
        plt.ylabel('train miscoverage rate')
        # Add threshold line
        plt.hlines(threshold, ks[0], ks[-1], colors='r', linestyles='dashed', label='threshold')
        plt.legend(framealpha=0.3)
        # Print summary statistics
        print(f'threshold: {threshold}')
        print(f'k_hat_train: {k_hat_train}')
        print(f'k_hat_test: {k_hat_test}')
        print(f'synthetic CI half-width: mean {np.mean(synth_CI_widths)}, std {np.std(synth_CI_widths)}')
        print(f'test synthetic CI half-width: mean {np.mean(test_synth_CI_widths)}, std {np.std(test_synth_CI_widths)}')
        print(f'test miscoverage rate: {test_G}')
    
    result = {
        'k_hat_train': k_hat_train,
        'synth_CI_widths': synth_CI_widths,
        'k_hat_test': k_hat_test,
        'test_G': test_G,
        'test_synth_CI_widths': test_synth_CI_widths,
        'Gks_train': Gks_train.tolist(),
        'Gks_test': Gks_test.tolist()
    }

    return result

def multiple_split_run(
    real_answers: dict,
    synthetic_answers: dict,
    questions_id: list,
    alpha: float,
    gamma: float,
    k_max: int,
    C: float,
    train_proportion: float,
    type: str,
    k_min: int = 2,
    full_param_CI: list = [0, 1],
    CI_type: str = 'clt',
    seed: int = 0,
    num_splits: int = 100,
    report: bool = True
) -> tuple:
    """
    Perform multiple train-test splits and compute statistics of evaluation metrics.
    
    Runs one_split_run multiple times with different random seeds to assess variability.
    Computes mean and standard deviation of k_hat, synthetic CI half-widths, and test
    miscoverage rates across multiple splits.
    
    Args:
        real_answers (dict): Dictionary mapping question IDs to real answer lists.
            See one_split_run for details.
        synthetic_answers (dict): Dictionary mapping question IDs to synthetic answer lists.
            See one_split_run for details.
        questions_id (list): List of question IDs to evaluate.
            See one_split_run for details.
        alpha (float): Significance level for synthetic CI.
            See one_split_run for details.
        gamma (float): Confidence level for real CI.
            See one_split_run for details.
        k_max (int): Maximum number of synthetic answers to evaluate.
            See one_split_run for details.
            If k_max exceeds available answers for some questions, uses all available
            answers and actual sample size in calculations.
        C (float): Scaling constant for synthetic CI.
            See one_split_run for details.
        train_proportion (float): Proportion of questions for training.
            See one_split_run for details.
        type (str): Type of coverage test ('general' or 'simple').
            See one_split_run for details.
        k_min (int, optional): Minimum k for valid synthetic CI. Defaults to 2. Must be at least 2.
        full_param_CI (list, optional): Full parameter space bounds. Defaults to [0, 1].
        CI_type (str, optional): Type of confidence interval to compute.
            Defaults to 'clt'.
            - 'clt': Central Limit Theorem-based confidence interval.
            - 'hoeffding': Hoeffding's inequality-based confidence interval.
            - 'bernstein': Bernstein's inequality-based confidence interval.
        seed (int, optional): Random seed for generating split seeds. Defaults to 0.
        num_splits (int, optional): Number of train-test splits to perform.
            Defaults to 100. More splits provide more reliable statistics but take longer.
        report (bool, optional): Whether to compute and return statistics.
            Defaults to True. If False, returns only raw results.
    
    Returns:
        tuple or pd.DataFrame: 
            If report=True: Returns (results, df_report) where:
                - results (pd.DataFrame): DataFrame with one row per split, columns:
                    - 'k_hat_train': Optimal k on training set
                    - 'synth_CI_widths_mean': Mean synthetic CI half-width across k values
                    - 'k_hat_test': Optimal k on test set
                    - 'test_miscov_rate': Test miscoverage rate
                    - 'test_synth_CI_widths_mean': Mean test synthetic CI half-width
                - df_report (pd.DataFrame): Summary statistics (mean and std) for each metric
            If report=False: Returns only results DataFrame
    
    Note:
        - Each split uses a different random seed for train-test splitting
        - Statistics are computed across all splits to assess variability
        - Standard deviations indicate the reliability of the metrics
    """
    # ========================================================================
    # GENERATE RANDOM SEEDS FOR EACH SPLIT
    # ========================================================================
    # Set random seed for reproducibility
    np.random.seed(seed)
    # Generate random seeds for each split (0 to 9999)
    split_seeds = np.random.randint(0, 10000, num_splits)
    split_seeds = [int(s) for s in split_seeds]
    
    # ========================================================================
    # INITIALIZE RESULTS DATAFRAME
    # ========================================================================
    # Create DataFrame to store results for each split
    results = pd.DataFrame(columns=[
        'k_hat_train', 'synth_CI_widths_mean', 'k_hat_test', 
        'test_miscov_rate', 'test_synth_CI_widths_mean', 'Gks_train', 'Gks_test'
    ])
    
    # ========================================================================
    # PRECOMPUTE SHARED STATISTICS
    # ========================================================================
    real_CIs = _precompute_real_CIs(real_answers, questions_id, gamma)
    synthetic_stats = _precompute_synthetic_stats(synthetic_answers, questions_id)

    # ========================================================================
    # RUN ONE_SPLIT_RUN FOR EACH SPLIT
    # ========================================================================
    for i in tqdm(range(num_splits), desc='Running splits'):
        # Run one_split_run with report=False to suppress individual outputs
        result = one_split_run(
            real_answers, synthetic_answers, questions_id, alpha, gamma, k_max, C, 
            train_proportion, type, k_min, full_param_CI, CI_type, split_seeds[i], False,
            real_CIs=real_CIs, synthetic_stats=synthetic_stats
        )
        # Store results for this split
        results.loc[i] = [
            result['k_hat_train'],
            np.mean(result['synth_CI_widths']),  # Mean CI half-width across all k values, not very useful
            result['k_hat_test'],
            result['test_G'],
            np.mean(result['test_synth_CI_widths']),  # Mean test CI half-width
            result['Gks_train'],
            result['Gks_test']
        ]
    
    # ========================================================================
    # COMPUTE STATISTICS (IF REQUESTED)
    # ========================================================================
    if report:
        # Compute mean and standard deviation for each metric
        statistics = [
            [np.mean(results['k_hat_train']), np.std(results['k_hat_train'])],
            [np.mean(results['synth_CI_widths_mean']), np.std(results['synth_CI_widths_mean'])],
            [np.mean(results['k_hat_test']), np.std(results['k_hat_test'])],
            [np.mean(results['test_miscov_rate']), np.std(results['test_miscov_rate'])],
            [np.mean(results['test_synth_CI_widths_mean']), np.std(results['test_synth_CI_widths_mean'])]
        ]
        # Create DataFrame with statistics
        df_report = pd.DataFrame(
            statistics,
            index=['k_hat_train', 'synth_CI_widths_mean', 'k_hat_test', 'test_miscov_rate', 'test_synth_CI_widths_mean'],
            columns=['mean', 'std']
        )
        return results, df_report
    else:
        return results

def get_reports(
    real_answers: dict,
    synthetic_answers: dict,
    questions_id: list,
    alphas: list,
    gamma: float,
    k_max: int,
    C: float,
    train_proportion: float,
    k_min: int = 2,
    full_param_CI: list = [0, 1],
    CI_type: str = 'clt',
    seed: int = 0,
    num_splits: int = 100,
    types: list | tuple | str = ('general', 'simple'),
) -> tuple:
    """
    Generate evaluation reports for multiple alpha values and both coverage types.
    
    Runs multiple_split_run for each alpha value and both 'general' and 'simple' types.
    Returns dictionaries of reports keyed by alpha for visualization and comparison.
    
    Args:
        real_answers (dict): Dictionary mapping question IDs to real answer lists.
        synthetic_answers (dict): Dictionary mapping question IDs to synthetic answer lists.
        questions_id (list): List of question IDs to evaluate.
        alphas (list): List of significance levels to evaluate.
            Example: [0.01, 0.05, 0.10] for 99%, 95%, 90% confidence levels.
        gamma (float): Coverage probability for real CI.
        k_max (int): Maximum number of synthetic answers to evaluate.
            If k_max exceeds available answers for some questions, uses all available
            answers and actual sample size in calculations (see synthetic_CI for details).
        C (float): Scaling constant for synthetic CI.
        train_proportion (float): Proportion of questions for training.
        k_min (int, optional): Minimum k for valid synthetic CI. Defaults to 2. Must be at least 2.
        full_param_CI (list, optional): Full parameter space bounds. Defaults to [0, 1].
        CI_type (str, optional): Type of confidence interval to compute.
            Defaults to 'clt'.
            - 'clt': Central Limit Theorem-based confidence interval.
            - 'hoeffding': Hoeffding's inequality-based confidence interval (modified with sqrt(C) factor).
            - 'bernstein': Bernstein's inequality-based confidence interval.
        seed (int, optional): Random seed. Defaults to 0.
        num_splits (int, optional): Number of train-test splits. Defaults to 100.
    
    Returns:
        dict: Keys are included only for the requested coverage types. For each type ('general' or 'simple'),
        the dictionary contains:
            - 'reports_{type}': alpha-keyed dicts of summary statistics (mean/std).
            - 'Gks_train_{type}': alpha-keyed lists of per-split train miscoverage trajectories.
            - 'Gks_test_{type}': alpha-keyed lists of per-split test miscoverage trajectories.
            - 'k_hat_train_{type}': alpha-keyed lists of train k_hat values (one per split).
            - 'k_hat_test_{type}': alpha-keyed lists of test k_hat values (one per split).
    """
    type_storage = {
        method_type: {
            'reports': {},
            'Gks_train': {},
            'Gks_test': {},
            'k_hat_train': {},
            'k_hat_test': {}
        }
        for method_type in types
    }
    # ========================================================================
    # EVALUATE EACH ALPHA VALUE
    # ========================================================================
    for alpha in alphas:
        print(f'-Evaluating alpha={alpha}...')
        for method_type in types:
            print(f'--Running evaluation for {method_type} method...')
            results_df, report_df = multiple_split_run(
                real_answers, synthetic_answers, questions_id, alpha, gamma, k_max, C,
                train_proportion, type=method_type, k_min=k_min, full_param_CI=full_param_CI,
                CI_type=CI_type, seed=seed, num_splits=num_splits, report=True
            )
            storage = type_storage[method_type]
            storage['Gks_train'][alpha] = results_df['Gks_train'].tolist()
            storage['Gks_test'][alpha] = results_df['Gks_test'].tolist()
            storage['k_hat_train'][alpha] = [int(k) for k in results_df['k_hat_train'].tolist()]
            storage['k_hat_test'][alpha] = [int(k) for k in results_df['k_hat_test'].tolist()]
            storage['reports'][alpha] = report_df.to_dict(orient='index')
        print(f'-alpha={alpha} evaluated successfully.')

    result = {}
    for method_type, storage in type_storage.items():
        suffix = 'general' if method_type == 'general' else 'simple'
        result[f'reports_{suffix}'] = storage['reports']
        result[f'Gks_train_{suffix}'] = storage['Gks_train']
        result[f'Gks_test_{suffix}'] = storage['Gks_test']
        result[f'k_hat_train_{suffix}'] = storage['k_hat_train']
        result[f'k_hat_test_{suffix}'] = storage['k_hat_test']
    return result

def get_reports_multiple(
    real_answers: dict,
    synthetic_answers_all: dict,
    questions_id: list,
    alphas: list,
    gamma: float,
    k_max: int,
    C: float,
    train_proportion: float,
    k_min: int = 2,
    full_param_CI: list = [0, 1],
    CI_type: str = 'clt',
    seed: int = 0,
    num_splits: int = 100,
    types: list | tuple | str = ('general', 'simple')
) -> tuple:
    """
    Generate evaluation reports for multiple LLM models.
    
    Evaluates multiple models by running get_reports for each model's synthetic answers.
    Returns dictionaries of reports keyed by model name for comparison.
    
    Args:
        real_answers (dict): Dictionary mapping question IDs to real answer lists.
        synthetic_answers_all (dict): Dictionary mapping model names to synthetic answer dictionaries.
            Keys: Model names (e.g., 'gpt-4o', 'claude-3.5-haiku')
            Values: Dictionaries with same structure as real_answers
            Example: {'gpt-4o': {'q1': [1,0,1,...], ...}, 'claude': {...}}
        questions_id (list): List of question IDs to evaluate.
        alphas (list): List of significance levels to evaluate.
        gamma (float): Coverage probability for real CI.
        k_max (int): Maximum number of synthetic answers to evaluate.
            If k_max exceeds available answers for some questions, uses all available
            answers and actual sample size in calculations (see synthetic_CI for details).
        C (float): Scaling constant for synthetic CI.
        train_proportion (float): Proportion of questions for training.
        k_min (int, optional): Minimum k for valid synthetic CI. Defaults to 2. Must be at least 2.
        full_param_CI (list, optional): Full parameter space bounds. Defaults to [0, 1].
        CI_type (str, optional): Type of confidence interval to compute.
            Defaults to 'clt'.
            - 'clt': Central Limit Theorem-based confidence interval.
            - 'hoeffding': Hoeffding's inequality-based confidence interval (modified with sqrt(C) factor).
            - 'bernstein': Bernstein's inequality-based confidence interval.
        seed (int, optional): Random seed. Defaults to 0.
        num_splits (int, optional): Number of train-test splits. Defaults to 100.
    
    Returns:
        tuple: (reports_all_by_type, for_sharpness_analysis) where:
            - reports_all_by_type (dict): Keys are coverage types; values are dictionaries mapping
                model names to alpha-keyed report dictionaries (mean/std metrics).
            - for_sharpness_analysis (dict): Keys are coverage types; values are dictionaries with
                entries 'Gks_train', 'Gks_test', 'k_hat_train', 'k_hat_test'. Each entry maps model
                names to alpha-keyed lists containing the raw per-split trajectories.
    """
    keys = list(synthetic_answers_all.keys())
    reports_all_by_type = {method_type: {} for method_type in types}
    sharpness_by_type = {
        method_type: {
            'Gks_train': {},
            'Gks_test': {},
            'k_hat_train': {},
            'k_hat_test': {}
        }
        for method_type in types
    }
    # ========================================================================
    # EVALUATE EACH MODEL
    # ========================================================================
    for key in keys:
        print(f'Evaluating model {key}...')
        synthetic_answers = synthetic_answers_all[key]
        # Get reports for this model
        result = get_reports(
            real_answers, synthetic_answers, questions_id, alphas, gamma, k_max, C,
            train_proportion, k_min=k_min, full_param_CI=full_param_CI,
            CI_type=CI_type,
            seed=seed, num_splits=num_splits, types=types
        )
        for method_type in types:
            suffix = 'general' if method_type == 'general' else 'simple'
            reports_all_by_type[method_type][key] = result[f'reports_{suffix}']
            sharpness_by_type[method_type]['Gks_train'][key] = result[f'Gks_train_{suffix}']
            sharpness_by_type[method_type]['Gks_test'][key] = result[f'Gks_test_{suffix}']
            sharpness_by_type[method_type]['k_hat_train'][key] = result[f'k_hat_train_{suffix}']
            sharpness_by_type[method_type]['k_hat_test'][key] = result[f'k_hat_test_{suffix}']
        print(f'Model {key} evaluated successfully.\n')
    return reports_all_by_type, sharpness_by_type

def plot_reports(
    reports_all: dict,
    num_splits: int,
    alphas: list,
    gamma: float,
    C: float,
    metric: str = 'test_miscov_rate',
    type: str = 'general',
    save_info: dict = {
        'dataset_name': 'EEDI',
        'save_folder_name': 'evaluation_results'
    }
) -> None:
    """
    Plot evaluation metrics across different alpha values for multiple models.
    
    Creates error bar plots comparing different LLM models across significance levels.
    Supports multiple metrics and automatically scales values based on coverage type.
    
    Args:
        reports_all (dict): Dictionary mapping model names to lists of report DataFrames.
            Keys: Model names (e.g., 'gpt-4o', 'claude-3.5-haiku')
            Values: Lists of report DataFrames, one per alpha value
            Each DataFrame has 'mean' and 'std' columns for evaluation metrics
        num_splits (int): Number of train-test splits used to generate reports.
            Used to compute standard errors for error bars.
        alphas (list): List of significance levels (x-axis values).
            Must match the order of reports in reports_all values.
        gamma (float): Coverage probability for real CI.
            Used to compute multiplier for scaling metrics.
        C (float): Scaling constant for synthetic CI.
        metric (str, optional): Metric to plot. Defaults to 'test_miscov_rate'.
            Options:
            - 'kappa_hat': Optimal k value (train k_hat / C)
            - 'synth_CI_width': Test synthetic CI half-width (mean half-width)
            - 'test_miscov_rate': Test miscoverage rate (scaled by multiplier)
        type (str, optional): Coverage type ('general' or 'simple').
            Defaults to 'general'. Must match the type of reports_all.
            Determines the multiplier for scaling metrics:
            - 'general': multiplier = gamma
            - 'simple': multiplier = 1/2
        save_info (dict, optional): Dictionary containing the following keys:
            - 'dataset_name': Name of the dataset.
            - 'save_folder_name': Name of the folder to save the plot.
            Defaults to {'dataset_name': 'EEDI', 'save_folder_name': 'evaluation_results'}.
            The plot will be saved to ../data/{save_info['dataset_name']}/{save_info['save_folder_name']}/{type}/{metric}.pdf.
    
    Returns:
        None: Creates and displays a matplotlib plot and saves it to the specified folder.
    
    Note:
        - Error bars represent 95% confidence intervals (1.96 * SE)
        - For 'test_miscov_rate', adds a threshold line (y = alpha)
        - Uses LLM_COLORS for consistent model coloring
    """
    # ========================================================================
    # DETERMINE MULTIPLIER BASED ON COVERAGE TYPE
    # ========================================================================
    means_all, errs_all = {}, {}
    if type == 'general':
        multiplier = gamma  # Multiplier for 'general' type
    elif type == 'simple':
        multiplier = 1 / 2  # Multiplier for 'simple' type
    
    # ========================================================================
    # EXTRACT MEANS AND ERROR BARS FOR EACH MODEL
    # ========================================================================
    for key in reports_all.keys():
        reports_raw = reports_all[key]
        # Validate reports_raw structure
        # reports_raw is a dict keyed by alpha values: {alpha1: {...}, alpha2: {...}, ...}
        if not isinstance(reports_raw, dict):
            raise ValueError(f"reports_raw for model '{key}' must be a dictionary keyed by alpha values")
        if len(reports_raw) != len(alphas):
            raise ValueError(f"Number of reports ({len(reports_raw)}) for model '{key}' does not match number of alphas ({len(alphas)})")
        
        # Convert dict to list in the order of alphas
        # Each report is a dict with structure: {'k_hat_train': {'mean': ..., 'std': ...}, ...}
        # Note: JSON keys might be strings, so we need to handle both float and string alpha values
        reports = []
        for alpha_val in alphas:
            # Try float first, then string
            alpha_key = alpha_val
            if alpha_key not in reports_raw:
                alpha_key = str(alpha_val)
            if alpha_key not in reports_raw:
                raise ValueError(f"Alpha value {alpha_val} (or '{alpha_key}') not found in reports for model '{key}'. Available alphas: {list(reports_raw.keys())}")
            report = reports_raw[alpha_key]
            # Reconstruct DataFrame from dict (orient='index' means dict keys are index names)
            # This creates a DataFrame with index as the outer keys and columns as 'mean' and 'std'
            reports.append(pd.DataFrame.from_dict(report, orient='index'))
        # Extract metric values and compute error bars (95% CI half-widths)
        if metric == 'kappa_hat':
            # Extract k_hat_train (index 'k_hat_train')
            means = np.array([reports[i].loc['k_hat_train', 'mean'] for i in range(len(reports))]) / C
            # Standard error = std / sqrt(n), error bar = 1.96 * SE
            errs = np.array([1.96 / np.sqrt(num_splits) * reports[i].loc['k_hat_train', 'std'] for i in range(len(reports))]) / C
        elif metric == 'synth_CI_width':
            # Extract test synthetic CI half-width (index 'test_synth_CI_widths_mean')
            means = np.array([reports[i].loc['test_synth_CI_widths_mean', 'mean'] for i in range(len(reports))])
            errs = np.array([1.96 / np.sqrt(num_splits) * reports[i].loc['test_synth_CI_widths_mean', 'std'] for i in range(len(reports))])
        elif metric == 'test_miscov_rate':
            # Extract test miscoverage rate (index 'test_miscov_rate') and scale by multiplier
            means = np.array([reports[i].loc['test_miscov_rate', 'mean'] for i in range(len(reports))]) / multiplier
            errs = np.array([1.96 / np.sqrt(num_splits) * reports[i].loc['test_miscov_rate', 'std'] for i in range(len(reports))]) / multiplier
        means_all[key] = means
        errs_all[key] = errs

    # ========================================================================
    # AN EXTRA BAR PLOT FOR KAPPA HAT AT THE SMALLEST ALPHA
    # ========================================================================
    if metric == 'kappa_hat':
        print('Creating an extra bar plot for kappa_hat at the smallest alpha...')
        means = np.array([means_all[key][0] for key in reports_all.keys()])
        errs = np.array([errs_all[key][0] for key in reports_all.keys()])
        labels = [LLM_PLOT_INFO[key]['label'] for key in reports_all.keys()]
        colors = [LLM_PLOT_INFO[key]['color'] for key in reports_all.keys()]
        
        plt.figure(figsize=(6, 6))
        x_pos = np.arange(len(labels))
        bars = plt.bar(x_pos, means, yerr=errs, capsize=3, color=colors, alpha=1, linewidth=0.4, error_kw={'elinewidth': 0.75})
        xmin, xmax = plt.xlim()
        plt.hlines(y=means_all['random'][0], xmin=xmin, xmax=bars[-1].get_x() + bars[-1].get_width()/2, linestyles='--', linewidth=1, color='gray', zorder=0)
        plt.xlim(xmin, xmax)
        # plt.xlabel('Model', fontsize=16)
        # plt.ylabel(r'$\hat{\kappa}$', fontsize=18)
        plt.xticks(x_pos, labels, rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        
        save_dir = os.path.join('..', 'data', save_info['dataset_name'], save_info['save_folder_name'], type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{type}_kappa.pdf'), bbox_inches='tight')
        plt.show()

    # ========================================================================
    # CREATE PLOT
    # ========================================================================
    plt.figure(figsize=(6, 6))
    # Plot error bars for each model
    if metric != 'test_miscov_rate':
        for key in reports_all.keys():
            means = means_all[key]
            errs = errs_all[key]
            plt.errorbar(alphas, means, errs, capsize=3, markersize=5, linestyle='-',
                        marker=LLM_PLOT_INFO[key]['marker'], label=LLM_PLOT_INFO[key]['label'], color=LLM_PLOT_INFO[key]['color'])
        
        # Set labels
        plt.xlabel(r'$\alpha$', fontsize=16)
        plt.xticks(alphas)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(framealpha=0.3, loc='center left', bbox_to_anchor=(1.02, 0.5))
    elif metric == 'test_miscov_rate':
        # Horizontal errorbar plot: models on y-axis, alpha values on x-axis
        model_keys = list(reports_all.keys())
        y_positions = np.arange(len(model_keys))
        model_labels = [LLM_PLOT_INFO[key]['label'] for key in model_keys]
        model_colors = [LLM_PLOT_INFO[key]['color'] for key in model_keys]
        
        # Plot horizontal errorbars for each model at each alpha
        # Each model gets a row, and for each alpha we plot a horizontal errorbar
        # Use different markers for each alpha value
        markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
        for i, key in enumerate(model_keys):
            means = means_all[key]
            errs = errs_all[key]
            # Plot horizontal errorbars (xerr) for each alpha value separately with different markers
            for j, alpha_idx in enumerate(range(len(alphas))):
                marker = markers[j % len(markers)]  # Cycle through markers if more than 4 alphas
                plt.errorbar(means[alpha_idx], y_positions[i], xerr=errs[alpha_idx], 
                            fmt=marker, capsize=3, capthick=1.5, markersize=6,
                            color=model_colors[i], 
                            elinewidth=1.5, alpha=0.8, zorder=2)
            # Add label only once per model (for legend)
            plt.errorbar([], [], fmt='o', color=model_colors[i], label=model_labels[i], 
                        markersize=6, alpha=0.8)
        
        # Add vertical dashed lines at each alpha threshold
        y_min, y_max = -0.5, len(model_keys) - 0.5
        for alpha in alphas:
            plt.vlines(alpha, y_min, y_max, color=THRESHOLD_COLOR, linestyle='--', 
                      linewidth=1.5, alpha=0.6, zorder=0)
        
        # Set labels and ticks
        # plt.xlabel('test miscoverage probability', fontsize=16)
        plt.xticks(alphas, [f'{alpha:.2f}' for alpha in alphas], fontsize=12)
        # plt.ylabel('Model', fontsize=16)
        plt.yticks(y_positions, model_labels, fontsize=12)
        plt.ylim(y_min, y_max)
        plt.grid(axis='x', alpha=0.3, linestyle=':', zorder=0)
    
    # Save plot
    save_dir = os.path.join('..', 'data', save_info['dataset_name'], save_info['save_folder_name'], type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    metric_name_map = {'test_miscov_rate': 'miscoverage', 'kappa_hat': 'kappa_all', 'synth_CI_width': 'halfwidth'}
    plt.savefig(os.path.join(save_dir, f'{type}_{metric_name_map[metric]}.pdf'), bbox_inches='tight')
    plt.show()

def table_reports(
    reports_all: dict,
    num_splits: int,
    alphas: list,
    gamma: float,
    C: float,
    metric: str = 'test_miscov_rate',
    type: str = 'general',
    save_info: dict = {
        'dataset_name': 'EEDI',
        'save_folder_name': 'evaluation_results'
    }
) -> None:
    """
    Create a table of evaluation metrics across different alpha values for multiple models.
    
    Creates a pandas DataFrame table showing metric values (mean ± error) for each model
    and alpha value. Suitable for presentation in papers or reports.
    
    Args:
        reports_all (dict): Dictionary mapping model names to lists of report DataFrames.
            See plot_reports for details.
        num_splits (int): Number of train-test splits used to generate reports.
        alphas (list): List of significance levels (x-axis values).
        gamma (float): Coverage probability for real CI.
        C (float): Scaling constant for synthetic CI.
        metric (str, optional): Metric to display. Defaults to 'test_miscov_rate'.
            See plot_reports for available metrics.
        type (str, optional): Coverage type ('general' or 'simple'). Defaults to 'general'.
        save_info (dict, optional): Dictionary containing the following keys:
            - 'dataset_name': Name of the dataset.
            - 'save_folder_name': Name of the folder to save the table.
            Defaults to {'dataset_name': 'EEDI', 'save_folder_name': 'evaluation_results'}.
            The table will be saved to ../data/{save_info['dataset_name']}/{save_info['save_folder_name']}/{type}/{metric}.csv.
    
    Returns:
        None: Prints the table to the console and saves it to a CSV file.
    """
    # ========================================================================
    # DETERMINE MULTIPLIER BASED ON COVERAGE TYPE
    # ========================================================================
    if type == 'general':
        multiplier = gamma
    elif type == 'simple':
        multiplier = 1 / 2
    
    # ========================================================================
    # INITIALIZE TABLE
    # ========================================================================
    keys = list(reports_all.keys())
    table = pd.DataFrame(index=keys, columns=alphas)
    
    # ========================================================================
    # EXTRACT METRICS FOR EACH MODEL
    # ========================================================================
    for key in keys:
        reports_raw = reports_all[key]
        # Validate reports_raw structure
        # reports_raw is a dict keyed by alpha values: {alpha1: {...}, alpha2: {...}, ...}
        if not isinstance(reports_raw, dict):
            raise ValueError(f"reports_raw for model '{key}' must be a dictionary keyed by alpha values")
        if len(reports_raw) != len(alphas):
            raise ValueError(f"Number of reports ({len(reports_raw)}) for model '{key}' does not match number of alphas ({len(alphas)})")
        
        # Convert dict to list in the order of alphas
        # Each report is a dict with structure: {'k_hat_train': {'mean': ..., 'std': ...}, ...}
        # Note: JSON keys might be strings, so we need to handle both float and string alpha values
        reports = []
        for alpha_val in alphas:
            # Try float first, then string
            alpha_key = alpha_val
            if alpha_key not in reports_raw:
                alpha_key = str(alpha_val)
            if alpha_key not in reports_raw:
                raise ValueError(f"Alpha value {alpha_val} (or '{alpha_key}') not found in reports for model '{key}'. Available alphas: {list(reports_raw.keys())}")
            report = reports_raw[alpha_key]
            # Reconstruct DataFrame from dict (orient='index' means dict keys are index names)
            # This creates a DataFrame with index as the outer keys and columns as 'mean' and 'std'
            reports.append(pd.DataFrame.from_dict(report, orient='index'))
        # Extract metric values and compute error bars
        if metric == 'kappa_hat':
            means = np.array([reports[i].loc['k_hat_train', 'mean'] for i in range(len(reports))]) / C
            errs = np.array([1.96 / np.sqrt(num_splits) * reports[i].loc['k_hat_train', 'std'] for i in range(len(reports))]) / C
        elif metric == 'synth_CI_width':
            # Extract test synthetic CI half-width
            means = np.array([reports[i].loc['test_synth_CI_widths_mean', 'mean'] for i in range(len(reports))])
            errs = np.array([1.96 / np.sqrt(num_splits) * reports[i].loc['test_synth_CI_widths_mean', 'std'] for i in range(len(reports))])
        elif metric == 'test_miscov_rate':
            means = np.array([reports[i].loc['test_miscov_rate', 'mean'] for i in range(len(reports))]) / multiplier
            errs = np.array([reports[i].loc['test_miscov_rate', 'std'] / np.sqrt(num_splits) for i in range(len(reports))]) / multiplier
        
        # ====================================================================
        # FORMAT VALUES AS "mean ± error"
        # ====================================================================
        # Create formatted strings: "mean ± error" (rounded to 4 decimals)
        to_record = [
            str(np.round(means[i], 4)) + u"\u00B1" + str(np.round(errs[i], 4)) 
            for i in range(len(means))
        ]
        table.loc[key] = to_record
    
    # Save table
    save_dir = os.path.join('..', 'data', save_info['dataset_name'], save_info['save_folder_name'], type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    metric_name_map = {'test_miscov_rate': 'miscoverage', 'kappa_hat': 'kappa_all', 'synth_CI_width': 'halfwidth'}
    table.to_csv(os.path.join(save_dir, f'{type}_{metric_name_map[metric]}.csv'))

    # print table
    print(table.to_string())

def evaluations(
    dataset_name: str = 'EEDI', # or 'OpinionQA'
    models: list = ['claude-3.5-haiku', 'deepseek-v3', 'gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o', 'gpt-5-mini', 'llama-3.3-70B-instruct-turbo', 'mistral-7B-instruct-v0.3', 'random'],
    synthetic_answer_folder_name: str = 'synthetic_answers', 
    evaluation_results_folder_name: str = 'evaluation_results',
    alphas: list = [0.05, 0.10, 0.15, 0.20],
    gamma: float = 0.5,
    k_max: int = 100,
    C: float = 3,
    train_proportion: float = 0.6,
    k_min: int = 2,
    CI_type: str = 'clt',
    seed: int = 0,
    num_splits: int = 100,
    types: list = ['general', 'simple']
) -> None:
    """
    Evaluate all models on the specified dataset and save evaluation reports.
    
    Pipeline: (1) Load real and synthetic answers, (2) Find common question keys,
    (3) Run evaluation for all models across multiple alpha values, (4) Save results
    to JSON files. Saved reports can be used with plot_from_saved_evaluations() to
    generate visualizations without re-running the evaluation.
    
    Args:
        dataset_name (str, optional): Name of the dataset. Must be 'EEDI' or 'OpinionQA'.
            Defaults to 'EEDI'.
            - 'EEDI': Educational assessment dataset with binary correctness (0/1)
            - 'OpinionQA': Opinion polling dataset with numeric scores in [-1, 1]
        models (list, optional): List of model names to evaluate.
            Model names should match the file names in the synthetic_answers folder.
            Defaults to ['claude-3.5-haiku', 'deepseek-v3', 'gpt-3.5-turbo', 'gpt-4o-mini',
            'gpt-4o', 'gpt-5-mini', 'llama-3.3-70B-instruct-turbo', 'mistral-7B-instruct-v0.3', 'random'].
        synthetic_answer_folder_name (str, optional): Name of folder containing synthetic answers.
            Defaults to 'synthetic_answers'.
            The function looks for files at: ../data/{dataset_name}/{synthetic_answer_folder_name}/clean/{model}.json
        evaluation_results_folder_name (str, optional): Name of folder to save evaluation results.
            Defaults to 'evaluation_results'.
            Results are saved to: ../data/{dataset_name}/{evaluation_results_folder_name}/
        alphas (list, optional): List of significance levels to evaluate.
            Defaults to [0.05, 0.10, 0.15, 0.20].
            Each alpha corresponds to a confidence level of (1-alpha).
            Example: alpha=0.05 means 95% confidence level.
        gamma (float, optional): Coverage probability for real confidence intervals.
            Defaults to 0.5 (50% coverage).
            Used to compute confidence intervals for real survey responses using CLT.
        k_max (int, optional): Maximum number of synthetic answers to evaluate.
            Defaults to 100.
            The evaluation will test k values from 1 to k_max to find optimal k_hat.
            If k_max exceeds the number of available synthetic answers for some models
            or questions, the function will use all available answers and the actual
            sample size in CI calculations (see synthetic_CI for details).
        C (float, optional): Scaling constant for synthetic confidence intervals.
            Defaults to 3.
            The synthetic CI half-width is multiplied by sqrt(C) to provide conservative uncertainty quantification.
        train_proportion (float, optional): Proportion of questions to use for training.
            Defaults to 0.6 (60% training, 40% testing).
            Must be between 0 and 1. Used for train-test split to find optimal k_hat.
        k_min (int, optional): Minimum k value required for valid synthetic CI.
            Defaults to 2. Must be at least 2.
            If n <= k_min (where n is the actual number of data points used), the function
            returns a conservative interval covering the full parameter space.
        CI_type (str, optional): Type of confidence interval to compute.
            Defaults to 'clt'.
            - 'clt': Central Limit Theorem-based confidence interval.
            - 'hoeffding': Hoeffding's inequality-based confidence interval (modified with sqrt(C) factor).
            - 'bernstein': Bernstein's inequality-based confidence interval.
        seed (int, optional): Random seed for reproducibility.
            Defaults to 0.
            Used to generate random train-test splits.
        num_splits (int, optional): Number of train-test splits to perform.
            Defaults to 100.
            More splits provide more reliable statistics but take longer to compute.
        type (str, list, tuple, optional): Coverage type(s) to evaluate. Accepts 'general',
            'simple', 'both', or an iterable of types. Defaults to both.
    
    Returns:
        None: Results are saved to disk as JSON files under
        ../data/{dataset_name}/{evaluation_results_folder_name}/ with per-type subdirectories:
            - {type}/reports_all.json: Reports for the specified coverage type(s)
            - {type}/sharpness_analysis_all.json: Sharpness analysis data (per split)
        Each report contains mean and std statistics for evaluation metrics across splits.
    
    Raises:
        ValueError: 
            - If dataset_name is not 'EEDI' or 'OpinionQA'
            - If no common question keys are found between real and synthetic answers
            - If survey data structure is invalid
        FileNotFoundError: 
            - If data file (eedi_data.json or opinionqa_data.json) is not found
            - If synthetic answer files are not found for any model
    
    Note:
        - The function only saves reports to JSON files. To generate plots and tables,
          call plot_from_saved_evaluations() separately.
        - Results are saved to: ../data/{dataset_name}/{evaluation_results_folder_name}/
        - Evaluation results are saved in per-type subdirectories:
          * general/reports_all.json and general/sharpness_analysis_all.json (if evaluated)
          * simple/reports_all.json and simple/sharpness_analysis_all.json (if evaluated)
        - The function automatically finds the intersection of question keys across all models
          to ensure all models are evaluated on the same set of questions.
        - If some models are missing synthetic answers for certain questions, only the
          common questions will be used for evaluation.
    """
    # ========================================================================
    # SET FULL PARAMETER SPACE BOUNDS BASED ON DATASET
    # ========================================================================
    if dataset_name == 'EEDI':
        # EEDI: Binary correctness scores in [0, 1]
        full_param_CI = [0, 1]
    elif dataset_name == 'OpinionQA':
        # OpinionQA: Opinion scores in [-1, 1]
        full_param_CI = [-1, 1]
    else:
        raise ValueError(f'Invalid dataset name: {dataset_name}. Must be "EEDI" or "OpinionQA".')

    # ========================================================================
    # LOAD REAL ANSWERS FROM DATASET
    # ========================================================================
    # Determine the correct data file name based on dataset
    if dataset_name == 'EEDI':
        data_file = 'eedi_data.json'
    elif dataset_name == 'OpinionQA':
        data_file = 'opinionqa_data.json'
    else:
        raise ValueError(f'Invalid dataset name: {dataset_name}')
    
    data_path = os.path.join('..', 'data', dataset_name, data_file)
    # Load data - survey data is stored as list of records (from to_dict(orient='records'))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # ========================================================================
    # EXTRACT REAL ANSWERS FROM SURVEY DATA
    # ========================================================================
    # Extract real answers - survey is a list of dicts where each dict is a record
    # For EEDI: extract 'IsCorrect' from each record (binary: 0 or 1)
    # For OpinionQA: extract 'RESPONSE_NUMERIC' from each record (numeric score in [-1, 1])
    real_answers = {}
    for key, value in data.items():
        survey_data = value.get('survey', [])
        if not isinstance(survey_data, list):
            raise ValueError(f"Expected 'survey' to be a list of records, but got {type(survey_data)} for key {key}")
        
        if dataset_name == 'EEDI':
            # Extract 'IsCorrect' values from survey records (binary correctness)
            real_answers[key] = [record.get('IsCorrect', 0) for record in survey_data]
        elif dataset_name == 'OpinionQA':
            # Extract 'RESPONSE_NUMERIC' values from survey records (opinion scores)
            real_answers[key] = [record.get('RESPONSE_NUMERIC', 0) for record in survey_data]

    # ========================================================================
    # LOAD SYNTHETIC ANSWERS FOR ALL MODELS
    # ========================================================================
    synthetic_answers_all = {}
    for model in models:
        synthetic_answer_path = os.path.join('..', 'data', dataset_name, synthetic_answer_folder_name, 'clean', f'{model}.json')
        if not os.path.exists(synthetic_answer_path):
            raise FileNotFoundError(f"Synthetic answers file not found for model '{model}': {synthetic_answer_path}")
        with open(synthetic_answer_path, 'r') as f:
            synthetic_answers_all[model] = json.load(f)
    
    # ========================================================================
    # FIND COMMON QUESTION KEYS ACROSS ALL MODELS
    # ========================================================================
    # Verify that synthetic answers have the same keys as real answers for each model
    # We use the intersection of all keys to ensure all models are evaluated on the same questions
    questions_id = list(data.keys())  # Convert dict_keys to list
    keys_to_use = set(questions_id)
    for model in models:
        # Find intersection: only keep keys that exist in both real answers and this model's synthetic answers
        keys_to_use = keys_to_use.intersection(set(synthetic_answers_all[model].keys()))
    questions_id = list(keys_to_use)
    
    # ========================================================================
    # VALIDATE THAT WE HAVE QUESTIONS TO EVALUATE
    # ========================================================================
    # Validate that we have at least some questions to evaluate
    if len(questions_id) == 0:
        raise ValueError(f"No common question keys found between real answers and synthetic answers for all models. Cannot perform evaluation.")
    
    # Print warning if not all questions are used
    if len(questions_id) < len(data):
        print(f'Warning: {len(questions_id)} questions out of {len(data)} are used for evaluation because some models do not have synthetic answers for all questions.')
    else:
        print(f'All {len(questions_id)} questions are used for evaluation.')
    
    # ========================================================================
    # FILTER REAL_ANSWERS TO ONLY INCLUDE COMMON QUESTIONS
    # ========================================================================
    # Filter real_answers to only include questions we're using
    # This ensures consistency and prevents KeyError issues downstream
    real_answers = {key: real_answers[key] for key in questions_id if key in real_answers}

    # ========================================================================
    # RUN EVALUATION FOR ALL MODELS
    # ========================================================================
    print(f"Evaluating models on {dataset_name} dataset for types: {', '.join(types)}...")
    reports_all_by_type, sharpness_by_type = get_reports_multiple(
        real_answers, synthetic_answers_all, questions_id, alphas, gamma, k_max,
        C, train_proportion, k_min, full_param_CI, CI_type, seed, num_splits, types=types
    )

    # ========================================================================
    # SAVE RESULTS TO JSON FILES
    # ========================================================================
    # Create directory if it doesn't exist
    save_dir = os.path.join('..', 'data', dataset_name, evaluation_results_folder_name)
    os.makedirs(save_dir, exist_ok=True)

    for method_type in types:
        reports_all = reports_all_by_type.get(method_type, {})
        sharpness_all = sharpness_by_type.get(method_type, {})
        type_dir = os.path.join(save_dir, method_type)
        os.makedirs(type_dir, exist_ok=True)
        with open(os.path.join(type_dir, 'reports_all.json'), 'w') as f:
            json.dump(reports_all, f)

        with open(os.path.join(type_dir, 'sharpness_analysis_all.json'), 'w') as f:
            json.dump(sharpness_all, f)

    print('Models evaluated and results saved.\n')
        

def plot_from_saved_evaluations(
    dataset_name: str = 'EEDI',
    evaluation_results_folder_name: str = 'evaluation_results',
    num_splits: int = 100,
    alphas: list = [0.05, 0.10, 0.15, 0.20],
    gamma: float = 0.5,
    C: float = 3,
    types: list | tuple | str = ('general', 'simple'),
) -> None:
    """
    Generate plots and tables from saved evaluation results.
    
    Loads previously saved evaluation reports and generates visualizations and tables
    for all metrics and coverage types. Allows regenerating plots without re-running
    the computationally expensive evaluation process.
    
    Generates: Error bar plots (PDF), summary tables (CSV), for both 'general' and
    'simple' coverage types.
    
    Args:
        dataset_name (str, optional): Name of the dataset. Must be 'EEDI' or 'OpinionQA'.
            Defaults to 'EEDI'.
            Must match the dataset_name used when calling evaluations().
        evaluation_results_folder_name (str, optional): Name of folder containing saved evaluation results.
            Defaults to 'evaluation_results'.
            The function looks for files at:
            ../data/{dataset_name}/{evaluation_results_folder_name}/general/reports_all.json
            ../data/{dataset_name}/{evaluation_results_folder_name}/simple/reports_all.json
        num_splits (int, optional): Number of train-test splits used in the evaluation.
            Defaults to 100.
            Must match the num_splits used when calling evaluations().
            Used to compute standard errors for error bars.
        alphas (list, optional): List of significance levels used in the evaluation.
            Defaults to [0.05, 0.10, 0.15, 0.20].
            Must match the alphas used when calling evaluations().
            Used as x-axis values in plots.
        gamma (float, optional): Coverage probability for real CI used in the evaluation.
            Defaults to 0.5.
            Must match the gamma used when calling evaluations().
            Used to compute multipliers for scaling metrics.
        C (float, optional): Scaling constant for synthetic CI used in the evaluation.
            Defaults to 3.
            Must match the C used when calling evaluations().
            Used to scale kappa_hat values (kappa_hat = k_hat / C).
    
    Returns:
        None: Plots are saved as PDF files and tables are saved as CSV files
        for each requested coverage type:
            - Plots: ../data/{dataset_name}/{evaluation_results_folder_name}/{type}/{metric}.pdf
            - Tables: ../data/{dataset_name}/{evaluation_results_folder_name}/{type}/{metric}.csv
        Tables are also printed to the console.
    
    Raises:
        FileNotFoundError: 
            - If general/reports_all.json is not found
            - If simple/reports_all.json is not found
    
    Note:
        - This function must be called after evaluations() has been run and saved results.
        - The parameters (num_splits, alphas, gamma, C) must match those used in evaluations().
        - Generated metrics:
            - 'kappa_hat': Optimal k value divided by C (kappa_hat = k_hat / C)
            - 'synth_CI_width': Mean half-width of test synthetic confidence intervals
            - 'test_miscov_rate': Test miscoverage rate (scaled by multiplier)
        - Coverage types:
            - 'general': Tests if real CI is contained within synthetic CI
            - 'simple': Tests if real mean is contained within synthetic CI
    """
    # ========================================================================
    # LOAD SAVED EVALUATION REPORTS
    # ========================================================================
    
    # ========================================================================
    # SET UP SAVE INFORMATION
    # ========================================================================
    # save_info is passed to plot_reports and table_reports to specify where to save files
    save_info = {
        'dataset_name': dataset_name,
        'save_folder_name': evaluation_results_folder_name
    }
    
    # ========================================================================
    # GENERATE VISUALIZATIONS FOR 'GENERAL' COVERAGE TYPE
    # ========================================================================
    for method_type in types:
        reports_path = os.path.join('..', 'data', dataset_name, evaluation_results_folder_name, method_type, 'reports_all.json')
        if not os.path.exists(reports_path):
            raise FileNotFoundError(f"Reports file not found for type '{method_type}': {reports_path}. Please run evaluations() for this type first.")
        with open(reports_path, 'r') as f:
            reports_all = json.load(f)

        print('\n--------------------------------')
        print(f'Visualizing results for {method_type} coverage type...')
        print('--------------------------------')

        print('-----kappa_hat (plot)-----')
        plot_reports(reports_all, num_splits, alphas, gamma, C, metric='kappa_hat', type=method_type, save_info=save_info)

        print('-----test_miscov_rate (plot)-----')
        plot_reports(reports_all, num_splits, alphas, gamma, C, metric='test_miscov_rate', type=method_type, save_info=save_info)

        print('-----synth_CI_width (plot)-----')
        plot_reports(reports_all, num_splits, alphas, gamma, C, metric='synth_CI_width', type=method_type, save_info=save_info)

        print('-----kappa_hat (table)-----')
        table_reports(reports_all, num_splits, alphas, gamma, C, metric='kappa_hat', type=method_type, save_info=save_info)

        print('-----test_miscov_rate (table)-----')
        table_reports(reports_all, num_splits, alphas, gamma, C, metric='test_miscov_rate', type=method_type, save_info=save_info)

        print('-----synth_CI_width (table)-----')
        table_reports(reports_all, num_splits, alphas, gamma, C, metric='synth_CI_width', type=method_type, save_info=save_info)


def sharpness_analysis(dataset_name: str = 'EEDI',
    evaluation_results_folder_name: str = 'evaluation_results',
    type: str = 'general',
    gamma: float = 0.5,
    histogram_model: str | None = None,
) -> None:
    """
    Create sharpness analysis plots and tables.
    
    Args:
        dataset_name (str, optional): Name of the dataset. Defaults to 'EEDI'.
        evaluation_results_folder_name (str, optional): Folder containing saved results. Defaults to 'evaluation_results'.
        type (str, optional): Coverage type ('general' or 'simple'). Defaults to 'general'.
        gamma (float, optional): Coverage probability for real CI. Defaults to 0.5.
        histogram_model (str, optional): Specific LLM/model name for additional diagnostics.
            When provided, the function also computes per-split relative errors
            |k_hat - k_sharp| / k_sharp and plots histograms across random seeds.
    """

    coverage_type = type
    type_dir = os.path.join('..', 'data', dataset_name, evaluation_results_folder_name, coverage_type)
    for_sharpness_analysis_path = os.path.join(type_dir, 'sharpness_analysis_all.json')
    if not os.path.exists(for_sharpness_analysis_path):
        raise FileNotFoundError(f"For sharpness analysis file not found: {for_sharpness_analysis_path}. Please run evaluations() for type '{coverage_type}' first.")
    with open(for_sharpness_analysis_path, 'r') as f:
        for_sharpness_analysis = json.load(f)

    reports_all_path = os.path.join(type_dir, 'reports_all.json')
    if not os.path.exists(reports_all_path):
        raise FileNotFoundError(f"Reports all file not found: {reports_all_path}. Please run evaluations() first.")
    with open(reports_all_path, 'r') as f:
        reports_all = json.load(f)
    
    alphas = [0.05, 0.10, 0.15, 0.20]
    keys = list(reports_all.keys())

    table = pd.DataFrame(index=keys, columns=alphas)
    rel_error_by_model = {model_key: {alpha: [] for alpha in alphas} for model_key in keys}
    
    # Collect data for plotting selected vs oracle k
    for model_key in keys:
        for alpha in alphas:
            if coverage_type == 'general':
                threshold = alpha * gamma
            elif coverage_type == 'simple':
                threshold = alpha * 1 / 2

            alpha_key = alpha
            if alpha_key not in for_sharpness_analysis['Gks_test'][model_key]:
                alpha_key = str(alpha)
            
            test_Gks = np.array(for_sharpness_analysis['Gks_test'][model_key][alpha_key])
            k_hat_train_splits = np.asarray(for_sharpness_analysis['k_hat_train'][model_key][alpha_key])
            # Find k_sharp for each random split
            k_sharps = []
            for i in range(test_Gks.shape[0]):
                ks = np.arange(1, len(test_Gks[i]) + 1)
                k_sharp_i = int(np.round(find_k_hat(ks, test_Gks[i], threshold)))
                k_sharps.append(k_sharp_i)
            
            rel_errors = []
            for split_k_sharp, split_k_hat in zip(k_sharps, k_hat_train_splits):
                if split_k_sharp == 0:
                    continue
                rel_errors.append(abs(split_k_hat - split_k_sharp) / split_k_sharp)
            rel_error_by_model[model_key][alpha].extend(rel_errors)

    # Save table
    os.makedirs(type_dir, exist_ok=True)
    # Replace table entries with percentile tuples
    for model_key in keys:
        for alpha in alphas:
            rel_errors = rel_error_by_model[model_key][alpha]
            p95 = np.percentile(rel_errors, 95)
            table.loc[model_key, alpha] = f"{np.round(p95, 2)}"
            print(f'{model_key} {alpha} {p95}')
    table.to_csv(os.path.join(type_dir, f'{coverage_type}_sharpness_analysis_table.csv'))

    # Plot histograms of relative error for the requested model
    if histogram_model is not None:
        for alpha, rel_errors in rel_error_by_model[histogram_model].items():
            if len(rel_errors) == 0 or alpha != 0.05:
                continue
            plt.figure(figsize=(6, 4))
            # Create bins starting at 0, incrementing by 0.025
            max_error = max(rel_errors) if len(rel_errors) > 0 else 1.0
            bin_edges = np.arange(0, max_error + 0.1, 0.05)
            counts, bin_edges, _ = plt.hist(rel_errors, bins=bin_edges, color='#4B67B3', edgecolor='black', alpha=0.8)
            # Set ticks at bin edges (0, 0.05, 0.1, etc.) for cleaner labels
            tick_positions = np.arange(0, max_error + 0.1, 0.1)
            plt.xticks(tick_positions)
            plt.tight_layout()
            hist_path = os.path.join(type_dir, f'{coverage_type}_sharpness_histogram_{histogram_model}.pdf')
            plt.savefig(hist_path, bbox_inches='tight')
            plt.close()