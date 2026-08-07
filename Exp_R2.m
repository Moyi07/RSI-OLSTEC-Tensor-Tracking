%% Experiment R2
% =========================================================================
% Prerequisites: Run `run_me_first.m` once to add all dependencies to path.
%
% Objective:
%   Evaluate RSI-OLSTEC and baseline algorithms on one real WAAM-ViD video.
%
% Protocol:
%   Use 10% observed pixels, full-frame NRE, equal matrix/tensor ranks, a
%   fixed OLSTEC lambda grid, and measured auxiliary metadata.
% =========================================================================
clear;
clc;
close all;

repo_root = fileparts(mfilename('fullpath'));
config = struct();
config.video_filename = fullfile(repo_root, 'dataset', 'video', ...
    '250312-110206-video_1.mp4');
config.meta_filename = fullfile(repo_root, 'dataset', ...
    'WAMVID_metadata.csv');
config.result_dir = fullfile(repo_root, 'result', 'R2');

config.fraction        = 0.10;
config.rank_r         = 20;
config.max_frames     = 623;
config.scale_ratio    = 0.20;
config.random_seed    = 42;
config.matrix_init_seed = 40001;
config.aux_missing_policy = 'trim_leading_nan';
config.initial_calibration_frames = 30;

% Robust thresholds are estimated from the initial calibration window. NaN
% enables MAD-based Huber estimation; 0.05 is the lower bound for the
% side-information gradient threshold.
config.lambda_list                 = [0.70, 0.80, 0.90, 0.99];
config.rsi_lambda_max              = 0.80;
config.rsi_lambda_min              = 0.70;
config.rsi_huber_delta             = NaN;
config.rsi_min_grad_floor          = 0.05;
config.rsi_grad_ema_alpha          = 0.999;

% Fixed display ranges for fair comparisons across methods and frames.
config.display_intensity_limits = [0, 1];
config.display_residual_limits  = [-1, 1];

% Figure output includes frame-by-frame animation by default for this R2 run.
config.make_figures       = true;
config.make_matrix_figure = true;
config.image_display_flag = true;
config.export_results     = true;
config.store_matrix_flag  = true;

run_r2_waamvid_single(config);
