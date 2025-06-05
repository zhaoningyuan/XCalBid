import argparse
import torch
import random
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou, cretio, yoyi, avazu')
    parser.add_argument('--campaign_id', default='1458', help='1458, 3427')
    parser.add_argument("--offline_iterations", type=int, default=50)
    parser.add_argument('--online_iterations', type=int, default=50)
    parser.add_argument('--model_name', default='Calql')
    parser.add_argument('--lr_A', type=float, default=3e-4)
    parser.add_argument('--lr_C', type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=1.00)
    parser.add_argument('--buffer_size', type=float, default=int(1e5))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--mixing_ratio", type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_log_dir', default='./log/')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--time_fraction', type=int, default=96)
    parser.add_argument('--state_dim', type=int, default=4)
    parser.add_argument('--action_dim', type=int, default=1)
    parser.add_argument("--max_action", type=float, default=1.0)

    parser.add_argument("--orthogonal_init", type=bool, default=False)
    parser.add_argument("--q_n_hidden_layers", type=int, default=3)
    parser.add_argument("--soft_target_update_rate", type=float, default=5e-3)
    parser.add_argument("--alpha_multiplier", type=float, default=1.0)
    parser.add_argument("--use_automatic_entropy_tuning", type=bool, default=True)
    parser.add_argument("--backup_entropy", type=bool, default=False)
    parser.add_argument("--bc_steps", type=int, default=1000)
    parser.add_argument("--target_update_period", type=int, default=1)
    parser.add_argument("--cql_n_actions", type=int, default=10)
    parser.add_argument("--cql_importance_sample", type=bool, default=True)
    parser.add_argument("--cql_lagrange", type=bool, default=False)
    parser.add_argument("--cql_target_action_gap", type=float, default=-1.0)
    parser.add_argument("--cql_temp", type=float, default=1.0)
    parser.add_argument("--cql_alpha", type=float, default=5.0)
    parser.add_argument("--cql_max_target_backup", type=bool, default=False)
    parser.add_argument("--cql_clip_diff_min", type=float, default=0.0)
    parser.add_argument("--cql_clip_diff_max", type=float, default=1.0)

    # not sure
    parser.add_argument("--cql_alpha_online", type=float, default=1.0)


    parser.add_argument('--budget_para', type=int, default=2, help='2,4,8,16')  # Budget adjustment ratio

    parser.add_argument('--reward_type', type=str, default='op', help='op, nop_2.0, clk')
    # op scales, nop does not scale, clk, directly use the number of clicks as a reward
    # op: r / 1000, nop: r, clk: clk

    # collect
    parser.add_argument("--sigma_max", type=float, default=0.5)
    parser.add_argument("--sigma_num", type=int, default=100)
    parser.add_argument("--offline_data_path", type=str, default='offline_data.npz')

    args = parser.parse_args()

    return args
