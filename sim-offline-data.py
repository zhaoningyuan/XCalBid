import pandas as pd
import numpy as np
import datetime
import os
import random

import torch
import torch.utils.data

import config
from tqdm import tqdm

np.seterr(all='raise')

def get_return_to_go(rewards, dones, discount=1.00, is_sparse_reward=False) -> np.ndarray:
    returns = []
    ep_ret, ep_len = 0.0, 0
    cur_rewards = []
    terminals = []
    N = len(rewards)
    for t, (r, d) in enumerate(zip(rewards, dones)):
        ep_ret += float(r)
        cur_rewards.append(float(r))
        terminals.append(float(d))
        ep_len += 1
        is_last_step = t == N - 1

        if d or is_last_step:
            discounted_returns = [0] * ep_len
            prev_return = 0
            if is_sparse_reward:
                discounted_returns = [r / (1 - discount)] * ep_len
            else:
                for i in reversed(range(ep_len)):
                    discounted_returns[i] = cur_rewards[
                        i
                    ] + discount * prev_return * (1 - terminals[i])
                    prev_return = discounted_returns[i]
            returns += discounted_returns
            ep_ret, ep_len = 0.0, 0
            cur_rewards = []
            terminals = []
    return returns

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def bidding(bid):
    return int(bid if bid <= 300 else 300)


def generate_bid_price(datas):
    '''
    :param datas: type list
    :return:
    '''
    return np.array(list(map(bidding, datas))).astype(int)


def bid_main(bid_prices, imp_datas, budget):
    '''
    Main bidding process
    param bid_prices: [bid_price]
    param imp_datas: [clk, pctr, mprice]
    return: Number of clicks obtained by the model, number of clicks of real data, pctr obtained by the model, pctr of real data, number of impressions of real data, number of impressions obtained by the model, cost
    '''
    win_imp_indexs = np.where(bid_prices >= imp_datas[:, 2])[0]
    win_imp_datas = imp_datas[win_imp_indexs, :]

    win_clks, real_clks, win_pctr, real_pctr, bids, imps, cost = 0, 0, 0, 0, 0, 0, 0
    if len(win_imp_datas):
        first, last = 0, win_imp_datas.shape[0] - 1

        final_index = 0
        while first <= last:
            mid = first + (last - first) // 2
            tmp_sum = np.sum(win_imp_datas[:mid, 2])
            if tmp_sum < budget:
                first = mid + 1
            else:
                last_sum = np.sum(win_imp_datas[:mid - 1, 2])
                if last_sum <= budget:
                    final_index = mid - 1
                    break
                else:
                    last = mid - 1

        final_index = final_index if final_index else first
        win_clks = np.sum(win_imp_datas[:final_index, 0])
        win_pctr = np.sum(win_imp_datas[:final_index, 1])
        origin_index = win_imp_indexs[final_index - 1]
        real_clks = np.sum(imp_datas[:origin_index, 0])
        real_pctr = np.sum(imp_datas[:origin_index, 1])
        imps = final_index
        bids = origin_index + 1

        cost = np.sum(win_imp_datas[:final_index, 2])
        current_cost = cost

        if len(win_imp_datas[final_index:, :]) > 0:
            if current_cost < budget:
                budget -= current_cost

                remain_win_imps = win_imp_datas[final_index:, :]
                mprice_less_than_budget_imp_indexs = np.where(remain_win_imps[:, 2] <= budget)[0]

                final_mprice_lt_budget_imps = remain_win_imps[mprice_less_than_budget_imp_indexs]
                last_win_index = 0
                for idx, imp in enumerate(final_mprice_lt_budget_imps):
                    tmp_mprice = final_mprice_lt_budget_imps[idx, 2]
                    if budget - tmp_mprice >= 0:
                        win_clks += final_mprice_lt_budget_imps[idx, 0]
                        win_pctr += final_mprice_lt_budget_imps[idx, 1]
                        imps += 1
                        # bids += (mprice_less_than_budget_imp_indexs[idx] - last_win_index + 1)
                        last_win_index = mprice_less_than_budget_imp_indexs[idx]
                        bids = win_imp_indexs[final_index + last_win_index] + 1
                        cost += tmp_mprice
                        budget -= tmp_mprice
                    else:
                        continue
                real_clks += np.sum(remain_win_imps[:last_win_index, 0])
                real_pctr += np.sum(remain_win_imps[:last_win_index, 1])
            else:
                win_clks, real_clks, bids, imps, cost = 0, 0, 0, 0, 0
                last_win_index = 0
                for idx, imp in enumerate(win_imp_datas):
                    tmp_mprice = win_imp_datas[idx, 2]
                    real_clks += win_imp_datas[idx, 0]
                    real_pctr += win_imp_datas[idx, 1]
                    if budget - tmp_mprice >= 0:
                        win_clks += win_imp_datas[idx, 0]
                        win_pctr += win_imp_datas[idx, 1]
                        imps += 1
                        bids += (win_imp_indexs[idx] - last_win_index + 1)
                        last_win_index = win_imp_indexs[idx]
                        cost += tmp_mprice
                        budget -= tmp_mprice

    return win_clks, real_clks, win_pctr, real_pctr, bids, imps, cost

def get_dataset(args):
    data_path = os.path.join(args.data_path + args.dataset_name, args.campaign_id)

    train_data_df = pd.read_csv(os.path.join(data_path, 'train.bid.lin.csv'))
    test_data_df = pd.read_csv(os.path.join(data_path, 'test.bid.lin.csv'))

    budget = []
    for index, day in enumerate(train_data_df.day.unique()):
        current_day_budget = np.sum(train_data_df[train_data_df.day.isin([day])].market_price)

        budget.append(current_day_budget)

    return train_data_df, test_data_df, budget


def reward_func(reward_type, fab_clks, hb_clks, fab_cost, hb_cost, fab_pctrs):
    # input: reward_type, fab_clks, lin_clks, fab_cost, lin_cost, fab_pctrs
    if fab_clks > hb_clks and fab_cost <= hb_cost:
        r = 5
    elif fab_clks > hb_clks and fab_cost > hb_cost:
        r = 1
    elif fab_clks < hb_clks and fab_cost >= hb_cost:
        r = -5
    elif fab_clks < hb_clks and fab_cost < hb_cost:
        r = -2.5
    else:
        r = 0

    if reward_type == 'op':
        return r / 1000
    elif reward_type == 'nop':
        return r
    elif reward_type == 'nop_2.0':
        return fab_clks / 1000
    elif reward_type == 'pctr':
        return fab_pctrs
    else:
        return fab_clks


def choose_init_base_bid(config):
    base_bid_path = os.path.join('../lin/result/ipinyou/{}/normal/test'.format(config['campaign_id']),
                                 'test_bid_log.csv')
    if not os.path.exists(base_bid_path):
        raise FileNotFoundError('Run LIN first before you train FAB')
    data = pd.read_csv(base_bid_path)
    base_bid = data[data['budget_prop'] == config['budget_para']].iloc[0]['base_bid']
    avg_pctr = data[data['budget_prop'] == config['budget_para']].iloc[0]['average_pctr']

    return avg_pctr, base_bid


if __name__ == '__main__':
    args = config.init_parser()
    time_fraction = args.time_fraction
    time_fraction_str = str(time_fraction) + '_time_fraction'
    train_data_df, test_data_df, budget = get_dataset(args)
    setup_seed(args.seed)
    log_dirs = [args.save_log_dir, os.path.join(args.save_log_dir, args.campaign_id)]
    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    result_path = os.path.join(args.result_path, args.campaign_id)
    os.makedirs(result_path, exist_ok=True)

    device = torch.device(args.device)  

    B = [b / args.budget_para for b in budget]

    avg_ctr, hb_base = choose_init_base_bid(vars(args))

    train_losses = []


    start_time = datetime.datetime.now()

    clk_index, ctr_index, mprice_index, hour_index = 0, 1, 2, 3

    offline_data = {
        "state": [],
        "action": [],
        "next_state": [],
        "done": [],
        "reward": [],
        "mc_return": [],
        "clks": [],
        "cost": [],
        "days": [],
    }
    for sig in tqdm(np.linspace(0, args.sigma_max, num=args.sigma_num)):
        # win_clks, real_clks, win_pctr, real_pctr, bids, imps, cost

        for day_index, day in enumerate(train_data_df.day.unique()):
            train_data = train_data_df[train_data_df.day.isin([day])]
            # train data: clk, pctr, market_price, time_fraction_96
            train_data = train_data[['clk', 'pctr', 'market_price', time_fraction_str]].values.astype(float)
            budget = B[day_index]
            current_day_budget = budget

            next_state = [1, 0, 0, 0]
            init_state = [1, 0, 0, 0]

            actions = np.random.normal(0, sig, time_fraction)
            actions = np.clip(actions, -.9, 1)

            done = 0
            for t in range(time_fraction):
                if budget > 0:
                    # hour_index is time_fraction_96
                    hour_datas = train_data[train_data[:, hour_index] == t]

                    state = torch.tensor(init_state).float() if not t else torch.tensor(next_state).float()

                    # bid = ctr_ratio * base_bid / (1 + action) 
                    # action = 0 or [-.2, .2] normal distribution
                    action = actions[t]
                    bid_datas = generate_bid_price((hour_datas[:, ctr_index] * (hb_base / avg_ctr)) / (1 + action))
                    res_ = bid_main(bid_datas, hour_datas, budget)
                    # win_clks, real_clks, win_pctr, real_pctr, bids, imps, cost

                    # train_records = [train_records[i] + res_[i] for i in range(len(train_records))]

                    left_hour_ratio = (time_fraction - 1 - t) / (time_fraction - 1) if t <= (time_fraction - 1) else 0

                    if (not left_hour_ratio) or (budget <= 0):
                        done = 1

                    # avg_budget_ratio, cost_ratio, ctr, win_rate
                    next_state = [
                            (budget / current_day_budget) / left_hour_ratio if left_hour_ratio else ( budget / current_day_budget),
                            res_[6] / current_day_budget, 
                            res_[0] / res_[5] if res_[5] else 0,
                            res_[5] / res_[4] if res_[4] else 0
                        ]


                    # lin method result: action = 0
                    hb_bid_datas = generate_bid_price(hour_datas[:, ctr_index] * hb_base / avg_ctr)
                    res_hb = bid_main(hb_bid_datas, hour_datas, budget)
                    # update budget, budget -> remaining budget
                    # input: reward_type, fab_clks, lin_clks, fab_cost, lin_cost, fab_pctrs
                    r_t = reward_func(args.reward_type, res_[0], res_hb[0], res_[6], res_hb[6], res_[2])

                    offline_data["state"].append(np.array(state))
                    offline_data["action"].append(action)
                    offline_data["next_state"].append(np.array(next_state))
                    offline_data["done"].append(done)
                    offline_data["reward"].append(r_t)
                    offline_data["clks"].append(res_[0])
                    offline_data["cost"].append(res_[6])
                    offline_data["days"].append(day_index)

    offline_data["mc_return"] = get_return_to_go(offline_data["reward"], offline_data["done"], discount=0.99, is_sparse_reward=False)
    # concat
    for key in offline_data.keys():
        offline_data[key] = np.array(offline_data[key])
    print("done")
    np.savez(args.offline_data_path, offline_data)
