import pandas as pd
import numpy as np
import datetime
import os
import random
import RL_brain_calql as brain

import torch
import torch.utils.data

import config
import logging
import sys
from tqdm import tqdm

np.seterr(all='raise')


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
    Main bidding program
    :param bid_prices: [bid_price]
    :param imp_datas: [clk, pctr, mprice]
    :return: Number of clicks obtained by the model, number of clicks in real data, pctr obtained by the model, pctr in real data, number of impressions in real data, number of impressions obtained by the model, cost
    '''
    win_imp_indexs = np.where(bid_prices >= imp_datas[:, 2])[0]
    win_imp_datas = imp_datas[win_imp_indexs, :]

    win_clks, real_clks, win_pctr, real_pctr, bids, imps, cost = 0, 0, 0, 0, 0, 0, 0

    if len(win_imp_datas) > 0:
        n = win_imp_datas.shape[0]
        low, high, best_k = 0, n, 0

        # Binary search to find the largest k such that the sum of the first k elements is <= budget
        while low <= high:
            mid = (low + high) // 2
            current_sum = np.sum(win_imp_datas[:mid, 2])
            if current_sum <= budget:
                best_k = mid
                low = mid + 1
            else:
                high = mid - 1

        final_index = best_k

        if final_index > 0:
            # Process the part obtained by the model
            win_clks = np.sum(win_imp_datas[:final_index, 0])
            win_pctr = np.sum(win_imp_datas[:final_index, 1])
            cost = np.sum(win_imp_datas[:final_index, 2])
            imps = final_index

            # Calculate the real data part
            origin_index = win_imp_indexs[final_index - 1]
            real_clks = np.sum(imp_datas[:origin_index + 1, 0])  # Include origin_index
            real_pctr = np.sum(imp_datas[:origin_index + 1, 1])
            bids = origin_index + 1

        # Handle remaining budget to buy subsequent impressions
        remaining_budget = budget - cost
        if remaining_budget > 0 and final_index < n:
            remaining_imps = win_imp_datas[final_index:]
            for i in range(len(remaining_imps)):
                mprice = remaining_imps[i, 2]
                if mprice <= remaining_budget:
                    win_clks += remaining_imps[i, 0]
                    win_pctr += remaining_imps[i, 1]
                    cost += mprice
                    imps += 1
                    remaining_budget -= mprice
                    # Update the real data part
                    origin_index = win_imp_indexs[final_index + i]
                    real_clks += imp_datas[origin_index, 0]
                    real_pctr += imp_datas[origin_index, 1]
                    bids = origin_index + 1
                else:
                    break

        return win_clks, real_clks, win_pctr, real_pctr, bids, imps, cost

def get_model(args):
    critic_1 = brain.FullyConnectedQFunction(
        args.state_dim,
        args.action_dim,
        args.orthogonal_init,
        args.q_n_hidden_layers,
    ).to(args.device)
    critic_2 = brain.FullyConnectedQFunction(
        args.state_dim,
        args.action_dim,
        args.orthogonal_init,
        args.q_n_hidden_layers,
    ).to(args.device)
    critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()), args.lr_C)
    critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()), args.lr_C)

    actor = brain.TanhGaussianPolicy(
        args.state_dim,
        args.action_dim,
        args.max_action,
        orthogonal_init=args.orthogonal_init,
    ).to(args.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), args.lr_A)

    kwargs = {
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": args.discount,
        "soft_target_update_rate": args.soft_target_update_rate,
        "device": args.device,
        # CQL
        "target_entropy": -np.prod(args.action_dim),
        "alpha_multiplier": args.alpha_multiplier,
        "use_automatic_entropy_tuning": args.use_automatic_entropy_tuning,
        "backup_entropy": args.backup_entropy,
        "policy_lr": args.lr_A,
        "qf_lr": args.lr_C,
        "bc_steps": args.bc_steps,
        "target_update_period": args.target_update_period,
        "cql_n_actions": args.cql_n_actions,
        "cql_importance_sample": args.cql_importance_sample,
        "cql_lagrange": args.cql_lagrange,
        "cql_target_action_gap": args.cql_target_action_gap,
        "cql_temp": args.cql_temp,
        "cql_alpha": args.cql_alpha,
        "cql_max_target_backup": args.cql_max_target_backup,
        # "cql_clip_diff_min": args.cql_clip_diff_min,
        # "cql_clip_diff_max": args.cql_clip_diff_max,
    }
    RL_model = brain.CalQL(**kwargs)

    return RL_model


def get_dataset(args):
    # Data Path
    data_path = os.path.join(args.data_path + args.dataset_name, args.campaign_id)

    # Read data
    train_data_df = pd.read_csv(os.path.join(data_path, 'train.bid.lin.csv'))
    test_data_df = pd.read_csv(os.path.join(data_path, 'test.bid.lin.csv'))

    # Get the daily budget, ECPC, average click - through rate, and average transaction price
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

class Env:
    def __init__(self, data_df, budget, args, time_fraction_str):
        self.data_df = data_df
        days = data_df.day.unique()
        self.budget = budget
        self.B = {day: b / args.budget_para for day, b in zip(days, budget)}
        self.time_fraction = args.time_fraction
        self.reward_type = args.reward_type
        self.t = 0
        self.current_day_budget = 0
        self.remaining_budget = 0

        self.time_fraction_str = time_fraction_str
        self.clk_index, self.ctr_index, self.mprice_index, self.hour_index = 0, 1, 2, 3
        self.avg_ctr, self.hb_base = choose_init_base_bid(vars(args))
        self.init_state = np.array([1., 0, 0, 0])
        self.current_day = 0
        
    def step(self, action):
        # win_clks, real_clks, win_pctr, real_pctr, bids, imps, cost
        hour_datas = self.data[self.data[:, self.hour_index] == self.t]
        bid_datas = generate_bid_price((hour_datas[:, self.ctr_index] * (self.hb_base / self.avg_ctr)) / (1 + action + 1e-6))
        res_ = bid_main(bid_datas, hour_datas, self.remaining_budget)
        # log
        info = {
            "win_clks": res_[0],
            "real_clks": res_[1],
            "win_pctr": res_[2],
            "real_pctr": res_[3],
            "bids": res_[4],
            "imps": res_[5],
            "cost": res_[6]
        }
        left_hour_ratio = (self.time_fraction - 1 - self.t) / (self.time_fraction - 1) if self.t <= (self.time_fraction - 1) else 0
        if (not left_hour_ratio) or (self.remaining_budget <= 0):
            done = 1
        else:
            done = 0

        # avg_budget_ratio, cost_ratio, ctr, win_rate
        next_state = [
                (self.remaining_budget / self.current_day_budget) / left_hour_ratio if left_hour_ratio else ( self.remaining_budget / self.current_day_budget),
                res_[6] / self.current_day_budget, 
                res_[0] / res_[5] if res_[5] else 0,
                res_[5] / res_[4] if res_[4] else 0
            ]
        hb_bid_datas = generate_bid_price(hour_datas[:, self.ctr_index] * self.hb_base / self.avg_ctr)
        res_hb = bid_main(hb_bid_datas, hour_datas, self.remaining_budget)
        # update budget, budget -> remaining budget
        self.remaining_budget -= res_[-1]
        # input: reward_type, fab_clks, lin_clks, fab_cost, lin_cost, fab_pctrs
        r_t = reward_func(self.reward_type, res_[0], res_hb[0], res_[6], res_hb[6], res_[2])

        self.t += 1
        return np.asarray(next_state), r_t, done, info

    def reset(self, day_index):
        self.current_day = day_index
        self.t = 0
        self.data = self.data_df[self.data_df.day.isin([day_index])]
        self.data = self.data[['clk', 'pctr', 'market_price', self.time_fraction_str]].values.astype(float)
        self.remaining_budget = self.current_day_budget = self.B[day_index]

        return self.init_state

def get_day_index(days, day_pass, num_days):
    new_epoch = day_pass % num_days == 0
    return days[day_pass % num_days], day_pass + 1, new_epoch

if __name__ == '__main__':
    args = config.init_parser()
    batch_size_offline = int(args.batch_size * args.mixing_ratio)
    batch_size_online = args.batch_size - batch_size_offline
    time_fraction = args.time_fraction
    time_fraction_str = str(time_fraction) + '_time_fraction'
    train_data_df, test_data_df, train_budget = get_dataset(args)
    test_budget = [np.sum(test_data_df[test_data_df.day.isin([day])].market_price) for day in test_data_df.day.unique()]
    setup_seed(args.seed)
    log_dirs = [args.save_log_dir, os.path.join(args.save_log_dir, args.campaign_id)]
    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(args.save_log_dir, str(args.campaign_id),
                                              args.model_name + '_output.log'),
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    result_path = os.path.join(args.result_path, args.campaign_id)
    os.makedirs(result_path, exist_ok=True)

    device = torch.device(args.device) 

    logger.info(args.campaign_id)
    # logger.info('RL model ' + args.model_name + ' has been training')
    # logger.info(args)

    # load buffer
    offline_buffer = brain.ReplayBuffer(
        args.state_dim,
        args.action_dim,
        int(args.buffer_size),
        args.device,
    )
    online_buffer = brain.ReplayBuffer(
        args.state_dim,
        args.action_dim,
        int(args.buffer_size),
        args.device,
    )
    with np.load(args.offline_data_path, allow_pickle=True) as data:
        offline_dataset = data["arr_0"].item()
    # offline_dataset = np.load(args.offline_data_path)
    offline_buffer.load_offline_dataset(offline_dataset)

    # make env
    train_env = Env(train_data_df, train_budget, args, time_fraction_str)
    test_env = Env(test_data_df, test_budget, args, time_fraction_str)

    rl_model = get_model(args)

    def test_policy(env, policy, test_days, device):
        policy.eval()

        # print(policy.base_network[0].weight.data.cpu().abs().sum().numpy())
        records = {"clks": 0, "real_clks": 0, "pctrs": 0, "real_pctr": 0, "bids": 0, "imps": 0, "cost": 0, "reward": 0, }
        for day_index in test_days:
            state, done = env.reset(day_index), False
            while not done:
                action = policy.act(state, device)
                state, reward, done, info = env.step(action)
                records["clks"]+=info["win_clks"]
                records["real_clks"]+=info["real_clks"]
                records["pctrs"]+=info["win_pctr"]
                records["real_pctr"]+=info["real_pctr"]
                records["bids"]+=info["bids"]
                records["imps"]+=info["imps"]
                records["cost"]+=info["cost"]
                records["reward"]+=reward
        policy.train()
        return records

    # logger.info('para:{}, budget:{}, base bid: {}'.format(args.budget_para, B, hb_base))
    # logger.info('\tclks\treal_clks\tbids\timps\tcost')

    start_time = datetime.datetime.now()

    train_days = train_data_df.day.unique()
    test_days = test_data_df.day.unique()

    num_train_days = len(train_days)
    train_day_pass = 0

    day_index, train_day_pass, new_epoch = get_day_index(train_days, train_day_pass, num_train_days)
    state, done = train_env.reset(day_index), False
    logger.info("Offline pretraining")

    steps_pre_iter = time_fraction * num_train_days
    offline_step = 0
    online_step = 0

    # win_clks, real_clks, win_pctr, real_pctr, bids, imps, cost
    train_records = {"ep": [], "clks": [], "real_clks": [], "pctrs": [], "real_pctr": [], "bids": [], "imps": [], "cost": [], "loss": [], }
    test_records = {"ep": [], "clks": [], "real_clks": [], "pctrs": [], "real_pctr": [], "bids": [], "imps": [], "cost": [], "reward": [], }

    train_record = {"clks": 0, "real_clks": 0, "pctrs": 0, "real_pctr": 0, "bids": 0, "imps": 0, "cost": 0, }

    t = 0
    q_loss = 0
    while t < int(args.online_iterations + args.offline_iterations):
        if t == args.offline_iterations and online_step == 0:
            logger.info("Online tuning")
            online_step = 1
            rl_model.switch_calibration()
            rl_model.cql_alpha = args.cql_alpha_online
        if t >= args.offline_iterations:
            action, _ = rl_model.actor(
                torch.tensor(
                    state.reshape(1, -1),
                    device=args.device,
                    dtype=torch.float32,
                )
            )
            action = action.cpu().data.numpy().flatten()
            next_state, reward, done, env_info = train_env.step(action)
            # logger.info(f'{env_info["win_clks"]}\t{env_info["real_clks"]}\t{env_info["bids"]}\t{env_info["imps"]}\t{env_info["cost"]}')

            train_record["clks"] += env_info["win_clks"]
            train_record["real_clks"] += env_info["real_clks"]
            train_record["pctrs"] += env_info["win_pctr"]
            train_record["real_pctr"] += env_info["real_pctr"]
            train_record["bids"] += env_info["bids"]
            train_record["imps"] += env_info["imps"]
            train_record["cost"] += env_info["cost"]

            real_done = done  
            online_buffer.add_transition(state, action, reward, next_state, real_done)
            state = next_state

            if done:
                day_index, train_day_pass, new_epoch = get_day_index(train_days, train_day_pass, num_train_days)
                state, done = train_env.reset(day_index), False
                if new_epoch:
                    online_step = steps_pre_iter
                    train_records["ep"].append(t - args.offline_iterations)
                    train_records["clks"].append(train_record["clks"])
                    train_records["real_clks"].append(train_record["real_clks"])
                    train_records["pctrs"].append(train_record["pctrs"])
                    train_records["real_pctr"].append(train_record["real_pctr"])
                    train_records["bids"].append(train_record["bids"])
                    train_records["imps"].append(train_record["imps"])
                    train_records["cost"].append(train_record["cost"])
                    train_records["loss"].append(q_loss)
                    train_record = {"clks": 0, "real_clks": 0, "pctrs": 0, "real_pctr": 0, "bids": 0, "imps": 0, "cost": 0}
                    t += 1


        if t < args.offline_iterations:
            batch = offline_buffer.sample(args.batch_size)
            batch = [b.to(args.device) for b in batch]
            offline_step += 1
            if offline_step >= steps_pre_iter:
                logger.info(f"Time steps: {t}")
                t += 1
                offline_step = 0
        else:
            offline_batch = offline_buffer.sample(batch_size_offline)
            online_batch = online_buffer.sample(batch_size_online)
            batch = [
                torch.vstack(tuple(b)).to(args.device)
                for b in zip(offline_batch, online_batch)
            ]
            # online_step += 1

        log_dict = rl_model.train(batch)
        q_loss = min(log_dict["qf1_loss"], log_dict["qf2_loss"])

        # Evaluate episode
        if (t >= args.offline_iterations and online_step == steps_pre_iter) or (t < args.offline_iterations and offline_step == 0):
            online_step = 1
            logger.info(f"Time steps: {t-1}")
            records = test_policy(test_env, rl_model.actor, test_days, args.device)
            test_records["clks"].append(records["clks"])
            test_records["real_clks"].append(records["real_clks"])
            test_records["pctrs"].append(records["pctrs"])
            test_records["real_pctr"].append(records["real_pctr"])
            test_records["bids"].append(records["bids"])
            test_records["imps"].append(records["imps"])
            test_records["cost"].append(records["cost"])
            test_records["reward"].append(records["reward"])
            # test_records["ep"].append(t - args.offline_iterations - 1)
            test_records["ep"].append(t)
            
    train_record_df = pd.DataFrame(data=train_records,)
    train_record_df.to_csv(
        os.path.join(result_path, 'calql_train_records_' + args.reward_type + '_' + str(
            args.budget_para) + '.csv'), index=None)

    test_record_df = pd.DataFrame(data=test_records)
    test_record_df.to_csv(
        os.path.join(result_path, 'calql_test_records_' + args.reward_type + '_' + str(
            args.budget_para) + '.csv'), index=None)
