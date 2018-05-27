import numpy as np
import torch
from torch.autograd import Variable
import copy
from utils import *
from dqn_utils import *
from dqn_agent import *
from mpc_utils import *

def test(args, env, net, avg_img, std_img):
    obs_buffer = ObsBuffer(args.frame_history_len)
    _ = env.reset(rand_reset=False) # always start from the same place
    prev_act = np.array([1.0, 0.0]) if args.continuous else 1
    obs, reward, done, info = env.step(prev_act)
    prev_info = copy.deepcopy(info)
    prev_xyz = np.array(info['pos'])
    rewards_with, rewards_without = 0, 0
    exploration = PiecewiseSchedule([(0, 0.0), (1000, 0.0)], outside_value = 0.0)
        
    if args.use_dqn:
        dqn_agent = DQNAgent(args.frame_history_len, args.num_dqn_action, args.lr, exploration, args.save_path, args=args)
        dqn_agent.load_model() 
    else:
        dqn_agent = None

    mpc_buffer = MPCBuffer(args)
    while not done:
        if args.use_dqn:
            dqn_action = dqn_agent.sample_action(obs, 1e8)
        ret = mpc_buffer.store_frame(obs)
        this_obs_np = obs_buffer.store_frame(obs, avg_img, std_img)
        obs_var = Variable(torch.from_numpy(this_obs_np).unsqueeze(0)).float().cuda()

        if args.continuous:
            action = sample_cont_action(args, net, obs_var, prev_action = prev_act, testing = True)
            action = np.clip(action, -1, 1)
            if args.use_dqn:
                if abs(action[1]) <= dqn_action * 0.1:
                    action[1] = 0
            real_action = action
        else:
            action = real_action = sample_discrete_action(args, net, obs_var, prev_action = prev_act)

        obs, reward, done, info = env.step(real_action)
        if args.continuous:
            print('action', "{0:.2f}".format(action[0]), "{0:.2f}".format(action[1]), ' pos ', "{0:.2f}".format(info['trackPos']), "{0:.2f}".format(info['pos'][0]), "{0:.2f}".format(info['pos'][1]),\
                ' reward ', "{0:.2f}".format(reward['with_pos']))
        else:
            print('action', '%d' % real_action, ' pos ', "{0:.2f}".format(info['trackPos']), "{0:.2f}".format(info['pos'][0]), "{0:.2f}".format(info['pos'][1]),\
                ' reward ', "{0:.2f}".format(reward['with_pos']))
        prev_act = action

        speed_np, pos_np, posxyz_np = get_info_np(info, use_pos_class = False)
        offroad_flag, coll_flag = info['off_flag'], info['coll_flag']
        speed_list, pos_list = get_info_ls(prev_info)
        if args.use_xyz:
            xyz = np.array(info['pos'])
            rela_xyz = xyz - prev_xyz
            prev_xyz = xyz
        else:
            rela_xyz = None

        seg = env.env.get_segmentation().reshape((1, 256, 256)) if args.use_seg else None
        mpc_buffer.store_effect(ret, action, done, coll_flag, offroad_flag, info['speed'], info['angle'], pos_list[0], rela_xyz, seg)
        rewards_with += reward['with_pos']
        rewards_without += reward['without_pos']
        prev_info = copy.deepcopy(info) 
        if args.use_dqn:
            dqn_agent.store_effect(dqn_action, reward['with_pos'], done)

    return {'with_pos': rewards_with, 'without_pos': rewards_without} 
