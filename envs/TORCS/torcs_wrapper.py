import cv2
import math
import numpy as np
import pdb
import copy


class DoneCondition:
    def __init__(self, size):
        self.size = size
        self.off_cnt = 0
        self.pos = []

    def isdone(self, pos, dist, posxyz, angle):
        if pos <= -6.2 and dist < 0:
            self.off_cnt += 1
        elif pos > -6.2 or dist > 0:
            self.off_cnt = 0
        if self.off_cnt > self.size:
            self.off_cnt = 0
            self.pos = []
            return True
        if abs(pos) >= 21.0:
            self.off_cnt = 0
            self.pos = []
            return True
        self.pos.append(list(posxyz))
        real_pos = np.concatenate(self.pos[-100:])
        real_pos = real_pos.reshape(-1,3)
        std = np.sum(np.std(real_pos, 0))
        if std < 2.0 and len(self.pos) > 100:
            self.pos = []
            self.off_cnt = 0
            return True
        return False


def naive_driver(info, continuous):
    if info['angle'] > 0.5 or (info['trackPos'] < -1 and info['angle'] > 0):
        return np.array([1.0, 0.1]) if continuous else 0
    elif info['angle'] < -0.5 or (info['trackPos'] > 3 and info['angle'] < 0):
        return np.array([1.0, -0.1]) if continuous else 2
    return np.array([1.0, 0.0]) if continuous else 1


class TorcsWrapper:
    def __init__(self, env, imsize=(256, 256), random_reset=True, continuous=True):
        self.env = env
        self.imsize = imsize
        self.random_reset = random_reset
        self.continuous = continuous
        self.doneCond = DoneCondition(10)
        self.epi_len = 0
        self.coll_cnt = 0
        self.done_cnt = 0
        self.last_done = np.array([467, 12.54])

    def reset(self, restart=False):
        obs = self.env.reset()
        _, _, _, info = self.env.step(naive_driver(self.env.get_info(), self.continuous))
        current_pos = np.array(info['pos'][:2])
        dist = np.sqrt(np.sum((current_pos-self.last_done)**2.0))
        while dist > 100 and not restart:
            obs, _, _, info = self.env.step(naive_driver(self.env.get_info(), self.continuous))
            current_pos = np.array(info['pos'][:2])
            dist = np.sqrt(np.sum((current_pos-self.last_done)**2.0))
        info = self.env.get_info()
        off_flag = int(info['trackPos'] >=3 or info['trackPos'] <= -1)
        coll_flag = int(abs(info['trackPos']) > 7.0)
        info['offroad'] = off_flag
        info['collision'] = coll_flag
        info['speed'] = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])))
        info['seg'] = self.env.get_segmentation().reshape(1, 256, 256)
        self.doneCond = DoneCondition(10)
        self.epi_len = 0
        self.coll_cnt = 0
        self.done_cnt = 0
        return cv2.resize(obs, self.imsize), info

    def step(self, action):
        if self.continuous:
            real_action = copy.deepcopy(action)
            real_action[0] = real_action[0] * 0.5 + 0.5
            real_action[1] = real_action[1] * 0.1
        else:
            real_action = copy.deepcopy(action)
        self.epi_len += 1
        obs, reward, real_done, info = self.env.step(real_action)
        reward_with_pos = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])) - np.abs(info['trackPos']) / 9.0) / 40.0
        reward_without_pos = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle']))) / 40.0
        info['speed'] = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])))
        done = self.doneCond.isdone(info['trackPos'], info['speed'], info['pos'], info['angle']) or self.epi_len > 1000 or self.done_cnt > 30
        off_flag = int(info['trackPos'] >= 3 or info['trackPos'] <= -1)
        coll_flag = int(abs(info['trackPos']) > 7.0)
        if coll_flag:
            self.coll_cnt += 1
        else:
            self.coll_cnt = 0
        if info['trackPos'] <= -4.0 or info['trackPos'] >= 5.0:
            self.done_cnt += 1
        else:
            self.done_cnt = 0
        obs = cv2.resize(obs, self.imsize)
        reward = reward_without_pos
        info['offroad'] = off_flag
        info['collision'] = coll_flag
        info['seg'] = self.env.get_segmentation().reshape(1, 256, 256)
        if done:
            self.last_done = np.array(info['pos'][:2])
        return obs, reward, done, info

    def close(self):
        self.env.close()
