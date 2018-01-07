class learning_param(object):
    def __init__(self, num_update_max, sample_interval, num_trial, gamma, num_episode, episode_len_max,
                 num_episode_eval, alpha_init, alpha_variance_adaptive_BAC=None, alpha_schedule_BAC=None,
                 alpha_update_param_BAC=None):
        """
        :param num_update_max: 参数更新的上限
        :param sample_interval: 评估的间隔
        :param num_trial: 实验的次数
        :param gamma: 折现的系数
        :param num_episode: 每次参数更新前执行的episode数目，对应论文中的M
        :param episode_len_max: 仿真的时间步数上限
        :param num_episode_eval: 评估时用来平均的次数
        :param alpha_init: alpha初值
        :param alpha_variance_adaptive_BAC:
        :param alpha_schedule_BAC:
        :param alpha_update_param_BAC:
        """
        self.num_update_max = num_update_max
        self.sample_interval = sample_interval
        self.num_trial = num_trial
        self.gamma = gamma
        self.num_episode = num_episode
        self.episode_len_max = episode_len_max
        self.num_episode_eval = num_episode_eval
        self.alpha_init = alpha_init
        if alpha_variance_adaptive_BAC:
            self.alpha_variance_adaptive_BAC = alpha_variance_adaptive_BAC
        if alpha_schedule_BAC:
            self.alpha_schedule_BAC = alpha_schedule_BAC
        if alpha_update_param_BAC:
            self.alpha_update_param_BAC = alpha_update_param_BAC


class state(object):
    def __init__(self, x=None, y=None, isgoal=None):
        self.x = x
        self.y = y
        self.isgoal = isgoal
