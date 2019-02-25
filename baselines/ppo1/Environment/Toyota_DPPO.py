"""
A simple version of OpenAI's Proximal Policy Optimization (PPO). [https://arxiv.org/abs/1707.06347]

Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.

The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]

View more on my tutorial website: https://morvanzhou.github.io/tutorials

Dependencies:
tensorflow r1.3
gym 0.9.2
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import threading, queue
import os
import shutil
from Environment import environment

height, width = 30, 30
maximal_path = height + width
vehNum = 3
T_s = 0.1
minimal_velocity = 30/3.6
accel_max = 3
accel_min = -7
EP_MAX = 10000
EP_LEN = 100                # int(maximal_path/(minimal_velocity*T_s))
N_WORKER = 8                # parallel workers
GAMMA = 0.99                # reward discount factor
A_LR = 0.0002               # learning rate for actor
C_LR = 0.0002               # learning rate for critic
MIN_BATCH_SIZE = 128        # minimum batch size for updating PPO
UPDATE_STEP = 10            # loop update operation n-steps
EPSILON = 0.2               # for clipping surrogate objective
S_DIM, A_DIM = vehNum * 2, vehNum         # state and action dimension
r_normalization = 500
pattern = [2, 10, 4]
hiddenUnitNum = 512
saveFreq = 1000
layerNum = 2
sigma_coeff = 1
selected_tensorboard_dir = "_".join(['vehNum'+str(vehNum), 'ALR'+str(A_LR), 'CLR'+str(C_LR), 'clip'+str(EPSILON),
                                     'layerNum'+str(layerNum), 'hiddenNum'+str(hiddenUnitNum),
                                     'batchNum'+str(MIN_BATCH_SIZE), 'sigma_coeff'+str(sigma_coeff)])


class PPO(object):
    def __init__(self, Model_Flag, Model_Path):
        if not os.path.exists('summaries'):
            os.mkdir('summaries')

        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        self.critic_layer_ids = []
        clayer = self.tfs
        for lid in range(1, layerNum):
            self.critic_layer_ids.append('cl'+str(lid))
            clayer = tf.layers.dense(clayer, hiddenUnitNum, tf.nn.elu, name='cl'+str(lid))
        self.critic_layer_ids.append('cout')
        self.v = tf.layers.dense(clayer, 1, name='cout')

        self.all_summaries = []  # w b hist in each layers in critic net and actor net
        for clid in self.critic_layer_ids:
            with tf.name_scope(clid + '_hist'):
                with tf.variable_scope(clid, reuse=True):
                    w, b = tf.get_variable('kernel'), tf.get_variable('bias')
                    # Create a histogram summary object for the loss so it can be displayed
                    tf_w_hist = tf.summary.histogram('weights_hist', tf.reshape(w, [-1]))
                    tf_b_hist = tf.summary.histogram('bias_hist', b)
                    self.all_summaries.extend([tf_w_hist, tf_b_hist])
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        self.actor_layer_ids = []
        pi, pi_params, self.mu, self.sigma = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params, self.oldmu, self.oldsigma = self._build_anet('oldpi', trainable=False)
        self.tf_param_summaries = tf.summary.merge(self.all_summaries)

        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action 1-d list
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))
        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        with tf.name_scope('performance'):
            # c_loss = tf.placeholder(tf.float32, shape=None, name='loss_summary')
            c_loss_summary = tf.summary.scalar('c_loss', self.closs)
            # tf_accuracy_ph = tf.placeholder(tf.float32, shape=None, name='accuracy_summary')
            a_loss_summary = tf.summary.scalar('a_loss', self.aloss)
            self.moving_average = tf.placeholder(tf.float32, shape=None, name='moving_average')
            moving_average_summary = tf.summary.scalar('moving_average_summary', self.moving_average)
            self.nomoving_average = tf.placeholder(tf.float32, shape=None, name='moving_average')
            no_moving_average_summary = tf.summary.scalar('no_moving_average_summary', self.nomoving_average)
        self.performance_summaries = tf.summary.merge([c_loss_summary, a_loss_summary])
        self.MoveAverage_summaries = tf.summary.merge([moving_average_summary, no_moving_average_summary])

        with tf.name_scope('network_out'):
            self.c_net_out = tf.placeholder(tf.float32, shape=[None, 1], name='c_net_out')
            c_net_out_summ = tf.summary.histogram('c_net_out_summ', tf.reshape(self.c_net_out, [-1]))
            self.a_net_out = tf.placeholder(tf.float32, shape=[None, vehNum], name='a_net_out')
            a_net_out_summ = tf.summary.histogram('a_net_out_summ', tf.reshape(self.a_net_out, [-1]))
            # 这里a_net_out输出的是一个batch中选择的动作的分布 而不是网络输出的均值方差
        self.network_out_summaries = tf.summary.merge([c_net_out_summ, a_net_out_summ])

        if Model_Flag:
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, Model_Path)
        else:
            self.sess.run(tf.global_variables_initializer())

        tb_obj_dir = os.path.join('summaries', selected_tensorboard_dir)
        if not os.path.exists(tb_obj_dir):
            os.mkdir(tb_obj_dir)
        elif os.listdir(tb_obj_dir):
            shutil.rmtree(tb_obj_dir)
            os.mkdir(tb_obj_dir)
        self.summ_writer = tf.summary.FileWriter(tb_obj_dir, self.sess.graph)

    def update(self):
        global GLOBAL_UPDATE_COUNTER, GLOBAL_EPOCH
        while not COORD.should_stop():
            GLOBAL_EPOCH += 1
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})

                # write tensorboard log
                c_net_out = self.sess.run(self.v, {self.tfs: s})
                network_out, perf, wbsumm = self.sess.run([self.network_out_summaries, self.performance_summaries, self.tf_param_summaries],
                                                          {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tfdc_r: r, self.c_net_out: c_net_out, self.a_net_out: a})
                self.summ_writer.add_summary(network_out, GLOBAL_EPOCH)
                self.summ_writer.add_summary(perf, GLOBAL_EPOCH)
                self.summ_writer.add_summary(wbsumm, GLOBAL_EPOCH)


                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                # v_out_for_debug = self.sess.run(self.v, {self.tfs: s})
                # mu_out_for_debug = self.sess.run(self.mu, {self.tfs: s})
                # sigma_out_for_debug = self.sess.run(self.sigma, {self.tfs: s})
                # print('state', s[0:5, :])
                # print('value', v_out_for_debug[1:5, 0])
                # print('mu', mu_out_for_debug[1:5, :])
                # print('sigma', sigma_out_for_debug[1:5, :])
                #if GLOBAL_EP % saveFreq == 1:
                #    saver.save(GLOBAL_PPO.sess, './Model/model.ckpt', global_step=GLOBAL_EP, write_meta_graph=False)
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            alayer = self.tfs
            for lid in range(1, layerNum):
                if name == 'pi':
                    self.actor_layer_ids.append('al' + str(lid))
                alayer = tf.layers.dense(alayer, hiddenUnitNum, tf.nn.elu, name='al' + str(lid), trainable=trainable)
            if name == 'pi':
                self.actor_layer_ids.append('a_mu')
                self.actor_layer_ids.append('a_sigma')
            mu = 1.5 * tf.layers.dense(alayer, A_DIM, tf.nn.tanh, name='a_mu', trainable=trainable)
            sigma = sigma_coeff * tf.layers.dense(alayer, A_DIM, tf.nn.sigmoid, name='a_sigma', trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
            if name == 'pi':
                for alid in self.actor_layer_ids:
                    with tf.name_scope(alid + '_hist'):
                        with tf.variable_scope(alid, reuse=True):
                            w, b = tf.get_variable('kernel'), tf.get_variable('bias')
                            # Create a histogram summary object for the loss so it can be displayed
                            tf_w_hist = tf.summary.histogram('weights_hist', tf.reshape(w, [-1]))
                            tf_b_hist = tf.summary.histogram('bias_hist', b)
                            self.all_summaries.extend([tf_w_hist, tf_b_hist])
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params, mu, sigma

    def choose_action(self, s): # s must be a 1-d vector
        s = s[np.newaxis, :] # turn 1-d vector to 2-d matrix, i.e. [[1,2,3,4]]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0] # 1-d vector
        return np.clip(a, accel_min, accel_max)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = environment.Env(vehNum, height, width)
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            s = self.env.manualSet(pattern)
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            endFlag = 'counter end'
            for t in range(EP_LEN):
                # self.env.showEnv()
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data
                a = self.ppo.choose_action(s)
                s_, r, badendFlag, goodendFlag = self.env.updateEnv(a.tolist())
                if badendFlag: endFlag = 'bad'
                if goodendFlag: endFlag = 'good'
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r/r_normalization)                    # normalize reward, find to be useful
                s = s_
                #print(endFlag,r)
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1                 # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or badendFlag == 1 or goodendFlag == 1:
                    if badendFlag == 1 or goodendFlag == 1:
                        v_s_ = 0
                    else:
                        v_s_ = self.ppo.get_v(s_)
                    discounted_r = []                           # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                        UPDATE_EVENT.set()
                        break
                    if badendFlag == 1 or goodendFlag == 1:
                        break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.99+ep_r*0.01)
            GLOBAL_EP += 1
            # print('{0:.1f}%'.format(GLOBAL_EP / EP_MAX * 100), '|W%i' % self.wid, '|Ep_r: %.2f' % ep_r,
            #      '|End_flag：' + endFlag)

            # write tensorboard log
            MoveAverage_summaries = self.ppo.sess.run(self.ppo.MoveAverage_summaries,
                                                      {self.ppo.moving_average: GLOBAL_RUNNING_R[-1],
                                                       self.ppo.nomoving_average: ep_r})
            self.ppo.summ_writer.add_summary(MoveAverage_summaries, GLOBAL_EP)


if __name__ == '__main__':
    for A_LR in [0.0001, 0.0002]:
        for C_LR in [0.0001, 0.0002]:
            for hiddenUnitNum in [64, 128, 512]:
                selected_tensorboard_dir = "_".join(
                    ['vehNum' + str(vehNum), 'pattern' + str(pattern), 'ALR' + str(A_LR), 'CLR' + str(C_LR), 'clip' + str(EPSILON),
                     'layerNum' + str(layerNum), 'hiddenNum' + str(hiddenUnitNum),
                     'batchNum' + str(MIN_BATCH_SIZE)])
                print(selected_tensorboard_dir)
                GLOBAL_PPO = PPO(0, './Model/model.ckpt')
                UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
                UPDATE_EVENT.clear()            # not update now
                ROLLING_EVENT.set()             # start to roll out
                workers = [Worker(wid=i) for i in range(N_WORKER)]

                GLOBAL_UPDATE_COUNTER, GLOBAL_EP, GLOBAL_EPOCH = 0, 0, 0
                GLOBAL_RUNNING_R = []
                COORD = tf.train.Coordinator()
                QUEUE = queue.Queue()           # workers putting data in this queue
                threads = []
                saver = tf.train.Saver(max_to_keep=30)
                for worker in workers:          # worker threads
                    t = threading.Thread(target=worker.work, args=())
                    t.start()                   # training
                    threads.append(t)
                # add a PPO updating thread
                threads.append(threading.Thread(target=GLOBAL_PPO.update, args=()))
                threads[-1].start()
                COORD.join(threads)
                GLOBAL_PPO.sess.close()
                tf.reset_default_graph()

    # plot reward change and test
    '''plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode'); plt.ylabel('Moving reward'); plt.show()
    env1 = environment.Env(vehNum, height, width)
    while True:
        s = env1.manualSet(pattern)
        for t in range(500):
            env1.showEnv()
            s, r, badendFlag, goodendFlag = env1.updateEnv(GLOBAL_PPO.choose_action(s).tolist())  # s is a 1-D list
            if badendFlag == 1 or goodendFlag == 1:
                break '''
