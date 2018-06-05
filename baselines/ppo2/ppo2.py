import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
from collections import defaultdict


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, cnn, model_name):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False, cnn=cnn, model_name=model_name)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True, cnn=cnn, model_name=model_name)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])
        W = tf.placeholder(tf.float32, [None])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy() * W)

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2) * W)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2) * W)
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope(model_name):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)
        adam_params = trainer.variables()

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, w, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr, W:w,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def get_params():
            ps = sess.run(params)
            adam_ps = sess.run(adam_params)
            return {'ps': ps, 'adam_ps': adam_ps}

        def save(save_path):
            joblib.dump(self.get_params(), save_path)

        def _restore(tensors, vals):
            restores = []
            for p, loaded_p in zip(tensors, vals):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        def load_dict(d, adam_stats='none'):
            loaded_params = d['ps']
            _restore(params, loaded_params)

            if adam_stats is {'all', 'weight_stats'}:
                loaded_params = d['adam_ps']
                if adam_stats == 'weight_stats':
                    _restore(adam_params[2:], loaded_params[2:])
                else:
                    _restore(adam_params, loaded_params)

        def load(load_path, adam_stats='none'):
            d = joblib.load(load_path)
            loaded_params = d['ps']
            _restore(params, loaded_params)

            if adam_stats is {'all', 'weight_stats'}:
                loaded_params = d['adam_ps']
                if adam_stats == 'weight_stats':
                    _restore(adam_params[2:], loaded_params[2:])
                else:
                    _restore(adam_params, loaded_params)

            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        self.load_dict = load_dict
        self.get_params = get_params
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101


class Runner(object):

    def __init__(self, *, env, models, nsteps, gamma, lam, nmixup):
        self.env = env
        self.models = models
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=models[0].train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = models[0].initial_state
        self.dones = [False for _ in range(nenv)]
        self.nmixup = nmixup
        self.mixup_time = True
        self.n_models = len(models)

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_dones,  = [], [], [], []
        mb_values_lst = [[] for _ in range(self.n_models)]
        actions_all_logs_lst = [[] for _ in range(self.n_models)]
        actions_ens_logs = []

        mb_states = self.states
        epinfos = []

        for _ in range(self.nsteps):

            # iterate all models
            actions_all_logs = []
            for i, m in enumerate(self.models):
                actions, actions_logs, values, self.states, neglogpacs = m.step(self.obs, self.states, self.dones)

                # имеют values,  actions, actions_logs первый shape имеют 1
                mb_values_lst[i].append(values[0])
                actions_all_logs.append(actions_logs[0])
                actions_all_logs_lst[i].append(actions_logs[0])

            tmp = np.sum(actions_all_logs, axis=0)
            tmp = tmp - max(tmp)
            tmp = np.exp(tmp)
            actions_probs = tmp / tmp.sum()
            # actions_probs = np.mean(actions_all_logs, axis=0).ravel()

            actions = np.random.choice(len(actions_probs), p=actions_probs)

            # логарифм вероятности ансамбля
            actions_ens_logs.append(np.log(actions_probs[actions]))

            # observations, actions and dones for all models same
            mb_obs.append(self.obs.copy()[0])
            mb_actions.append(actions)
            mb_dones.append(self.dones[0])

            self.obs[:], rewards, self.dones, infos = self.env.step(np.asarray([actions]))
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

            mb_rewards.append(rewards[0])

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.int32)

        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        # shape (n_models, 1)
        last_values_lst = []
        for m in self.models:
            last_values_lst.append(m.value(self.obs, self.states, self.dones)[0])

        # this is one, per model
        # shape (n_models, n_steps,)
        mb_values_lst = np.asarray(mb_values_lst, dtype=np.float32)
        actions_all_logs_lst = np.asarray(actions_all_logs_lst)

        #discount/bootstrap off value fn
        mb_returns_lst = []
        mb_advs_lst = []
        mb_neglogpacs_lst = []
        mb_weights = []

        off_policy_logs = np.cumsum(actions_ens_logs[::-1])[::-1]
        # TODO: check all shapes
        for i in range(self.n_models):
            mb_advs = np.zeros_like(mb_rewards)
            lastgaelam = 0
            last_values = last_values_lst[i]
            mb_values = mb_values_lst[i]

            # надо выбрать neglogprob исходим из допущения, что одна среда
            actions_all_logs = actions_all_logs_lst[i]

            log_probs = actions_all_logs[np.arange(self.nsteps), mb_actions.ravel()]

            on_policy_logs = np.cumsum(log_probs[::-1])[::-1]
            mb_weights.append(np.exp(on_policy_logs - off_policy_logs))

            mb_neglogpacs_lst.append(-1 * log_probs)

            for t in reversed(range(self.nsteps)):
                if t == self.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones[0]
                    nextvalues = last_values
                else:
                    nextnonterminal = 1.0 - mb_dones[t+1]
                    nextvalues = mb_values[t+1]
                delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

            mb_returns = mb_advs + mb_values

            mb_returns_lst.append(mb_returns)
            mb_advs_lst.append(mb_advs)

        return mb_obs, mb_returns_lst, mb_dones, mb_actions, mb_values_lst, \
               mb_neglogpacs_lst, mb_states, epinfos, mb_weights


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val
    return f


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, weights_path=None, adam_stats='all', nmixup=1,
            weights_choose_eps=10, cnn='cnn'):

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    nbatch = nenvs * nsteps
    if policy.recurrent:
        envsperbatch = max(nenvs // nminibatches, 1)
        nbatch_train = envsperbatch * nsteps
    else:
        nbatch_train = nbatch // nminibatches

    logger.info("batch size: {}".format(nbatch_train))

    models = []
    for i, w in enumerate(weights_path):
        make_model = lambda: Model(
            policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
            nbatch_train=nbatch_train, nsteps=nsteps, ent_coef=ent_coef,
            vf_coef=vf_coef, max_grad_norm=max_grad_norm, cnn=cnn, model_name='model_{}'.format(i))

        model = make_model()
        model.load(w, adam_stats)
        models.append(model)

    # if save_interval and logger.get_dir():
    #     import cloudpickle
    #     with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
    #         fh.write(cloudpickle.dumps(make_model))


    # cur_w = 0
    # choose_weights = False
    # if weights_path is not None:
    #     if isinstance(weights_path, str):
    #         model.load(weights_path, adam_stats)
    #     elif isinstance(weights_path, list):
    #         cur_w = 0
    #         choose_weights = True
    #         model.load(weights_path[0], adam_stats)
    #         w_res = defaultdict(list)
    #         w_params = {}

    runner = Runner(env=env, models=models, nsteps=nsteps, gamma=gamma, lam=lam, nmixup=nmixup)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    checkdir = osp.join(logger.get_dir(), 'checkpoints')
    os.makedirs(checkdir, exist_ok=True)

    nupdates = total_timesteps//nbatch

    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, states, epinfos, mb_weights = runner.run()
        epinfobuf.extend(epinfos)
        mblossvals = []
        if not policy.recurrent: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]

                    # TODO: may be optimize loops
                    for i, m in enumerate(models):
                        slices = (arr[mbinds] for arr in
                                  (obs, returns[i], masks, actions, values[i], neglogpacs[i], mb_weights[i]))
                        mblossvals.append(m.train(lrnow, cliprangenow, *slices))

        else: # recurrent version
            # assert nenvs % nminibatches == 0
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for end in range(envsperbatch, nenvs, envsperbatch):
                    start = end - envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            # ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            # logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf if 'r' in epinfo]))
            logger.logkv('eprewmean_exp_x', safemean([epinfo['r_exp_x'] for epinfo in epinfobuf if 'r_exp_x' in epinfo]))
            logger.logkv('eprewmean_exp_obs', safemean([epinfo['r_exp_obs'] for epinfo in epinfobuf if 'r_exp_obs' in epinfo]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf if 'l' in epinfo]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)

        if save_interval:
            # save last weights
            savepath = osp.join(checkdir, "last")
            model.save(savepath)

    env.close()


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)