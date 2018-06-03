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
                nsteps, ent_coef, vf_coef, max_grad_norm, cnn):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False, cnn=cnn)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True, cnn=cnn)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)
        adam_params = trainer.variables()

        def train(lr, cliprange, obs, obs2, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, train_model.X2:obs2, A:actions, ADV:advs, R:returns, LR:lr,
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

    def __init__(self, *, env, model, nsteps, gamma, lam, nmixup):
        self.env = env
        self.model = model
        nenv = env.num_envs

        self.obs = np.zeros((nenv,) + env.observation_space[0].shape, dtype=model.train_model.X.dtype.name)
        self.obs2 = np.zeros((nenv,) + env.observation_space[1].shape, dtype=model.train_model.X2.dtype.name)

        tmp = env.reset()
        self.obs[:] = tmp[:, 0].tolist()
        self.obs2[:] = tmp[:, 1].tolist()

        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.nmixup = nmixup
        self.mixup_time = True

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_obs2 = []
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.obs2, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_obs2.append(self.obs2.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            tmp, rewards, self.dones, infos = self.env.step(actions)
            self.obs[:] = tmp[:, 0].tolist()
            self.obs2[:] = tmp[:, 1].tolist()

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_obs2 = np.asarray(mb_obs2, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.obs2, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        # mixup episodes
        ne = self.env.num_envs
        # TODO: broken
        for _ in range(self.nmixup):
            w0 = np.random.beta(5, 1, size=(self.nsteps)).astype(np.float32)
            m0 = np.random.choice(ne, size=(self.nsteps, 2))
            m1 = m0[:, 0]
            m2 = m0[:, 1]
            i1 = np.arange(self.nsteps)
            i2 = np.arange(self.nsteps)
            if self.mixup_time:
                np.random.shuffle(i1)
                np.random.shuffle(i2)
            wo = w0.reshape((-1, 1, 1, 1))

            mix_obs = np.asarray(wo * mb_obs[i1, m1] + (1 - wo) * mb_obs[i2, m2], dtype=self.obs.dtype)
            mix_obs2 = np.asarray(wo * mb_obs2[i1, m1] + (1 - wo) * mb_obs2[i2, m2], dtype=self.obs.dtype)

            mix_returns = w0 * mb_returns[i1, m1] + (1 - w0) * mb_returns[i2, m2]
            m3 = np.select([w0 > 0.5, True], [m1, m2])
            i3 = np.select([w0 > 0.5, True], [i1, i2])
            mix_dones = mb_dones[i3, m3]
            mix_actions = mb_actions[i3, m3]
            mix_values = w0 * mb_values[i1, m1] + (1 - w0) * mb_values[i2, m2]
            mix_neglogpacs = w0 * mb_neglogpacs[i1, m1] + (1 - w0) * mb_neglogpacs[i2, m2]
            # evaluate values and neglogpacs for new mixup observations
            #mix_values = mb_values[i3, m3].copy()
            #mix_neglogpacs = mb_neglogpacs[i3, m3].copy()
            # TODO: build independet value and neglogp functions with apropriate batch size
            #for t in range(0, self.nsteps, ne):
            #    if t+ne <= self.nsteps:
            #        mix_v, mix_p = self.model.value_and_neglogp(mix_obs[t:t+ne], mix_actions[t:t+ne])
            #        mix_values[t:t+ne] = mix_v
            #        mix_neglogpacs[t:t+ne] = mix_p
            # insert the mixup episode
            mb_obs = np.concatenate((mb_obs, mix_obs[:,np.newaxis,:,:,:]), axis=1)
            mb_returns = np.concatenate((mb_returns, mix_returns[:,np.newaxis]), axis=1)
            mb_dones = np.concatenate((mb_dones, mix_dones[:,np.newaxis]), axis=1)
            mb_actions = np.concatenate((mb_actions, mix_actions[:,np.newaxis]), axis=1)
            mb_values = np.concatenate((mb_values, mix_values[:,np.newaxis]), axis=1)
            mb_neglogpacs = np.concatenate((mb_neglogpacs, mix_neglogpacs[:,np.newaxis]), axis=1)
        # TODO: need to fegure out how to mixup rewards
        #mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_obs2, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()


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

    make_model = lambda : Model(
        policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
        nbatch_train=nbatch_train, nsteps=nsteps, ent_coef=ent_coef,
        vf_coef=vf_coef, max_grad_norm=max_grad_norm, cnn=cnn)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()

    cur_w = 0
    choose_weights = False
    if weights_path is not None:
        if isinstance(weights_path, str):
            model.load(weights_path, adam_stats)
        elif isinstance(weights_path, list):
            cur_w = 0
            choose_weights = True
            model.load(weights_path[0], adam_stats)
            w_res = defaultdict(list)
            w_params = {}

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, nmixup=nmixup)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    checkdir = osp.join(logger.get_dir(), 'checkpoints')
    os.makedirs(checkdir, exist_ok=True)

    nupdates = total_timesteps//nbatch
    skip_first = False
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, obs2, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []
        if not policy.recurrent: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, obs2, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        # TODO: broken
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

        if choose_weights:
            # ugly but fast
            rewards = [epinfo['r'] for epinfo in epinfos if 'r' in epinfo]

            if cur_w == 0:
                w_res[cur_w].extend(rewards)
            elif skip_first and len(rewards) > 0:
                w_res[cur_w].extend(rewards[1:])
                skip_first = False
            else:
                w_res[cur_w].extend(rewards)
                skip_first = False

            # enough episodes
            if len(w_res[cur_w]) > weights_choose_eps:
                # here we will choose best weights
                w_params[cur_w] = model.get_params()
                cur_w += 1
                skip_first = True

                # load next weights
                if cur_w < len(weights_path):
                    model.load(weights_path[cur_w], adam_stats)

            # tested all weights, time to choose best
            if cur_w == len(weights_path):
                # TODO: may be can choose better criteria
                def _criteria(rews):
                    return np.mean(rews)

                best_score = -np.inf
                best_id = None
                logger.info(w_res)
                for i in w_res:
                    score = _criteria(w_res[i])
                    if score > best_score:
                        best_score = score
                        best_id = i

                assert best_id is not None

                # choose best weights
                model.load_dict(w_params[best_id], adam_stats)

                # do not enter in this block anymore
                choose_weights = False

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
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