"""
Reinforcement learning environment for long-horizon IoT sensor-network management.

This module defines a custom Gymnasium environment for battery-aware sensor scheduling
in a resource-constrained IoT/WSN system. The environment simulates node-level battery
state, solar energy harvesting, randomized energy consumption, battery degradation,
wireless connectivity uncertainty, sensing feasibility, and application-level service
quality (i.e., modal assurance criteria (MAC)). The environment is designed for studying
deployment-relevant reinforcement learning mechanisms.


The environment supports both:
    - value-based control with DQN using a flattened discrete action space, and
    - policy-gradient control with PPO-family algorithms using a MultiDiscrete
      node-level action space.

Optional action masking is provided for feasibility-aware control, where invalid
actions can be masked based on battery and network-operation constraints.
"""


# Import all the required packages
import matplotlib.pyplot as plt
from gymnasium import spaces, Env
from gymnasium.utils import seeding
import random
import numpy as np
import os
from env.data_loader import load_pvdata, load_conn_prob, load_temperature, load_modeshape
from env.energy import compute_energy_consumption, generate_randomized_pv_temp
from env.env_output import observation_shaping, reward_shaping
from env.utils import dqn_decode_action, sensing_feasibility,\
evaluate_mac_qos, log_step, take_ppo_action
from env.battery_model import compute_battery_degradation


class Jindo(Env):
    """
        The Jindo environment models a resource-constrained IoT sensing network in which
        multiple wireless sensor nodes must be scheduled over a long operating horizon.

        Notes
        -----
        This environment follows the Gymnasium API and returns:

        - ``reset() -> (observation, info)``
        - ``step(action) -> (observation, reward, terminated, truncated, info)``

        The current implementation uses ``terminated`` for end-of-episode completion
        and always returns ``False`` for ``truncated``.

        """
    curr_step = 0

    def __init__(self, mode,config, enable_action_masking=False):
        """
            Initialize the long-horizon IoT sensor-network environment.

            This method loads environmental data, initializes battery and degradation
            states, defines the action and observation spaces, and prepares the environment
            for training, validation, or testing.

            Parameters
            ----------
            mode : str
                Simulation mode. Expected values are:
                ``"train"``, ``"val"``, or ``"test"``.
                    The selected mode determines the episode length and input data split.
                    Use ``"train"`` for training episodes, ``"val"`` for validation/evaluation,
                    and ``"test"`` for final policy testing.

            config : dict
                Configuration dictionary used to initialize the environment. The dictionary
                should contain:

                ``seed`` : int or None
                    Random seed for reproducibility.

                ``env_config`` : dict
                    Environment-level settings such as number of sensors, time slice,
                    battery capacity, battery voltage, minimum battery level, and power
                    consumption parameters.

                ``rl_config`` : dict
                    RL-level settings such as algorithm name and episode lengths for
                    training, validation, and testing.

                ``data_dir`` : str
                    Directory containing PV, temperature, connectivity, and mode-shape data.

                ``generated_data_dir`` : str
                    Directory where environment outputs and evaluation logs are saved.

                ``file_no`` : int or str
                    Identifier used when saving output logs.


            enable_action_masking : bool, optional
                Whether to enable action masking for feasibility-aware control
                through :meth:`get_action_mask`.This is mainly intended for Maskable PPO.
                Default is False.

        Attributes
        ----------
            action_space : gymnasium.Space
                RL action space. Uses ``spaces.Discrete`` for DQN and
                ``spaces.MultiDiscrete`` for PPO-family algorithms.

            observation_space : gymnasium.spaces.Dict
                Observation space with local node-level features and global system-level
                features.

        Raises
        ------
        KeyError
            If required configuration keys are missing.
        """

        self.config = config
        seed_no = config['seed']
        env_config = config['env_config']
        rl_config = config['rl_config']
        self.file_no = config['file_no']
        if seed_no is not None:
            self.seed(seed_no)
        self.SEED = seed_no
        self.enable_action_masking = enable_action_masking
        self.totalCHnum = env_config["sensor_count"]  # Total number of channels in the network
        data_dir = config['data_dir']
        self.connProb = load_conn_prob(self.totalCHnum)
        self.tempdata = load_temperature(data_dir, mode)
        self.temp_max = max(self.tempdata)
        self.tdata = load_pvdata(data_dir, mode).T
        self.pv_max = max(self.tdata)
        self.mdata = load_modeshape(data_dir, self.totalCHnum)

        self.mode = mode  # Train, val or test
        
        # Each step and episodes variables
        if mode == 'train': self.stepn = rl_config['train_steps_per_episode']
        elif mode == 'val': self.stepn = rl_config['eval_total_timesteps']
        elif mode =='test': self.stepn = rl_config['test_total_timesteps']
            
        self.episode_N = 0  # number of current episode
        self.step_in_episode = 0
        self.reward = 0
        self.tmp = 0

        # pv dataset parameters
        self.t_k = env_config["time_slice"] # time slice in hours
        self.pvarray = 0
        self.temp_array = 0
        self.algorithm = rl_config["rl_algorithm"]

        # Battery health parameters
        self.Batt_max = env_config["battery_capacity"] * env_config["battery_nom_voltage"]  # (in mWh)
        self.Batt_cap = np.ones(self.totalCHnum) * self.Batt_max
        self.Batt_level = np.ones(self.totalCHnum) * self.Batt_max  # (in mWh)
        self.Bt_diff = np.zeros(self.totalCHnum)  # (in mWh)
        self.Bthistory = []  # (in mWh)
        self.Bt_min = env_config["battery_min"] * env_config["battery_nom_voltage"] # in (mWh)
        self.pv_total = np.zeros(self.stepn) 
        self.energy_harvest = np.zeros(self.totalCHnum)  # Harvested Energy

        # For SHM-A board on Imote2 (4.5V DC power supply, 104MHz)
        self.params = {}
        self.Pa = env_config["P_active"]  # Consumed power during active observation (mW)
        self.Pi = env_config["P_idle"]  # Consumed power during idle observation (mW)
        self.Ps = env_config["P_sleep"]  # Consumed power during deep-sleep observation (mW)
        self.params = {
            'Pa': self.Pa, 'Pi': self.Pi, 'Ps': self.Ps}

        # Battery degradation parameters
        self.cycle = np.zeros(self.totalCHnum)
        self.fec = np.zeros(self.totalCHnum)
        self.del_fec = np.zeros(self.totalCHnum)
        self.dg = np.zeros(self.totalCHnum)
        self.dg_current = np.zeros(self.totalCHnum)
        self.capacity_loss_cyclic = np.zeros(self.totalCHnum)
        self.capacity_loss_cal = np.zeros(self.totalCHnum)

        # Battery status parameters
        self.sleepCT = np.zeros(self.totalCHnum)
        self.nodeStatus = np.zeros(self.totalCHnum, dtype=int)
        self.node_mac = np.ones(self.totalCHnum)

        # Connectivity parameters for all nodes
        self.successlist = np.zeros(self.totalCHnum)

        self.mac = np.ones(5)
        self.modeshape = np.zeros((self.totalCHnum, 5), float)

        # ------------------------------------------------------------------
        # Action-space definition
        # ------------------------------------------------------------------
        # For DQN, the action is represented as a single discrete index that
        # encodes one node-level status change.
        #
        # For PPO-family algorithms, the action is represented as a
        # MultiDiscrete vector with one action per node.
        #
        # Node status convention:
        #   0 = active
        #   1 = idle
        #   2 = sleep
        #
        # PPO-family node-level action convention:
        #   0 = active toggle
        #   1 = sleep toggle
        #   2 = no action / maintain status
        # ------------------------------------------------------------------

        if self.algorithm == 'dqn':
            self.action_space = spaces.Discrete(2 * self.totalCHnum + 1)  # 2 toggle cases at each node

        else:
            actspace = []
            for i in range(self.totalCHnum):
                actspace.append(3)
            self.action_space = spaces.MultiDiscrete(actspace)  # three discrete actions at each node

        for k in range(self.totalCHnum):
            exec(f'self.act{k} = 0')

        # ------------------------------------------------------------------
        # Observation-space definition
        # ------------------------------------------------------------------
        # The observation is represented as a dictionary with:
        #
        #   local  : node-level features with shape (num_local_features, 1, num_nodes)
        #   global : system-level features with shape (num_global_features,)
        #
        # This structure supports CNN-style processing of local node features
        # while preserving global context such as time, weather, or network-level
        # summary information.
        # ------------------------------------------------------------------

        self.observation_space = spaces.Dict({
                    "local": spaces.Box(low=-0.01, high=2.01, shape=(6, 1, self.totalCHnum), dtype=np.float32),
                    "global": spaces.Box(low=-1, high=1, shape=(11,), dtype=np.float32)
                })
        
        local_obs = np.zeros((6, 1, self.totalCHnum), float)
        global_obs = np.zeros((11,), dtype=np.float32)
        self.observation = {
        'local': local_obs.astype(np.float32),
        'global': global_obs}


    def seed(self, seed=None):
        self.np_random, seed_used = seeding.np_random(seed)
        print(f"Which seed is used in seed method:{seed_used}")
        return [seed_used]

    def step(self, action):
        self.successlist = self.np_random.binomial(1, self.connProb)  # Connectivity uncertainty
        # between agent and leaf nodes. Here, 1 = success, 0 = fail

        # Obtain node status from taken actions (dqn or other ppo family)
        if self.algorithm == 'dqn':
            node_index, node_new_status = dqn_decode_action(action, self.nodeStatus)

            if node_index is not None and self.successlist[node_index] == 1:
                self.nodeStatus[node_index] = node_new_status  # Change status if connection is succeeded

        else:
            for index, val in enumerate(self.successlist):
                if val == 1:  # if connection success
                    self.nodeStatus[index] = take_ppo_action(action[index], self.nodeStatus[index])
        
        # Check sensing feasibility in terms of battery active and idle node counts and battery level
        self.node_mac = sensing_feasibility(self.Batt_level,self.Bt_min, 
                                        self.successlist,self.nodeStatus, self.Batt_cap, self.dg)

        self.energy_harvest = self.pvarray[self.step_in_episode]

        Bt_tmp = self.Batt_level  # Current battery level in mWh

        consumption = compute_energy_consumption(self.nodeStatus, self.node_mac, self.params,
                                                 self.step_in_episode, self.t_k, self.np_random)
        self.Batt_level = self.Batt_level + self.energy_harvest - consumption
        self.Batt_level = np.clip(self.Batt_level, 0, self.Batt_cap)

        self.Bt_diff = Bt_tmp - self.Batt_level  # Battery consumption at the current step  (in mWh)

        # Battery time history accumulation
        Bttmp = []
        for i in range(self.totalCHnum):
            Bttmp.append(self.Batt_level[i])

        self.Bthistory.append(Bttmp)
        self.tmp = np.array(self.Bthistory)
        
        soc= np.ones(self.totalCHnum)
        rj= np.zeros(self.totalCHnum)
        crate= np.zeros(self.totalCHnum)
        fec_prev = self.fec.copy()
        cycle_prev= self.cycle.copy()
        del_fec_prev = self.del_fec.copy()
        dg_prev = self.dg.copy()
        dg_current_prev = self.dg_current.copy()
        temp_in_step =self.temp_array[self.step_in_episode]

        # Predict battery degradation for each channel:
        for i in range(self.totalCHnum):
            self.fec[i], self.cycle[i], dg1, dg2, soc[i],rj[i],crate[i],self.del_fec[i] = compute_battery_degradation(data=self.tmp[:, i],
                                                                  step_ep=self.step_in_episode+1,
                                                                  full_eq_cycle_before=fec_prev[i],
                                                                  cycle_t0=cycle_prev[i],
                                                                  delta_fec_before=del_fec_prev[i],
                                                                  pu_loss_t0=self.dg[i],
                                                                  temperature=temp_in_step,
                                                                  batt_cap=self.Batt_cap[i])
            self.capacity_loss_cyclic[i] += dg1
            self.capacity_loss_cal[i] += dg2
            self.dg[i] = abs(self.capacity_loss_cyclic[i] + self.capacity_loss_cal[i])
            self.dg[i] = np.clip(self.dg[i], 0, 0.99) 
            self.dg_current[i] = abs(dg1 + dg2)
            self.dg_current[i] = np.clip(self.dg_current[i], 0, 0.99) 
            """
            # For check
            if self.dg[i] >1:
                raise ValueError(f"Total degradation exceeded threshold: {self.dg[i]} > {1}")"""


        cycle_diff = self.cycle - cycle_prev
        # Find only active node location --> 1 being active 0--> being idle or sleep:
        ct = self.nodeStatus == 0  # only active count
        ct = ct.astype(int)
        
        sleep = (self.nodeStatus == 2).astype(int)
        self.sleepCT += sleep  # total active node count at each channels

        self.mac, self.modeshape = evaluate_mac_qos(self.mdata, self.nodeStatus, self.node_mac, self.totalCHnum)

        currnum = self.step_in_episode + 1
        self.reward = reward_shaping(self.algorithm, currnum, self.mac, self.dg, dg_prev, self.stepn, self.sleepCT)

        print("step", self.step_in_episode)
        done = self.step_in_episode >= self.stepn - 1
        if done:
            print('EPISODE DONE')
        
        pv_norm = self.pv_total[self.step_in_episode]/self.pv_max
        temp_norm = temp_in_step/self.temp_max


        self.observation = observation_shaping(self.Batt_level, self.Batt_cap, self.Bt_diff,
                                               self.nodeStatus,dg_prev, self.dg, cycle_diff, rj, self.node_mac, self.mac,
                                               self.slice_n, self.step_in_episode, self.stepn, pv_norm, temp_norm)
                                               

        #print(f'observation in step {currnum} is: {self.observation}')

        # increase step number in episode
        type(self).curr_step += 1
        self.step_in_episode += 1

        # ---------------------------------for output generation------------------------------------------
        output = np.concatenate(
            [np.array([self.step_in_episode, self.reward]), self.mac, self.Batt_level, self.node_mac,
             self.capacity_loss_cyclic, self.capacity_loss_cal, self.cycle, self.nodeStatus])
        """

        output2 = np.concatenate(
            [np.array([self.temp_array[self.step_in_episode - 1]]), soc, rj, crate, self.del_fec])"""


        # Save output
        log_step(output, self.mode, 360, self.config, file_no = self.file_no)  # controlling this can save total
        # training time significantly. For larger network increase the buffer_size to increase the frame rate.
        
        
        # Save output
        """
        filename = f'{self.algorithm}_{self.mode}_{self.totalCHnum}_output2.txt'
        file_path = os.path.join(self.config["generated_data_dir"], filename)
        with open(file_path, "ab") as f:  # was 'ab'
            f.write(b'\n')
            np.savetxt(f, output2[None, :], fmt='%5f', delimiter=',', newline='')"""

        info = {}
        
        return self.observation, self.reward, done, False, info


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        print(f"seed no in reset method: {seed}")
        print('----reset in training/testing---- step#: ', type(self).curr_step, ' episode#: ', int(self.episode_N))
        self.episode_N += 1
        self.step_in_episode = 0
        self.reward = 0
        self.sleepCT = np.zeros(self.totalCHnum)
        self.cycle = np.zeros(self.totalCHnum)
        self.fec = np.zeros(self.totalCHnum)
        self.dg = np.zeros(self.totalCHnum)
        self.dg_current = np.zeros(self.totalCHnum)
        cycle_diff = np.zeros(self.totalCHnum)
        soc = np.ones(self.totalCHnum)
        rj = np.zeros(self.totalCHnum)
        dg_prev = np.zeros(self.totalCHnum)
        self.capacity_loss_cyclic = np.zeros(self.totalCHnum)
        self.capacity_loss_cal = np.zeros(self.totalCHnum)
        self.Bthistory = []

        # Noise in energy consumption
        self.Pa = self.np_random.uniform(746, 766)
        self.Pi = self.np_random.uniform(368, 378)
        self.Ps = self.np_random.uniform(2, 2.5)

        self.params = {
            'Pa': self.Pa, 'Pi': self.Pi,'Ps': self.Ps}

        # Battery capacity variation to incorporate manufacturing variability
        #self.Batt_cap = np.ones(self.totalCHnum) * self.Batt_max  # (in mWh) # without capacity variation

        # with capacity variation
        batt_cap = np.ones(self.totalCHnum) * self.Batt_max
        for i in range(self.totalCHnum):
            batt_cap[i] = self.np_random.uniform(0.9, 1.1) * self.Batt_max
            self.Batt_level[i] = batt_cap[i]
        self.Batt_cap = batt_cap
        
        # Generate final PV array from Gaussian Field
        self.pv_total, self.pvarray, self.temp_array, self.slice_n = generate_randomized_pv_temp(
                                    self.mode, self.tdata, self.tempdata, self.stepn, 
                                    self.totalCHnum, self.np_random, self.SEED)
                                    
        pv_norm = self.pv_total[self.step_in_episode]/self.pv_max
        temp_norm = self.temp_array[self.step_in_episode]/self.temp_max

        self.observation = observation_shaping(self.Batt_level, self.Batt_cap, self.Bt_diff,
                                               self.nodeStatus,dg_prev, self.dg, cycle_diff, rj,self.node_mac,self.mac, 
                                               self.slice_n, self.step_in_episode, self.stepn, pv_norm,temp_norm)

        info = {}
        
        return self.observation, info

    def get_action_mask(self):
        """
        Returns a MultiDiscrete-compatible action mask with shape (num_nodes, 3).
        """
        # Start with all-True, shape (N, 3)
        mask = np.ones((self.totalCHnum, 3), dtype=bool)
        # Apply battery level threshold masking
        
        # Status flags
        sleep_node = (self.nodeStatus == 2)
        idle_node = (self.nodeStatus == 1)
        active_node = (self.nodeStatus == 0)

        # Battery thresholds
        # red_idle = battery threshold + energy required to sustain the node in idle condition for the time slice (i.e., 360)
        # red_active = battery threshold + energy required to sustain the node in active condition for the time slice
        req_idle = self.Bt_min + 360
        req_active = self.Bt_min + 965

        # Nodes that fail their respective thresholds
        bad_sleep = sleep_node & (self.Batt_level < req_idle)
        bad_idle = idle_node & (self.Batt_level < req_idle)
        bad_active = active_node & (self.Batt_level < req_active)

        # Mask out: [active toggle, non-action toggle]
        mask[bad_sleep, 0] = False
        mask[bad_sleep, 1] = False
        mask[bad_idle, 0] = False
        mask[bad_idle, 2] = False
        mask[bad_active, 0] = False
        mask[bad_active, 2] = False

        # Enforce min 3 and max 4 active node constraints

        current_active = np.sum(active_node.astype(int))
        if current_active > 3:
            mask[idle_node, 0] = False  # mask out 'active' toggle for idle
            mask[sleep_node, 0] = False  # mask out 'active' toggle for sleep"""
            
        """
        if current_active <= 2:
            mask[:, 1] = False  # mask out 'sleep' toggle
        """

        return mask.reshape(-1) #mask flattening (?)

    def render(self, mode='human'):
        """
        Render node statuses as a color-coded horizontal bar (Active = green, Idle = yellow, Sleep = gray)
        """
        color_map = {0: 'green', 1: 'orange', 2: 'red'}
        node_colors = [color_map.get(status, 'red') for status in self.nodeStatus]

        fig, ax = plt.subplots(figsize=(8, 1))
        ax.bar(range(len(self.nodeStatus)), [1]*len(self.nodeStatus), color=node_colors, edgecolor='black')
        ax.set_yticks([])
        ax.set_xticks(range(len(self.nodeStatus)))
        ax.set_xticklabels([f'N{i}' for i in range(len(self.nodeStatus))])
        ax.set_title(f"Step {self.step_in_episode} — Node States")
        plt.tight_layout()
        plt.show()
        plt.close()

"""

