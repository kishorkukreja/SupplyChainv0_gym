import gym
from gym.utils import seeding
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gym import spaces


class InventoryEnv(gym.Env):
    """
    General Inventory Control Environment.

    Currently tested with:
    - A reinforcement learning model for supply chain ordering management:
    An application to the beer game - Chaharsooghi (2002)
    """
    def __init__(self, case, action_low, action_high, action_min, action_max,
        state_low, state_high, method,
        coded=False, fix=True, ipfix=True):
        self.case = case
        self.case_name = case.__class__.__name__
        self.n = case.leadtime_ub + 1
        self.coded = coded
        self.fix = fix
        self.ipfix = ipfix
        self.method = method
        if self.method == 'DRL':
            self.action_low = action_low
            self.action_high = action_high
            self.action_min = action_min
            self.action_max = action_max
            self.state_low = state_low
            self.state_high = state_high
        self.determine_potential_actions()
        self.determine_potential_states()

    def seed (self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def generate_leadtime(self,t, dist, lowerbound, upperbound):
        """
        Generate the leadtime of the dataset from paper or distribution.

        Returns: Integer
        """
        if self.dist == 'uniform':
            self.leadtime = random.randrange(self.lowerbound, self.upperbound + 1)
        else:
            raise Exception
        return self.leadtime

    def determine_potential_actions(self):
        """
        Possible actions returned as Gym Space
        each period
        """
        self.feasible_actions = 0
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.int32)

    def determine_potential_states(self):
        """
        Based on the mean demand, we determine the maximum and minimum
        inventory to prevent the environment from reaching unlikely states
        # Observation space consists of the current timestep and inventory positions of every echelon
        """
        self.observation_space = spaces.Box(self.state_low, self.state_high, dtype=np.int32)

    def _generate_demand(self):
        """
        Generate the demand using a predefined distribution.

        Writes the demand to the orders table.
        """
        source, destination = np.nonzero(self.case.connections)
        for retailer, customer in zip(source[-self.case.no_customers:],destination[-self.case.no_customers:]):
            if self.case.demand_dist == 'poisson':
                demand_mean = random.randrange(self.case.demand_lb,self.case.demand_ub + 1)
                demand = np.random.poisson(demand_mean)
            elif self.case.demand_dist == 'uniform':
                demand = random.randrange(self.case.demand_lb,self.case.demand_ub + 1)
            self.O[0, customer, retailer] = demand

    def calculate_reward(self):
        """
        Calculate the reward for the current period.

        Returns: holding costs, backorder costs
        """
        backorder_costs = np.sum(self.BO[0] * self.case.bo_costs)
        holding_costs = np.sum(self.INV[0] * self.case.holding_costs)
        return holding_costs, backorder_costs

    def _initialize_state(self):
        """
        Initialize the inventory position for every node.

        Copies the inventory position from the previous timestep.
        """
        #(...)
        return None

    def _receive_incoming_delivery(self):
         """
         Receives the incoming delivery for every stockpoint.

         Customers are not taken into account because of zero lead time
         Based on the amount stated in T
         """
         # Loop over all suppliers and stockpoints
         for i in range(0, self.case.no_stockpoints + self.case.no_suppliers):
         # Loop over all stockpoints
         # Note that only forward delivery is possible, hence 'i+1'
             for j in range(i + 1, self.case.no_stockpoints +elf.case.no_suppliers):
                    delivery = self.T[0, i, j]
                    self.INV[0, j] += delivery
                    self.in_transit[0, i, j] -= delivery
                    self.T[0, i, j] = 0

    def _recieve_incoming_orders_customers(self):
        i_list, j_list = np.nonzero(self.case.connections)
        for i, j in zip(i_list[-self.case.no_customers:], j_list[-self.case.no_customers:]):
            if self.O[0, j, i] > 0:
                # Check if the current order can be fulfilled
                if self.INV[0, i] >= self.O[0, j, i]:
                    self._fulfill_order(i, j, self.O[0, j, i])
                    # Else, fulfill as far as possible
                else:
                    inventory = max(self.INV[0, i], 0)
                    quantity = self.O[0, j, i] - inventory
                    self._fulfill_order(i, j, inventory)
                    # Add to backorder if applicable
                    if self.case.unsatisfied_demand == 'backorders':
                        self.BO[0, j, i] += quantity
        if self.case.unsatisfied_demand == 'backorders':
            for i, j in zip(i_list[-self.case.no_customers:], j_list[-self.case.no_customers:]):
                inventory = self.INV[0, i]
                # If there are any backorders, fulfill them afterwards
                if inventory > 0:
                    # If the inventory is larger than the backorder
                    # Fulfill the whole backorder
                    backorder = self.BO[0, j, i]
                    if inventory >= backorder:
                        # Dit vind ik heel onlogisch, maar voorzover ik nu kan zien
                        # in de IPs komt de backorder nooit aan.
                        # Nu wel gedaan dmv fix
                        if self.fix:
                            self._fulfill_order(i, j, backorder)
                        else:
                            self.INV[0, i] -= backorder
                        self.BO[0, j, i] = 0
                    # Else, fulfill the entire inventory
                    else:
                        self._fulfill_order(i, j, inventory)
                        self.BO[0, j, i] -= inventory
        
        
    def _recieve_incoming_orders_customers(self):
        i_list, j_list = np.nonzero(self.case.connections)
        for i, j in zip(i_list[-self.case.no_customers:], j_list[-self.case.no_customers:]):
            if self.O[0, j, i] > 0:
            # Check if the current order can be fulfilled
                if self.INV[0, i] >= self.O[0, j, i]:
                    self._fulfill_order(i, j, self.O[0, j, i])
                    # Else, fulfill as far as possible
                else:
                    inventory = max(self.INV[0, i], 0)
                    quantity = self.O[0, j, i] - inventory
                    self._fulfill_order(i, j, inventory)
                    # Add to backorder if applicable
                    if self.case.unsatisfied_demand == 'backorders':
                        self.BO[0, j, i] += quantity
        if self.case.unsatisfied_demand == 'backorders':
            for i, j in zip(i_list[-self.case.no_customers:], j_list[-self.case.no_customers:]):
                inventory = self.INV[0, i]
                # If there are any backorders, fulfill them afterwards
                if inventory > 0:
                    # If the inventory is larger than the backorder
                    # Fulfill the whole backorder
                    backorder = self.BO[0, j, i]
                    if inventory >= backorder:
                    # Dit vind ik heel onlogisch, maar voorzover ik nu kan zien
                    # in de IPs komt de backorder nooit aan.
                    # Nu wel gedaan dmv fix
                        if self.fix:
                            self._fulfill_order(i, j, backorder)
                        else:
                            self.INV[0, i] -= backorder
                            self.BO[0, j, i] = 0
                            # Else, fulfill the entire inventory
                    else:
                        self._fulfill_order(i, j, inventory)
                        self.BO[0, j, i] -= inventory
                        
    def _recieve_incoming_orders_divergent(self):
    # Ship from supplier to warehouse
        self._fulfill_order(0, 1, self.O[0, 1, 0])
        # Check if the warehouse can ship all orders
        if self.INV[0, 1] >= np.sum(self.O[0, :, 1], 0):
            i_list, j_list = np.nonzero(self.case.connections)
            for i, j in zip(i_list[self.case.no_suppliers:self.case.no_suppliers+
                self.case.no_stockpoints],
                j_list[self.case.no_suppliers:self.case.no_suppliers+
                self.case.no_stockpoints]):
                if self.O[0, j, i] > 0:
                    self._fulfill_order(i, j, self.O[0, j, i])
        else:
            IPlist = {}
            i_list, _ = np.nonzero(self.O[0])
            bo_echelon = np.sum(self.BO[0], 0)
            for i in i_list:
                IPlist[i] = self.INV[0, i] - bo_echelon[i]
        # Check the lowest inventory position and sort these on lowest IP
        sorted_IP = {k: v for k, v in sorted(IPlist.items(), key=lambda item: item[1])}
        # Check if there is still inventory left
        if self.INV[0, 1] >= 0:
            for i in sorted_IP:
            # Check if the remaining order can be fulfilled completely
                if self.INV[0, 1] >= self.O[0, i, 1]:
                    self._fulfill_order(1, i, self.O[0, i, 1])
                else:
                    # Else, fulfill how far possible
                    inventory = max(self.INV[0, 1], 0)
                    quantity = self.O[0, i, 1] - inventory
                    self._fulfill_order(1, i, inventory)
                    break
                    
    def _fulfill_order(self, source, destination, quantity):
        # Customers don't have any lead time.
        if destination >= self.case.no_nodes - self.case.no_customers:
            leadtime = 0
        else:
            leadtime = self.leadtime
        # The order is fulfilled immediately for the customer
        # or whenever the leadtime is 0
        if leadtime == 0:
        # The new inventorylevel is increased with the shipped quantity
            self.INV[0, destination] += quantity
        else:
            # If the order is not fulfilled immediately, denote the time when
            # the order will be delivered. This can not be larger than the horizon
            if leadtime < self.n:
                self.T[leadtime, source, destination] += quantity
            else:
                raise NotImplementedError
                for k in range(0, min(leadtime, self.n) + 1):
                    self.in_transit[k, source, destination] += quantity
        # Suppliers have unlimited capacity
        if source >= self.case.no_suppliers:
            self.INV[0, source] -= quantity
            
    def _place_outgoing_order(self, t, action):
        k = 0
        incomingOrders = np.sum(self.O[0], 0)
        # Loop over all suppliers and stockpoints
        for j in range(self.case.no_suppliers, self.case.no_stockpoints +
            self.case.no_suppliers):
            RandomNumber = random.random()
            probability = 0
            for i in range(0, self.case.no_stockpoints + self.case.no_suppliers):
                if self.case.connections[i, j] == 1:
                    self._place_order(i,j,t,k, action, incomingOrders)
                    k += 1
                elif self.case.connections[i,j] > 0:
                    probability += self.case.connections[i,j]
                    if RandomNumber < probability:
                        self._place_order(i,j,t,k, action, incomingOrders)
                        k += 1
                        break
    def _place_order(self, i, j, t, k, action, incomingOrders):
        if self.case.order_policy == 'X':
            self.O[t, j, i] += action[k]
            if (self.t < self.case.horizon - 1) and (self.t >= self.case.warmup-1):
                self.TotalDemand[j,i] += action[k]
        elif self.case.order_policy == 'X+Y':
            self.O[t, j, i] += incomingOrders[j] + action[k]
            if (self.t < self.case.horizon - 1) and (self.t >= self.case.warmup-1):
                self.TotalDemand[j,i] += incomingOrders[j] + action[k]
        elif self.case.order_policy == 'BaseStock':
            bo_echelon = np.sum(self.BO[0], 0)
            self.O[t, j, i] += max(0, action[k]-(self.INV[0,j]+self.in_transit[0,i,j]-bo_echelon[j]))
            if (self.t < self.case.horizon - 1) and (self.t >= self.case.warmup-1):
                self.TotalDemand[j,i] += max(0,action[k]-self.INV[0,j]+self.in_transit[0,i,j]-bo_echelon[j])
        else:
            raise NotImplementedError
            
    def _code_state(self):
         #(...)
        return CIP

    def _check_action_space(self, action):
        if isinstance(self.action_space, spaces.Box):
            low = self.action_space.low
            high = self.action_space.high
            max = self.action_max
            min = self.action_min
            action_clip = np.clip(action, low, high)
            for i in range(len(action_clip)):
                action_clip[i] = ((action_clip[i]-low[i])/(high[i]-low[i]))*((max[i]-min[i]))+min[i]
                action = [np.round(num) for num in action_clip]
        return action
    
    def step(self, action, visualize=False):
        """
        Execute one step in the RL method.
        
        input: actionlist, visualize
        """
        self.leadtime = generate_leadtime(0, self.case.leadtime_dist,self.case.leadtime_lb, self.case.leadtime_ub)
        action, penalty = self._check_action_space(action)
        self._initialize_state()
        if visualize: self._visualize("0. IP")
        if self.case_name == "BeerGame" or self.case_name == "General":
            self._generate_demand()
            self._receive_incoming_delivery()
            if visualize: self._visualize("1. Delivery")
            self._receive_incoming_orders()
            if visualize: self._visualize("2. Demand")
            self._place_outgoing_order(1, action)
        elif self.case_name == "Divergent":
            # According to the paper:
            # (1) Warehouse places order to external supplier
            self._place_outgoing_order(0, action)
            if visualize: self._visualize("1. Warehouse order")
            # (2) Warehouse ships the orders to retailers taking the inventory position into account
            self._recieve_incoming_orders_divergent()
            if visualize: self._visualize("2. Warehouse ships")
            # (3) Warehouse and retailers receive their orders
            self._receive_incoming_delivery()
            if visualize: self._visualize("3. Orders received")
            # (4) Demand from customers is observed
            self._generate_demand()
            self._recieve_incoming_orders_customers()
            if visualize: self._visualize("4. Demand")
        else:
            raise NotImplementedError
        CIP = self._code_state()
        holding_costs, backorder_costs = self.calculate_reward()
        reward = holding_costs + backorder_costs + penalty
        return CIP, -reward/self.case.divide, False

    def reset(self):
         None
