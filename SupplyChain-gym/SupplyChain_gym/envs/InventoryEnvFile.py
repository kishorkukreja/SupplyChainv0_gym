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
        self.t=0
        #self.n=0
        self.seed_int = 0
        self.num_of_periods=self.case.num_of_periods
        # set random generation seed (unless using user demands)
        self.seed(self.seed_int)
        if self.method == 'DRL':
            self.action_low = action_low
            self.action_high = action_high
            self.action_min = action_min
            self.action_max = action_max
            self.state_low = state_low
            self.state_high = state_high
        
        self.determine_potential_actions()
        self.determine_potential_states()
        
    def seed(self,seed=None):
        '''
        Set random number generation seed
        '''
        # seed random state
        if seed != None:
            np.random.seed(seed=int(seed))
        
    def generate_leadtime(self,t, dist, lowerbound, upperbound):
        """
        Generate the leadtime of the dataset from paper or distribution.

        Returns: Integer
        """
        if dist == 'uniform':
            leadtime = random.randrange(lowerbound, upperbound + 1)
        else:
            raise Exception
        return leadtime

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
        t=self.period
        source, destination = np.nonzero(self.case.connections)
        for retailer, customer in zip(source[-self.case.no_customers:],destination[-self.case.no_customers:]):
            if self.case.demand_dist == 'poisson':
                demand_mean = random.randrange(self.case.demand_lb,self.case.demand_ub + 1)
                demand = np.random.poisson(demand_mean)
            elif self.case.demand_dist == 'uniform':
                demand = random.randrange(self.case.demand_lb,self.case.demand_ub + 1)
            self.O[t, customer, retailer] = demand
            #print('Demand',demand)

    def calculate_reward(self):
        """
        Calculate the reward for the current period.

        Returns: holding costs, backorder costs
        """
        t=self.period
        print('Inventory State:',np.array(self.INV[t]))
        print('Back order State:',np.array(self.BO[t]))
        backorder_costs = np.sum(np.array(self.BO[t]) * np.array(self.case.bo_costs))
        hc=self.case.holding_costs
        holding_costs = np.sum(np.array(self.INV[t]) * np.array(hc))
        return holding_costs, backorder_costs

    def _initialize_state(self):
        """
        Initialize the inventory position for every node.
        
        Copies the inventory position from the previous timestep.
        """
        ##Inventory
        ##Assuming unlimited inventory for source i.e. raw materials nodes 
        inv=100000000.0
        t=self.period
        if t==0:
            for i,j in enumerate(range(self.case.no_suppliers,self.case.no_nodes-self.case.no_customers)):##4,5,6,7,8,9,10,11,12
                #self.INV[0,i]=inv
                self.INV[0,j]=self.case.state_high[2+i:2+i+1][0]/2 ##set initial inventory as half of overall inventory 
            for i,j in enumerate(range(0,self.case.no_suppliers)):##01,2,3
                self.INV[0,j]=inv    
            ##BackOrders ##Initial Backorder=0
            i_list, j_list = np.nonzero(self.case.connections)
            for i, j in zip(i_list[self.case.no_suppliers:], j_list[self.case.no_suppliers:]):
                    self.BO[0, j, i]=0        
              
            ##Intransit ##Initial Intransit=0  
            i_list, j_list = np.nonzero(self.case.connections)
            for i, j in zip(i_list[:-self.case.no_customers], j_list[:-self.case.no_customers]):
                    self.in_transit[0, i, j]=0
        else:   
            for i,j in enumerate(range(0,self.case.no_suppliers)):##01,2,3
                self.INV[t,j]=inv  
            # initial inventory#self.case.no_stockpoints
        #self.Y.loc[0,:]=np.zeros(PS) # initial pipeline inventory
        #self.action_log = np.zeros([T, PS])

        # set state
        self._update_state()
        
        #return None
    
    def _update_state(self):
        # State is a concatenation of demand, inventory, and pipeline at each time step
        #demand = np.hstack([self.D[d].iloc[self.period] for d in self.retail_links])
        t=self.period
        if t == 0:
        
            ##inventory
            inventory = np.hstack([self.INV[0,j] for j in range(self.case.no_suppliers,self.case.no_nodes-self.case.no_customers)])
            tot_inventory=np.array(np.sum(inventory))
            #np.hstack([self.X[n].iloc[self.period] for n in self.main_nodes])
            
            ##Backorders
            # All Backorders     
            bo=[]
            i_list, j_list = np.nonzero(self.case.connections)
            for i, j in zip(i_list[self.case.no_suppliers:], j_list[self.case.no_suppliers:]):
                bo.append(self.BO[0, j, i])
            bo=np.hstack(bo)
            tot_bo=np.array(np.sum(bo))
            
            
            ##Backorders
            # All Backorders     
            it=[]
            i_list, j_list = np.nonzero(self.case.connections)
            for i, j in zip(i_list[:-self.case.no_customers], j_list[:-self.case.no_customers]):
                it.append(self.in_transit[0, i, j])
            it=np.hstack(it)
        
        else:
            ##inventory
            inventory = np.hstack([self.INV[t-1,j] for j in range(self.case.no_suppliers,self.case.no_nodes-self.case.no_customers)])
            
            tot_inventory=np.array(np.sum(inventory))
            #np.hstack([self.X[n].iloc[self.period] for n in self.main_nodes])
            
            ##Backorders
            # All Backorders     
            bo=[]
            i_list, j_list = np.nonzero(self.case.connections)
            for i, j in zip(i_list[self.case.no_suppliers:], j_list[self.case.no_suppliers:]):
                bo.append(self.BO[t-1, j, i])
            bo=np.hstack(bo)
            tot_bo=np.array(np.sum(bo))
            
            
            ##Intransit
            # All Intransit     
            it=[]
            i_list, j_list = np.nonzero(self.case.connections)
            for i, j in zip(i_list[:-self.case.no_customers], j_list[:-self.case.no_customers]):
                it.append(self.in_transit[t-1, i, j])
            it=np.hstack(it)
            
            
            ## for next time period copy the value of inventory from previous time period 
            for i,j in enumerate(range(self.case.no_suppliers,self.case.no_nodes-self.case.no_customers)):##4,5,6,7,8,9,10,11,12
                #self.INV[0,i]=inv
                self.INV[t,j]=self.INV[t-1,j]
                
                
            ## for next time period copy the value of Back orders from previous time period     
            #i_list, j_list = np.nonzero(self.case.connections)
            #for i, j in zip(i_list[self.case.no_suppliers:], j_list[self.case.no_suppliers:]):
            #    self.BO[t, j, i]=self.BO[t-1, j, i]
            
            ## for next time period copy the value of Intransit from previous time period     
            #i_list, j_list = np.nonzero(self.case.connections)
            #for i, j in zip(i_list[:-self.case.no_customers], j_list[:-self.case.no_customers]):
            #    self.in_transit[t, i, j]=self.in_transit[t-1, i, j]
            
        
        
        # Pipeline values won't be of proper dimension if current
        # current period < lead time. We need to add 0's as padding.
        # if self.period == 0:
        #     _pipeline = [[self.Y[k].iloc[0]]
        #         for k, v in self.lead_times.items()]
        # else:
        #     _pipeline = [self.Y[k].iloc[max(self.period-v,0):self.period].values
        #         for k, v in self.lead_times.items()]
        # pipeline = []
        # for p, v in zip(_pipeline, self.lead_times.values()):
        #     if v == 0:
        #         continue
        #     if len(p) <= v:
        #         pipe = np.zeros(v)
        #         pipe[-len(p):] += p
        #     pipeline.append(pipe)
        # pipeline = np.hstack(pipeline)
        # 
        
        self.state = np.hstack([tot_inventory,tot_bo, inventory, bo,it]) ## 1,1,9,19,18 dimensions

    def _receive_incoming_delivery(self):
        t=self.period
        # # Loop over all suppliers and stockpoints
        # for i in range(0, self.case.no_stockpoints + self.case.no_suppliers):
        # # Loop over all stockpoints
        # # Note that only forward delivery is possible, hence 'i+1'
            # for j in range(i + 1, self.case.no_stockpoints +self.case.no_suppliers):
                   # delivery = self.T[0, i, j]
                   # self.INV[0, j] += delivery
                   # self.in_transit[0, i, j] -= delivery
                   # self.T[0, i, j] = 0
        # Loop over all suppliers and stockpoints
        for i in range(0, self.case.no_stockpoints + self.case.no_suppliers):
        # Loop over all stockpoints
        # Note that only forward delivery is possible, hence 'i+1'
            for j in range(i + 1, self.case.no_stockpoints +self.case.no_suppliers):
                   #delivery = self.T[t, i, j] ## all deliveries for current time step 
                   delivery = self.T[t, j, i] ## all deliveries for current time step
                   #print(f'Receiving into {j} from {i} of quantity {delivery}')
                   self.INV[t, j] += delivery
                   self.in_transit[t, i, j] -= delivery
                   #self.T[t, i, j] = 0
                   self.T[t, j, i] = 0

    def _receive_incoming_orders(self):
    # # Loop over every stockpoint
        # for i in range(self.case.no_stockpoints + self.case.no_suppliers):
            # # Check if the inventory is larger than all incoming orders
            # if self.INV[0, i] >= np.sum(self.O[0, :, i], 0):
                # for j in np.nonzero(self.case.connections[i])[0]:
                    # if self.O[0, j, i] > 0:
                        # self._fulfill_order(i, j, self.O[0, j, i])
                            # if self.t >= self.case.warmup:
                                # self.TotalFulfilled[j,i] += self.O[0,j,i]
            # else:
                # IPlist = {}
                # # Generate a list of stockpoints that have outstanding orders
                # k_list = np.nonzero(self.O[0, :, i])[0]
                # bo_echelon = np.sum(self.BO[0], 0)
                # for k in k_list:
                # IPlist[k] = self.INV[0, k] - bo_echelon[k]
                # # Check the lowest inventory position and sort these on lowest IP
                # sorted_IP = {k: v for k, v in sorted(IPlist.items(), key=lambda item: item[1])}
                # for j in sorted_IP:
                    # inventory = self.INV[0, i]
                    # # Check if the remaining order can be fulfilled completely
                    # if inventory >= self.O[0, j, i]:
                        # self._fulfill_order(i, j, self.O[0, j, i])
                            # if self.t >= self.case.warmup:
                                # self.TotalFulfilled[j,i] += self.O[0,j,i]
                    # else:
                        # # Else, fulfill how far possible
                        # quantity = self.O[0, j, i] - inventory
                        # self._fulfill_order(i, j, inventory)
                        # if self.t >= self.case.warmup:
                            # self.TotalFulfilled[j,i] += inventory
                        # if self.case.unsatisfied_demand == 'backorders':
                            # self.BO[0, j, i] += quantity
                            # if self.t >= self.case.warmup:
                                # self.TotalBO[j,i] += quantity
        # if self.case.unsatisfied_demand == 'backorders':
            # i_list, j_list = np.nonzero(self.case.connections)
            # for i, j in zip(i_list, j_list):
                # inventory = self.INV[0, i]
                # # If there are any backorders, fulfill them afterwards
                # if inventory > 0:
                # # If the inventory is larger than the backorder
                # # Fulfill the whole backorder
                # backorder = self.BO[0, j, i]
                # if inventory >= backorder:
                    # if self.fix:
                        # self._fulfill_order(i, j, backorder)
                    # else:
                        # self.INV[0, i] -= backorder
                    # self.BO[0, j, i] = 0
                # # Else, fulfill the entire inventory
                # else:
                    # self._fulfill_order(i, j, inventory)
                    # self.BO[0, j, i] -= inventory
                    
        t=self.period         
        self.t=self.period         
        
        # Loop over every stockpoint
        for i in range(self.case.no_stockpoints + self.case.no_suppliers):
            # Check if the inventory is larger than all incoming orders
            print(f'Node-{i}')
            print(f'Inventory Needed-{np.sum(self.O[t, :, i], 0)}')
            print(f'Inventory Available-{np.sum(self.INV[t, i])}')
            if self.INV[t, i] >= np.sum(self.O[t, :, i], 0): ##big inventory
                for j in np.nonzero(self.case.connections[i])[0]:
                    if self.O[t, j, i] > 0:
                        self._fulfill_order(i, j, self.O[t, j, i])
                        print(f'Outer Loop - Inventory from {i} to {j} of quantity {self.O[t, j, i]}')
                        if self.t >= self.case.warmup:
                            self.TotalFulfilled[j,i] += self.O[t,j,i]
            else:
                IPlist = {}
                # Generate a list of stockpoints that have outstanding orders
                k_list = np.nonzero(self.O[t, :, i])[0]
                bo_echelon = np.sum(self.BO[t], 0)
                for k in k_list:
                    IPlist[k] = self.INV[t, k] - bo_echelon[k]
                    # Check the lowest inventory position and sort these on lowest IP
                    sorted_IP = {k: v for k, v in sorted(IPlist.items(), key=lambda item: item[1])}
                    print(f'Sorted IP -{sorted_IP}')
                    for j in sorted_IP:
                        inventory = self.INV[t, i]
                        # Check if the remaining order can be fulfilled completely
                        if inventory >= self.O[t, j, i]:
                            self._fulfill_order(i, j, self.O[t, j, i])
                            print(f'Inventory from {i} to {j} of quantity {self.O[t, j, i]}')
                            if self.t >= self.case.warmup:
                                self.TotalFulfilled[j,i] += self.O[t,j,i]
                        else:
                            # Else, fulfill how far possible
                            quantity = self.O[t, j, i] - inventory
                            self._fulfill_order(i, j, inventory)
                            print(f'After 1st else Inventory from {i} to {j} of quantity {inventory}')
                            if self.t >= self.case.warmup:
                                self.TotalFulfilled[j,i] += inventory
                            if self.case.unsatisfied_demand == 'backorders':
                                self.BO[t, j, i] += quantity
                                print(f'After 1st else Backorder from {j} to {i} of quantity {quantity}')
                                if self.t >= self.case.warmup:
                                    self.TotalBO[j,i] += quantity
                                    
                                    
        if self.case.unsatisfied_demand == 'backorders':
            i_list, j_list = np.nonzero(self.case.connections)
            for i, j in zip(i_list, j_list):
                inventory = self.INV[t, i]
                # If there are any backorders, fulfill them afterwards
                if inventory > 0:
                # If the inventory is larger than the backorder
                # Fulfill the whole backorder
                    backorder = self.BO[t, j, i]
                    if inventory >= backorder:
                        if self.fix:
                            self._fulfill_order(i, j, backorder)
                            print(f'Outer if Inventory from {i} to {j} of quantity {backorder}')
                        else:
                            self.INV[t, i] -= backorder
                        self.BO[t, j, i] = 0
                        
                    # Else, fulfill the entire inventory
                    else:
                        self._fulfill_order(i, j, inventory)
                        print(f'Outer if else Inventory from {i} to {j} of quantity {inventory}')
                        self.BO[t, j, i] -= inventory
                        #print(f'Backorder from {j} to {i} of quantity {quantity}')
        
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
        t=self.period
        
        if destination >= self.case.no_nodes - self.case.no_customers:
            leadtime = 0
        else:
            leadtime = self.leadtime
        # The order is fulfilled immediately for the customer
        # or whenever the leadtime is 0
        
        if leadtime == 0:
        # The new inventorylevel is increased with the shipped quantity
            self.INV[t, destination] += quantity
            print(f'Inventory Increased of Node {destination} by quantity {quantity}')
        else:
            # If the order is not fulfilled immediately, denote the time when
            # the order will be delivered. This can not be larger than the horizon
            if leadtime < self.n:
                #self.T[t+leadtime, source, destination] += quantity
                self.T[t+leadtime, destination,source] += quantity
                self.in_transit[t+leadtime, source, destination] += quantity
                print(f'Inventory wiil be increased on period {t+leadtime} of Node {destination} by quantity {quantity}')
            else:
                raise NotImplementedError
                for k in range(0, min(leadtime, self.n) + 1):
                    self.in_transit[t+k, source, destination] += quantity
        # Suppliers have unlimited capacity
        if source >= self.case.no_suppliers:
            self.INV[t, source] -= quantity
            print(f'Inventory Decreased of Node {source} by quantity {quantity}')
            
    def _place_outgoing_order(self, t, action):
        k = 0
        
        incomingOrders = np.sum(self.O[t-1], 0)
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
        
        self.t=self.period
        
        if self.case.order_policy == 'X':
            self.O[t, j, i] += action[k]
            print(f'Placing order from {j} to {i} of quantity {action[k]}')
            if (self.t < self.case.horizon - 1) and (self.t >= self.case.warmup-1):
                self.TotalDemand[j,i] += action[k]
        
        elif self.case.order_policy == 'X+Y':
            self.O[t, j, i] += incomingOrders[j] + action[k]
            print(f'Placing order from {j} to {i} of quantity {incomingOrders[j] + action[k]}')
            if (self.t < self.case.horizon - 1) and (self.t >= self.case.warmup-1):
                self.TotalDemand[j,i] += incomingOrders[j] + action[k]
        
        elif self.case.order_policy == 'BaseStock':
            bo_echelon = np.sum(self.BO[0], 0)
            self.O[t, j, i] += max(0, action[k]-(self.INV[0,j]+self.in_transit[0,i,j]-bo_echelon[j]))
            print(f'Placing order from {j} to {i} of quantity {max(0, action[k]-(self.INV[0,j]+self.in_transit[0,i,j]-bo_echelon[j]))}')
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
            #for i in range(len(action_clip)):
            #    action_clip[i] = ((action_clip[i]-low[i])/(high[i]-low[i]))*((max[i]-min[i]))+min[i]
            action = [np.round(num) for num in action_clip]
        return action,0.0
    
    def step(self, action, visualize=False):
        """
        Execute one step in the RL method.
        
        input: actionlist, visualize
        """
        self.leadtime = self.generate_leadtime(0, self.case.leadtime_dist,self.case.leadtime_lb, self.case.leadtime_ub)
        print('-----------------------------------Period :',self.period)
        action, penalty = self._check_action_space(action)
        
        self._initialize_state()
        print('Action :',action)
        print('State :',self.state)
        if visualize: self._visualize("0. IP")
        
        

        if self.case_name == "General":
            self._generate_demand() ## order from customer to retail i.e. last leg
            self._receive_incoming_delivery()
            if visualize: self._visualize("1. Delivery")
            self._receive_incoming_orders()
            if visualize: self._visualize("2. Demand")
            #self._place_outgoing_order(1, action)
            self._place_outgoing_order(self.period+1, action)
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
            
        #CIP = self._code_state()
        holding_costs, backorder_costs = self.calculate_reward()
        reward = holding_costs + backorder_costs + penalty
            
        # update period
        self.period += 1
        
        # determine if simulation should terminate
        if self.period >= self.num_of_periods:
            done = True
        else:
            done = False
            # update stae
            #self._update_state()
        # CIP is next state
        
        
        print('Holding Costs :',holding_costs)
        print('Back Order Costs :',backorder_costs)
        print('Reward :',reward)
        
        
        return self.state, -reward/self.case.divide, done,{}

    def reset(self):
    
        # initializetion
        self.period = 0 # initialize time
    
        self.INV = [[0 for _ in range(self.case.no_nodes)] for _ in range(self.num_of_periods+10)] #Inventory at each stock point
        
        self.BO = [[[0 for _ in range(self.case.no_nodes)] for _ in range(self.case.no_nodes)] for _ in range(self.num_of_periods+10)] #BackOrder for every i,j connection
        
        self.in_transit = [[[0 for _ in range(self.case.no_nodes)] for _ in range(self.case.no_nodes)] for _ in range(self.num_of_periods+10)] #Intrasnit for every i,j connection
        
        self.T = [[[0 for _ in range(self.case.no_nodes)] for _ in range(self.case.no_nodes)] for _ in range(self.num_of_periods+10)] #Delivery for every i,j connection
        
        self.O = [[[0 for _ in range(self.case.no_nodes)] for _ in range(self.case.no_nodes)] for _ in range(self.num_of_periods+10)] #Order for every i,j connection
        
        self.TotalFulfilled = [[0 for _ in range(self.case.no_nodes)] for _ in range(self.case.no_nodes)] #Total Fulfilled demand for every i,j connection
        
        self.TotalDemand = [[0 for _ in range(self.case.no_nodes)] for _ in range(self.case.no_nodes)] #Total Demand for every i,j connection
        
        self.TotalBO = [[0 for _ in range(self.case.no_nodes)] for _ in range(self.case.no_nodes)] #Total Backorders for every i,j connection
        
        self.INV=np.array(self.INV)
        self.BO=np.array(self.BO)
        self.in_transit=np.array(self.in_transit)
        self.T=np.array(self.T)
        self.O=np.array(self.O)
        self.TotalFulfilled=np.array(self.TotalFulfilled)
        self.TotalDemand=np.array(self.TotalDemand)
        self.TotalBO=np.array(self.TotalBO)
        return self.state_low
