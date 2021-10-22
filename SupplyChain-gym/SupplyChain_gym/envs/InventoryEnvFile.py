import gym
from gym.utils import seeding
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gym import spaces
from SupplyChain_gym.utils import assign_env_config


class InventoryEnv(gym.Env):
    """
    General Inventory Control Environment.

    Currently tested with:
    - A reinforcement learning model for supply chain ordering management:
    An application to the beer game - Chaharsooghi (2002)
    """
    def __init__(self,*args,**kwargs):
    #case, action_low, action_high, action_min, action_max,
    #   state_low, state_high, method,
    #    coded=False, fix=True, ipfix=True
        
        # assign_env_config(self, kwargs)
        # self = case
        # self.case_name = case_name
        # self.stockpoints_echelon = stockpoints_echelon
        # self.no_suppliers = no_suppliers
        # self.no_customers = no_customers
        # self.no_stockpoints = no_stockpoints
        # self.no_suppliers = no_suppliers
        # self.no_nodes = no_nodes
        # self.no_echelons = no_echelons
        # self.connections = connections
        # self.unsatisfied_demand = unsatisfied_demand
        # self.holding_costs = holding_costs
        # self.bo_costs = bo_costs
        # self.demand_dist = demand_dist
        # self.demand_lb = demand_lb
        # self.demand_ub = demand_ub
        # self.leadtime_dist = leadtime_dist
        # self.leadtime_lb = leadtime_lb
        # self.leadtime_ub = leadtime_ub
        # self.num_of_periods = num_of_periods
        # self.cost_price = cost_price
        # self.selling_price = selling_price
        # self.order_policy = order_policy
        # self.horizon = horizon
        # self.warmup = warmup
        # self.divide = divide
        # self.n =self.leadtime_ub + 1
        # self.coded = coded
        # self.fix = fix
        # self.ipfix = ipfix
        # self.method = method    
        # self.t=0
        # #self.n=0
        # self.seed_int = 0
        # #self.num_of_periods=num_of_periods
        # # set random generation seed (unless using user demands)
        # self.seed(self.seed_int)
        
        
        # self.num=0
        # if self.method == 'DRL':
            # self.action_low = action_low
            # self.action_high = action_high
            # self.action_min = action_min
            # self.action_max = action_max
            # self.state_low = state_low
            # self.state_high = state_high
            
        self.case_name='General'
        self.stockpoints_echelon = [4, 4, 5, 5]
        # Number of suppliers
        self.no_suppliers = self.stockpoints_echelon[0]
        # Number of customers
        self.no_customers = self.stockpoints_echelon[-1]
        # Number of stockpoints
        self.no_stockpoints = sum(self.stockpoints_echelon) - \
        self.no_suppliers - self.no_customers

        # Total number of nodes
        self.no_nodes = sum(self.stockpoints_echelon)
        # Total number of echelons, including supplier and customer
        self.no_echelons = len(self.stockpoints_echelon)
        # Connections between every stockpoint
        self.connections = np.array([
        #0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 0
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 1
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 2
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 3
        [0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0.5, 0.15, 0, 0, 0, 0, 0, 0, 0], # 4
        [0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.4, 0.80, 0.1, 0, 0, 0, 0, 0, 0], # 5
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0.7, 0, 0, 0, 0, 0], # 6
        [0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.05, 0.1, 0.3, 0, 0, 0, 0, 0], # 7
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # 9
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # 10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # 11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 12
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 13
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 14
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 15
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 16
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # 17
        ])
        # Determines what happens with unsatisfied demand, can be either 'backorders' or 'lost_sales'
        self.unsatisfied_demand = 'backorders'
        # Holding costs per stockpoint # for both warehouse and plants
        self.holding_costs = [0, 0, 0, 0, 7, 7, 7, 7, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0]
        self.initial_inventory = [1000000, 0, 0, 0, 0, 0, 0, 0]
        # Backorder costs per stockpoint #only for WHs here
        self.bo_costs = [0, 0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 13, 0, 0, 0, 0, 0]
        self.lo_costs = [0, 0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 13, 0, 0, 0, 0, 0]
        # Demand distribution, can be either 'poisson' or 'uniform'
        self.demand_dist = 'poisson'
        # Lower bound of the demand distribution
        self.demand_lb = 100
        # Upper bound of the demand distribution
        self.demand_ub = 500
        # Leadtime distribution, can only be 'uniform'
        self.leadtime_dist = 'uniform'
        # Lower bound of the leadtime distribution
        self.leadtime_lb = 1
        # Upper bound of the leadtime distribution
        self.leadtime_ub = 1
        ##periods
        self.num_of_periods=5
        ##price
        self.cost_price=[5,4,4,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.selling_price=[0,0,0,0,0,0,0,0,15,16,15,16,16,0,0,0,0,0]
        # Predetermined order policy, can be either 'X' or 'X+Y' or 'BaseStock'
        self.order_policy = 'X'
        self.horizon = 75
        self.warmup = 50
        self.divide = 1000
        self.coded = False
        self.fix = True
        self.ipfix = True
        self.method = 'DRL'
        self.n = 10 #self.leadtime_ub + 1,
        self.leadtime=1
        
        self.action_low = np.array([-5,-5,-5,-5,-5,-5,-5,-5,-5]) #9
        self.action_high = np.array([5,5,5,5,5,5,5,5,5])         #9
        self.action_min = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])  #9 #first 4 from plants last 5 from WHs orderQty
        self.action_max = np.array([300, 300, 300, 300, 75, 75, 75, 75, 75]) #9 #first 4 from plants last 5 from WHs orderQty
        self.state_low = np.zeros(48)
        self.state_high = np.array([13500, 28500, # Total inventory and backorders
        1500,1500,1500,1500,1500,1500,1500,1500,1500, # Inventory per stockpoint
        #11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, #per connection backorder
        1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500, #=7750
        #30, 31, 32, 33,34,35,36,37,38,39,40,41,42,43,44,45,46,47
        1500,1500,1500,1500,750,750,750,750,750,750,750,750,750,750,750,750,750,750]) # Per connection Intransit Qty
        assign_env_config(self, kwargs)
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
        source, destination = np.nonzero(self.connections)
        for retailer, customer in zip(source[-self.no_customers:],destination[-self.no_customers:]):
            if self.demand_dist == 'poisson':
                demand_mean = random.randrange(self.demand_lb,self.demand_ub + 1)
                demand = np.random.poisson(demand_mean)
            elif self.demand_dist == 'uniform':
                demand = random.randrange(self.demand_lb,self.demand_ub + 1)
            self.O[t, customer, retailer] = demand
            #print('Demand',demand)

    def calculate_reward(self):
        """
        Calculate the reward for the current period.

        Returns: holding costs, backorder costs
        """
        t=self.period
        
        
        
        if self.unsatisfied_demand == 'backorders':
            #print('Back order State:',np.array(self.BO[t]))
            backorder_costs = np.sum(np.array(self.BO[t]) * np.array(self.bo_costs))
            hc=self.holding_costs
            holding_costs = np.sum(np.array(self.INV[t]) * np.array(hc))
            revenue=np.sum(np.array(self.TotalSales[t]) * np.array(self.selling_price))
            cost_of_goods=np.sum(np.array(self.TotalCOGS[t]) * np.array(self.cost_price))
            #self.cost_price=self.cost_price
            #self.selling_price=self.selling_price
            
            ## Penalty applying
            if t>0:
                if np.sum(np.array(self.BO[t]))>np.sum(np.array(self.BO[t-1])):
                    backorder_costs=backorder_costs+(t+1/t)
                if np.sum(np.array(self.INV[t]))>np.sum(np.array(self.INV[t-1])):
                    holding_costs=holding_costs+(t+1/t)
            else:
                backorder_costs=backorder_costs
                holding_costs=holding_costs
            lost_sales_costs=0
                
        elif self.unsatisfied_demand != 'backorders':
            #print('Back order State:',np.array(self.BO[t]))
            lost_sales_costs = np.sum(np.array(self.LO[t]) * np.array(self.lo_costs))
            hc=self.holding_costs
            holding_costs = np.sum(np.array(self.INV[t]) * np.array(hc))
            revenue=np.sum(np.array(self.TotalSales[t]) * np.array(self.selling_price))
            cost_of_goods=np.sum(np.array(self.TotalCOGS[t]) * np.array(self.cost_price))
            #self.cost_price=self.cost_price
            #self.selling_price=self.selling_price
            
            ## Penalty applying
            if t>0:
                if np.sum(np.array(self.LO[t]))>np.sum(np.array(self.LO[t-1])):
                    lost_sales_costs=lost_sales_costs+(t+1/t)
                if np.sum(np.array(self.INV[t]))>np.sum(np.array(self.INV[t-1])):
                    holding_costs=holding_costs+(t+1/t)
            else:
                lost_sales_costs=lost_sales_costs
                holding_costs=holding_costs
            backorder_costs=0
        
        return holding_costs, backorder_costs,lost_sales_costs,revenue,cost_of_goods

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
            for i,j in enumerate(range(self.no_suppliers,self.no_nodes-self.no_customers)):##4,5,6,7,8,9,10,11,12
                #self.INV[0,i]=inv
                self.INV[0,j]=self.state_high[2+i:2+i+1][0]/2 ##set initial inventory as half of overall inventory 
            for i,j in enumerate(range(0,self.no_suppliers)):##01,2,3
                self.INV[0,j]=inv    
            ##BackOrders ##Initial Backorder=0
            i_list, j_list = np.nonzero(self.connections)
            for i, j in zip(i_list[self.no_suppliers:], j_list[self.no_suppliers:]):
                    self.BO[0, j, i]=0        
              
            ##Intransit ##Initial Intransit=0  
            i_list, j_list = np.nonzero(self.connections)
            for i, j in zip(i_list[:-self.no_customers], j_list[:-self.no_customers]):
                    self.in_transit[0, i, j]=0
        else:   
            for i,j in enumerate(range(0,self.no_suppliers)):##01,2,3
                self.INV[t,j]=inv  
            # initial inventory#self.no_stockpoints
        #self.Y.loc[0,:]=np.zeros(PS) # initial pipeline inventory
        #self.action_log = np.zeros([T, PS])

        # set state
        self._update_state()
        
        #return None
    
    def _update_state(self):
        # State is a concatenation of demand, inventory, and pipeline at each time step
        #demand = np.hstack([self.D[d].iloc[self.period] for d in self.retail_links])
        t=self.period
        if self.case_name=='General' or self.case_name=='Linear':
            if t == 0:
            
                ##inventory
                inventory = np.hstack([self.INV[0,j] for j in range(self.no_suppliers,self.no_nodes-self.no_customers)])
                tot_inventory=np.array(np.sum(inventory))
                #np.hstack([self.X[n].iloc[self.period] for n in self.main_nodes])
                
                ##Backorders
                # All Backorders     
                bo=[]
                i_list, j_list = np.nonzero(self.connections)
                for i, j in zip(i_list[self.no_suppliers:], j_list[self.no_suppliers:]):
                    bo.append(self.BO[0, j, i])
                bo=np.hstack(bo)
                tot_bo=np.array(np.sum(bo))
                
                
                ##Backorders
                # All Backorders     
                it=[]
                i_list, j_list = np.nonzero(self.connections)
                for i, j in zip(i_list[:-self.no_customers], j_list[:-self.no_customers]):
                    it.append(self.in_transit[0, i, j])
                it=np.hstack(it)
            
            else:
                ##inventory
                inventory = np.hstack([self.INV[t-1,j] for j in range(self.no_suppliers,self.no_nodes-self.no_customers)])
                
                tot_inventory=np.array(np.sum(inventory))
                #np.hstack([self.X[n].iloc[self.period] for n in self.main_nodes])
                
                ##Backorders
                # All Backorders     
                bo=[]
                i_list, j_list = np.nonzero(self.connections)
                for i, j in zip(i_list[self.no_suppliers:], j_list[self.no_suppliers:]):
                    bo.append(self.BO[t-1, j, i])
                bo=np.hstack(bo)
                tot_bo=np.array(np.sum(bo))
                
                
                ##Intransit
                # All Intransit     
                it=[]
                i_list, j_list = np.nonzero(self.connections)
                for i, j in zip(i_list[:-self.no_customers], j_list[:-self.no_customers]):
                    it.append(self.in_transit[t, i, j])
                it=np.hstack(it)
                
                
                ## for next time period copy the value of inventory from previous time period 
                for i,j in enumerate(range(self.no_suppliers,self.no_nodes-self.no_customers)):##4,5,6,7,8,9,10,11,12
                    #self.INV[0,i]=inv
                    self.INV[t,j]=self.INV[t-1,j]
                    
                    
                ## for next time period copy the value of Back orders from previous time period     
                i_list, j_list = np.nonzero(self.connections)
                for i, j in zip(i_list[self.no_suppliers:], j_list[self.no_suppliers:]):
                    self.BO[t, j, i]=self.BO[t-1, j, i]
                
                ## for next time period copy the value of Intransit from previous time period     
                #i_list, j_list = np.nonzero(self.connections)
                #for i, j in zip(i_list[:-self.no_customers], j_list[:-self.no_customers]):
                #    self.in_transit[t, i, j]=self.in_transit[t-1, i, j]
                
        elif self.case_name=='Divergent':
            if t == 0:
            
                ##inventory
                inventory = np.hstack([self.INV[0,j] for j in range(self.no_suppliers,self.no_nodes-self.no_customers)])
                tot_inventory=np.array(np.sum(inventory))
                #np.hstack([self.X[n].iloc[self.period] for n in self.main_nodes])
                
                ##Backorders
                # All Backorders     
                bo=[]
                i_list, j_list = np.nonzero(self.connections)
                for i, j in zip(i_list[self.no_suppliers:], j_list[self.no_suppliers:]):
                    bo.append(self.BO[0, j, i])
                bo=np.hstack(bo)
                tot_bo=np.array(np.sum(bo))
                
                
                ##Backorders
                # All Backorders     
                it=[]
                i_list, j_list = np.nonzero(self.connections)
                for i, j in zip(i_list[:-self.no_customers], j_list[:-self.no_customers]):
                    it.append(self.in_transit[0, i, j])
                it=np.hstack(it)
            
            else:
                ##inventory
                inventory = np.hstack([self.INV[t-1,j] for j in range(self.no_suppliers,self.no_nodes-self.no_customers)])
                
                tot_inventory=np.array(np.sum(inventory))
                #np.hstack([self.X[n].iloc[self.period] for n in self.main_nodes])
                
                ##Backorders
                # All Backorders     
                bo=[]
                i_list, j_list = np.nonzero(self.connections)
                for i, j in zip(i_list[self.no_suppliers:], j_list[self.no_suppliers:]):
                    bo.append(self.BO[t-1, j, i])
                bo=np.hstack(bo)
                tot_bo=np.array(np.sum(bo))
                
                
                ##Intransit
                # All Intransit     
                it=[]
                i_list, j_list = np.nonzero(self.connections)
                for i, j in zip(i_list[:-self.no_customers], j_list[:-self.no_customers]):
                    it.append(self.in_transit[t, i, j])
                it=np.hstack(it)
                
                
                ## for next time period copy the value of inventory from previous time period 
                for i,j in enumerate(range(self.no_suppliers,self.no_nodes-self.no_customers)):##4,5,6,7,8,9,10,11,12
                    #self.INV[0,i]=inv
                    self.INV[t,j]=self.INV[t-1,j]
                    
                    
                ## for next time period copy the value of Back orders from previous time period     
                i_list, j_list = np.nonzero(self.connections)
                for i, j in zip(i_list[self.no_suppliers:], j_list[self.no_suppliers:]):
                    self.BO[t, j, i]=self.BO[t-1, j, i]
                
                ## for next time period copy the value of Intransit from previous time period     
                #i_list, j_list = np.nonzero(self.connections)
                #for i, j in zip(i_list[:-self.no_customers], j_list[:-self.no_customers]):
                #    self.in_transit[t, i, j]=self.in_transit[t-1, i, j]

            
        
        
        
        self.state = np.hstack([tot_inventory,tot_bo, inventory, bo,it]) ## 1,1,9,19,18 dimensions

    def _receive_incoming_delivery(self):
        t=self.period
        # # Loop over all suppliers and stockpoints
        # for i in range(0, self.no_stockpoints + self.no_suppliers):
        # # Loop over all stockpoints
        # # Note that only forward delivery is possible, hence 'i+1'
            # for j in range(i + 1, self.no_stockpoints +self.no_suppliers):
                   # delivery = self.T[0, i, j]
                   # self.INV[0, j] += delivery
                   # self.in_transit[0, i, j] -= delivery
                   # self.T[0, i, j] = 0
        # Loop over all suppliers and stockpoints
        for i in range(0, self.no_stockpoints + self.no_suppliers):
        # Loop over all stockpoints
        # Note that only forward delivery is possible, hence 'i+1'
            for j in range(i + 1, self.no_stockpoints +self.no_suppliers):
                   #delivery = self.T[t, i, j] ## all deliveries for current time step 
                   delivery = self.T[t, j, i] ## all deliveries for current time step
                   print(f'Receiving into {j} from {i} of quantity {delivery}')
                   self.INV[t, j] += delivery
                   print(f'Increasing Inventory of {j} of quantity {delivery}')
                   self.in_transit[t, i, j] -= delivery
                   #self.T[t, i, j] = 0
                   self.T[t, j, i] = 0

    def _receive_incoming_orders(self):
    # # Loop over every stockpoint
        # for i in range(self.no_stockpoints + self.no_suppliers):
            # # Check if the inventory is larger than all incoming orders
            # if self.INV[0, i] >= np.sum(self.O[0, :, i], 0):
                # for j in np.nonzero(self.connections[i])[0]:
                    # if self.O[0, j, i] > 0:
                        # self._fulfill_order(i, j, self.O[0, j, i])
                            # if self.t >= self.warmup:
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
                            # if self.t >= self.warmup:
                                # self.TotalFulfilled[j,i] += self.O[0,j,i]
                    # else:
                        # # Else, fulfill how far possible
                        # quantity = self.O[0, j, i] - inventory
                        # self._fulfill_order(i, j, inventory)
                        # if self.t >= self.warmup:
                            # self.TotalFulfilled[j,i] += inventory
                        # if self.unsatisfied_demand == 'backorders':
                            # self.BO[0, j, i] += quantity
                            # if self.t >= self.warmup:
                                # self.TotalBO[j,i] += quantity
        # if self.unsatisfied_demand == 'backorders':
            # i_list, j_list = np.nonzero(self.connections)
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
        tmp=np.array(self.stockpoints_echelon)        
        tmp1=np.sum(tmp[:-2]) ## since last layer is retailer and its previous is WH
        #self.num=self.stock ## i greater than 8 
        # Loop over every stockpoint
        for i in range(self.no_stockpoints + self.no_suppliers):
            # Check if the inventory is larger than all incoming orders
            print(f'Node-{i}')
            print(f'Inventory Needed-{np.sum(self.O[t, :, i], 0)}')
            print(f'Inventory Available-{np.sum(self.INV[t, i])}')
            if self.INV[t, i] >= np.sum(self.O[t, :, i], 0): ##big inventory
                for j in np.nonzero(self.connections[i])[0]:
                    if self.O[t, j, i] > 0:
                        self._fulfill_order(i, j, self.O[t, j, i])
                        if i <= self.no_nodes - self.no_customers and i >= tmp1:
                            self.TotalSales[t,i]+=self.O[t, j, i]
                        if i <= self.no_suppliers:
                            self.TotalCOGS[t,i]+=self.O[t, j, i]
                        print(f'Outer Loop - Inventory from {i} to {j} of quantity {self.O[t, j, i]}')
                        if self.t >= self.warmup:
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
                            if i <= self.no_nodes - self.no_customers and i >= tmp1:
                                self.TotalSales[t,i]+=self.O[t, j, i]
                            if i <= self.no_suppliers:
                                self.TotalCOGS[t,i]+=self.O[t, j, i]
                            print(f'Inventory from {i} to {j} of quantity {self.O[t, j, i]}')
                            if self.t >= self.warmup:
                                self.TotalFulfilled[j,i] += self.O[t,j,i]
                        else:
                            # Else, fulfill how far possible
                            quantity = self.O[t, j, i] - inventory
                            self._fulfill_order(i, j, inventory)
                            if i <= self.no_nodes - self.no_customers and i >= tmp1:
                                self.TotalSales[t,i]+=inventory
                            if i <= self.no_suppliers:
                                self.TotalCOGS[t,i]+=inventory
                            print(f'After 1st else Inventory from {i} to {j} of quantity {inventory}')
                            if self.t >= self.warmup:
                                self.TotalFulfilled[j,i] += inventory
                            if self.unsatisfied_demand == 'backorders':
                                self.BO[t, j, i] += quantity
                                print(f'Backorder from {j} to {i} of quantity {quantity}')
                                #print(f'After 1st else Backorder from {j} to {i} of quantity {quantity}')
                                if self.t >= self.warmup:
                                    self.TotalBO[j,i] += quantity
                            else:
                                self.LO[t, j, i] += quantity
                                print(f'Lost Order from {j} to {i} of quantity {quantity}')
            print('COGS for Node',self.TotalCOGS[t,i])                         
            print('Sales for Node',self.TotalSales[t,i])                      
        if self.unsatisfied_demand == 'backorders':
            i_list, j_list = np.nonzero(self.connections)
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
                            if i <= self.no_nodes - self.no_customers and i >= tmp1:
                                self.TotalSales[t,i]+=backorder
                            if i <= self.no_suppliers:
                                self.TotalCOGS[t,i]+=backorder
                            print(f'Outer if Inventory from {i} to {j} of quantity {backorder}')
                        else:
                            self.INV[t, i] -= backorder
                        self.BO[t, j, i] = 0
                        
                    # Else, fulfill the entire inventory
                    else:
                        self._fulfill_order(i, j, inventory)
                        if i <= self.no_nodes - self.no_customers and i >= tmp1:
                                self.TotalSales[t,i]+=inventory
                        if i <= self.no_suppliers:
                                self.TotalCOGS[t,i]+=inventory
                        print(f'Outer if else Inventory from {i} to {j} of quantity {inventory}')
                        self.BO[t, j, i] -= inventory
                        print(f'Backorder from {j} to {i} of quantity {inventory}')
                #print('COGS for Node',self.TotalCOGS[t,i])                         
                #print('Sales for Node',self.TotalSales[t,i])   
        
    def _recieve_incoming_orders_customers(self):
        t=self.period         
        self.t=self.period  
        tmp=np.array(self.stockpoints_echelon)        
        tmp1=np.sum(tmp[:-2]) ## since last layer is retailer and its previous is WH
        
        i_list, j_list = np.nonzero(self.connections)
        for i, j in zip(i_list[-self.no_customers:], j_list[-self.no_customers:]):
            print(f'Node-{i}')
            print(f'Inventory Needed-{np.sum(self.O[t, j, i], 0)}')
            print(f'Inventory Available-{np.sum(self.INV[t, i])}')
            if self.O[t, j, i] > 0:
            # Check if the current order can be fulfilled
                if self.INV[t, i] >= self.O[t, j, i]:
                    self._fulfill_order(i, j, self.O[t, j, i])
                    print(f'Outermost Loop Inventory from {i} to {j} of quantity {self.O[t, j, i]}')
                    if i <= self.no_nodes - self.no_customers and i >= tmp1:
                            self.TotalSales[t,i]+=self.O[t, j, i]
                    if i <= self.no_suppliers:
                            self.TotalCOGS[t,i]+=self.O[t, j, i]
                    # Else, fulfill as far as possible
                else:
                    inventory = max(self.INV[t, i], 0)
                    quantity = self.O[t, j, i] - inventory
                    self._fulfill_order(i, j, inventory)
                    print(f'Outermost Loop Next Inventory from {i} to {j} of quantity {inventory}')
                    if i <= self.no_nodes - self.no_customers and i >= tmp1:
                            self.TotalSales[t,i]+=inventory
                    if i <= self.no_suppliers:
                            self.TotalCOGS[t,i]+=inventory
                    # Add to backorder if applicable
                    if self.unsatisfied_demand == 'backorders':
                        self.BO[t, j, i] += quantity
                        print(f'Backorder from {j} to {i} of quantity {quantity}')
                    else:
                        self.LO[t, j, i] += quantity
                        print(f'Lost Order from {j} to {i} of quantity {quantity}')
        if self.unsatisfied_demand == 'backorders':
            for i, j in zip(i_list[-self.no_customers:], j_list[-self.no_customers:]):
                inventory = self.INV[t, i]
                print(f'Node-{i}')
                print(f'Inventory Available-{inventory}')
                # If there are any backorders, fulfill them afterwards
                if inventory > 0:
                    # If the inventory is larger than the backorder
                    # Fulfill the whole backorder
                    backorder = self.BO[t, j, i]
                    if inventory >= backorder:
                    # I find this very illogical, but as far as I can see now
                    # in the IPs the backorder never arrives.
                    # Now done with fix
                        if self.fix:
                            self._fulfill_order(i, j, backorder)
                            print(f'Outermost but one if else Inventory from {i} to {j} of quantity {backorder}')
                            if i <= self.no_nodes - self.no_customers and i >= tmp1:
                                self.TotalSales[t,i]+=backorder
                            if i <= self.no_suppliers:
                                self.TotalCOGS[t,i]+=backorder
                        else:
                            self.INV[t, i] -= backorder
                            self.BO[t, j, i] = 0
                            # Else, fulfill the entire inventory
                    else:
                        self._fulfill_order(i, j, inventory)
                        print(f'Outermost if else Inventory from {i} to {j} of quantity {inventory}')
                        if i <= self.no_nodes - self.no_customers and i >= tmp1:
                                self.TotalSales[t,i]+=inventory
                        if i <= self.no_suppliers:
                                self.TotalCOGS[t,i]+=inventory
                        self.BO[t, j, i] -= inventory

                        
    def _recieve_incoming_orders_divergent(self):
    # Ship from supplier to warehouse
        t=self.period         
        self.t=self.period  
        tmp=np.array(self.stockpoints_echelon)        
        tmp1=np.sum(tmp[:-2])
        
        self._fulfill_order(0, 1, self.O[t, 1, 0])
        self.TotalCOGS[t,0]+=self.O[t, 1, 0]
        # Check if the warehouse can ship all orders
        print(f'Node-1')
        print(f'Inventory Needed-{np.sum(self.O[t, :, 1], 0)}')
        print(f'Inventory Available-{np.sum(self.INV[t, 1])}')
        if self.INV[t, 1] >= np.sum(self.O[t, :, 1], 0):
            i_list, j_list = np.nonzero(self.connections)
            for i, j in zip(i_list[self.no_suppliers:self.no_suppliers+
                self.no_stockpoints],
                j_list[self.no_suppliers:self.no_suppliers+
                self.no_stockpoints]):
                if self.O[t, j, i] > 0:
                    self._fulfill_order(i, j, self.O[t, j, i])
                    print(f'Outer Loop - Inventory from {i} to {j} of quantity {self.O[t, j, i]}')
                    if i <= self.no_nodes - self.no_customers and i >= tmp1:
                            self.TotalSales[t,i]+=self.O[t, j, i]
                    if i <= self.no_suppliers:
                            self.TotalCOGS[t,i]+=self.O[t, j, i]
        else:
            IPlist = {}
            #i_list, _ = np.nonzero(self.O[t])
            i_list, _ = np.nonzero(self.O[t])
            #print('i_list:',i_list)
            #print('BO:',self.BO[t])
            bo_echelon = np.sum(self.BO[t], 0)
            #print('bo_echelon:',bo_echelon)
            for i in i_list:
                #print(i)
                IPlist[i] = self.INV[t, i] - bo_echelon[i]
            # Check the lowest inventory position and sort these on lowest IP
            sorted_IP = {k: v for k, v in sorted(IPlist.items(), key=lambda item: item[1])}
            print('sorted_IP:',sorted_IP)
            # Check if there is still inventory left
            if self.INV[t, 1] >= 0:
                for i in sorted_IP:
                    # Check if the remaining order can be fulfilled completely
                    if self.INV[t, 1] >= self.O[t, i, 1]:
                        #print('i',i)
                        #self._fulfill_order(t+1, i, self.O[t, i, 1])
                        self._fulfill_order(1, i, self.O[t, i, 1])
                        print(f'Outer Else 1 - Inventory from 1 to {i} of quantity {self.O[t, i, 1]}')
                        if i <= self.no_nodes - self.no_customers and i >= tmp1:
                                self.TotalSales[t,i]+=self.O[t, i, 1]
                        if i <= self.no_suppliers:
                                self.TotalCOGS[t,i]+=self.O[t, i, 1]
                    else:
                        # Else, fulfill how far possible
                        inventory = max(self.INV[t, 1], 0)
                        quantity = self.O[t, i, 1] - inventory
                        self._fulfill_order(1, i, inventory)
                        print(f'Outer Else 2 - Inventory from 1 to {i} of quantity {inventory}')
                        if i <= self.no_nodes - self.no_customers and i >= tmp1:
                                self.TotalSales[t,i]+=inventory
                        if i <= self.no_suppliers:
                                self.TotalCOGS[t,i]+=inventory
                        if self.unsatisfied_demand == 'backorders':
                            self.BO[t, i, 1] += quantity
                            print(f'Backorder from {i} to 1 of quantity {quantity}')
                        else:
                            self.LO[t, i, 1] += quantity
                            print(f'Lost Sales from {i} to 1 of quantity {quantity}')
                        #break
                    
    def _fulfill_order(self, source, destination, quantity):
        # Customers don't have any lead time.
        t=self.period
        
        if destination >= self.no_nodes - self.no_customers:
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
            #print((leadtime))
            #print((self.n))
            #print((self.horizon))
            #print((self.warmup))
            #print((self.divide))
            #print((self.num_of_periods))
            if leadtime < self.n:
                #self.T[t+leadtime, source, destination] += quantity
                self.T[t+leadtime, destination,source] += quantity
                self.in_transit[t+leadtime, source, destination] += quantity
                print(f'Inventory will be increased on period {t+leadtime} of Node {destination} by quantity {quantity}')
            else:
                raise NotImplementedError
                for k in range(0, min(leadtime, self.n) + 1):
                    self.in_transit[t+k, source, destination] += quantity
        # Suppliers have unlimited capacity
        if source >= self.no_suppliers:
            self.INV[t, source] -= quantity
            print(f'Inventory Decreased of Node {source} by quantity {quantity}')
            
    def _place_outgoing_order(self, t, action):
        k = 0
        if self.case_name=='General':
            incomingOrders = np.sum(self.O[t-1], 0)
            # Loop over all suppliers and stockpoints
            for j in range(self.no_suppliers, self.no_stockpoints +
                self.no_suppliers):
                RandomNumber = random.random()
                probability = 0
                for i in range(0, self.no_stockpoints + self.no_suppliers):
                    if self.connections[i, j] == 1:
                        self._place_order(i,j,t,k, action, incomingOrders)
                        
                        k += 1
                    elif self.connections[i,j] > 0:
                        probability += self.connections[i,j]
                        if RandomNumber < probability:
                            self._place_order(i,j,t,k, action, incomingOrders)
                            k += 1
                            break
        elif self.case_name=='Divergent':
            incomingOrders = np.sum(self.O[t], 0)
            # Loop over all suppliers and stockpoints
            for j in range(self.no_suppliers, self.no_stockpoints +
                self.no_suppliers):
                RandomNumber = random.random()
                probability = 0
                for i in range(0, self.no_stockpoints + self.no_suppliers):
                    if self.connections[i, j] == 1:
                        self._place_order(i,j,t,k, action, incomingOrders)
                        
                        k += 1
                    elif self.connections[i,j] > 0:
                        probability += self.connections[i,j]
                        if RandomNumber < probability:
                            self._place_order(i,j,t,k, action, incomingOrders)
                            k += 1
                            break
                        
    def _place_order(self, i, j, t, k, action, incomingOrders):
        
        self.t=self.period
        
        if self.order_policy == 'X':
            self.O[t, j, i] += action[k]
            print(f'Placing order from {j} to {i} of quantity {action[k]}')
            if (self.t < self.horizon - 1) and (self.t >= self.warmup-1):
                self.TotalDemand[j,i] += action[k]
        
        elif self.order_policy == 'X+Y':
            self.O[t, j, i] += incomingOrders[j] + action[k]
            print(f'Placing order from {j} to {i} of quantity {incomingOrders[j] + action[k]}')
            if (self.t < self.horizon - 1) and (self.t >= self.warmup-1):
                self.TotalDemand[j,i] += incomingOrders[j] + action[k]
        
        elif self.order_policy == 'BaseStock':
            bo_echelon = np.sum(self.BO[0], 0)
            self.O[t, j, i] += max(0, action[k]-(self.INV[0,j]+self.in_transit[0,i,j]-bo_echelon[j]))
            print(f'Placing order from {j} to {i} of quantity {max(0, action[k]-(self.INV[0,j]+self.in_transit[0,i,j]-bo_echelon[j]))}')
            if (self.t < self.horizon - 1) and (self.t >= self.warmup-1):
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
        return action,0.0
    
    def step(self, action, visualize=False):
        """
        Execute one step in the RL method.
        
        input: actionlist, visualize
        """
        self.leadtime = self.generate_leadtime(0, self.leadtime_dist,self.leadtime_lb, self.leadtime_ub)
        print('-----------------------------------Period :',self.period)
        action, penalty = self._check_action_space(action)
        
        self._initialize_state()
        
        print('Action :',action)
        print('State at start :',self.state)
        if visualize: self._visualize("0. IP")
        
        

        if self.case_name == "General" or self.case_name=='Linear':
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
            self._place_outgoing_order(self.period, action)
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
        holding_costs, backorder_costs,lost_sales_costs,revenue,cost_of_goods = self.calculate_reward()
        reward = revenue-(cost_of_goods+holding_costs + backorder_costs+lost_sales_costs + penalty )
        
        print('Inventory at end of period :',self.INV[self.period])
        
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
        print('Revenue :',revenue)
        print('COGS :',cost_of_goods)
        print('Holding Costs :',holding_costs)
        print('Back Order Costs :',backorder_costs)
        print('Lost Order Sales :',lost_sales_costs)
        print('Reward :',reward)
        
        
        return self.state, reward/self.divide, done,{}

    def reset(self):
    
        # initializetion
        self.period = 0 # initialize time
    
        self.INV = [[0 for _ in range(self.no_nodes)] for _ in range(self.num_of_periods+10)] #Inventory at each stock point
        
        self.BO = [[[0 for _ in range(self.no_nodes)] for _ in range(self.no_nodes)] for _ in range(self.num_of_periods+10)] #BackOrder for every i,j connection
        self.LO = [[[0 for _ in range(self.no_nodes)] for _ in range(self.no_nodes)] for _ in range(self.num_of_periods+10)] #BackOrder for every i,j connection ##lost sales 
        
        self.in_transit = [[[0 for _ in range(self.no_nodes)] for _ in range(self.no_nodes)] for _ in range(self.num_of_periods+10)] #Intrasnit for every i,j connection
        
        self.T = [[[0 for _ in range(self.no_nodes)] for _ in range(self.no_nodes)] for _ in range(self.num_of_periods+10)] #Delivery for every i,j connection
        
        self.O = [[[0 for _ in range(self.no_nodes)] for _ in range(self.no_nodes)] for _ in range(self.num_of_periods+10)] #Order for every i,j connection
        
        self.TotalFulfilled = [[0 for _ in range(self.no_nodes)] for _ in range(self.no_nodes)] #Total Fulfilled demand for every i,j connection
        
        self.TotalDemand = [[0 for _ in range(self.no_nodes)] for _ in range(self.no_nodes)] #Total Demand for every i,j connection
        
        self.TotalBO = [[0 for _ in range(self.no_nodes)] for _ in range(self.no_nodes)] #Total Backorders for every i,j connection
        
        self.TotalSales = [[0 for _ in range(self.no_nodes)] for _ in range(self.num_of_periods+10)] #Inventory at each stock point
        self.TotalCOGS = [[0 for _ in range(self.no_nodes)] for _ in range(self.num_of_periods+10)] #Inventory at each stock point
        #self.TotalCOGS = [[0 for _ in range(self.no_nodes)] for _ in range(self.num_of_periods+10)] #Inventory at each stock point
        
        self.INV=np.array(self.INV)
        self.BO=np.array(self.BO)
        self.LO=np.array(self.LO)
        self.in_transit=np.array(self.in_transit)
        self.T=np.array(self.T)
        self.O=np.array(self.O)
        self.TotalFulfilled=np.array(self.TotalFulfilled)
        self.TotalDemand=np.array(self.TotalDemand)
        self.TotalBO=np.array(self.TotalBO)
        self.TotalSales=np.array(self.TotalSales)
        self.TotalCOGS=np.array(self.TotalCOGS)
        return self.state_low
