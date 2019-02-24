import numpy as np
import pandas as pd

##Class Implementing the Simulation Engine
class simulation:
    ##Prepare necessary data
    def _prepare_data(self, data, take_avg):
        '''
            Function that takes the raw data and processes them to get the volume and price for every minute interval
            Input
            -----
                data : (pd dataframe) EOM data
                take_avg : (list) List of params to take average of to get the price points
        '''
        self.volume = list(data['Volume'])
        ##if any fancy processing of volume to be done - do here
        self.data['avg'] = self.data[take_avg].mean(axis=1)
        self.price = list(self.data['avg'])
        self.time = 0 #Finds at what time we are located
        self.time_last_order = 0
        self.last_permanent_impact = 0
        self.last_temp_impact = 0
        self.executed_orders = []
        assert len(self.volume) == len(self.price) #qual check


    ##Constructor
    def __init__(self, data_file, params={'data_avg':['Close'],'method':'almgren','method_params':{'ro':7}}):
        '''
            Constructor function
            Input
            -----
                data_file : (pd dataframe) EOM data
                params : (dictionary) Other parameters
        '''
        self.data = data_file
        self._prepare_data(self.data, params['data_avg'])
        if(params['method']=='almgren'):
            self.method = 'almgren'
            self.alm_p1 = params['method_params']['p1']
            self.alm_p2 = params['method_params']['p2']
            self.exp_param = params['method_params']['ro']
            self.tou = params['method_params']['tou']
            self.alm_window_length = params['method_params']['window_length']


    ##main order call function:
    def order(self, n_t, t):
        if(self.method=='almgren'):
            return(self.__almgren_order(n_t,t))


    ##almgeren chris feeder function
    def __almgren_estimate_parameter(self,iter_step,window_length,p1,p2):
        ##Windows intial price
        initial_price = self.price[iter_step-1]
        assert iter_step-window_length>=0
        bid_ask_list = []
        volume_list = []
        returns = []
        ##snapshot of the current window
        for i in range(iter_step-window_length,iter_step):
            bid_ask_list.append(self.data[i]['High'] - self.data[i]['Low'])
            volume_list.append(self.volume[i])
        ##some major params
        bid_ask_spread = np.mean(bid_ask_list)
        avg_volume = np.mean(volume_list)
        ##parameter estimation
        epsilon = bid_ask_spread/2
        eta = bid_ask_spread/(p1*avg_volume)
        gamma = bid_ask_spread/(p2*avg_volume)
        #alpha = avg_returns*initial_price // not required
        #sigma = std_returns*initial_price // not required
        return(gamma,eta,epsilon)

    #order update that follows the almgren way
    def __almgren_order(self, n_t, t):
        assert t>=self.time_last_order
        if(self.time_last_order==t):
            self.price = self.__update_subsequent(self.price, t, -self.last_permanent_impact, self.exp_param)
        else:
            self.last_permanent_impact = 0
        gamma,eta,epsilon = self.__almgren_estimate_parameter(t,self.alm_window_length,self.alm_p1,self.alm_p2)
        #calculate the permanent/temp impact
        self.last_permanent_impact += (self.tou*self.__g(n_t,self.tou,gamma))
        self.last_temp_impact = self.__h(epsilon, n_t, eta, self.tou)
        #update the impacts
        current_price = self.price[t] - self.last_temp_impact ##calculated before perm impact updation
        self.price = self.__update_subsequent(self.price, t,self.last_permanent_impact, self.exp_param)
        ##return
        self.time = t
        self.time_last_order = t
        self.executed_orders.append((current_price,n_t))
        return(current_price)

    #update permanent impact across all future prices
    def __update_subsequent(self, iter, impact, ro):
        price = self.price
        for i in range(iter, len(price)):
            assert i-iter>=0
            price[i] -= (impact*(exp(-ro*(i-iter))))
        return price

    #permanent impact function
    def __g(self, n_t, tou, gamma):
        return(gamma*n_t/tou)

    #temporary impact function
    def __h(self, epsilon, n_t, eta, tou):
        temp = epsilon*np.sign(n_t) + (eta/tou)*n_t
        return(temp)


######################################################
#  n_t -> +ve(Sell), -ve(buy)
######################################################
