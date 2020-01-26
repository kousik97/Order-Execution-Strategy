##Libraries
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


##Simulation Engine
from simulation_engine import simulation


class gbm:

    ##Constructor function
    def __init__(self, num_shares, algo_params={'window_length':15,'T':375,'T_0':100,'VAR_Q':0.995,'p1':0.01,'p2':0.1,'p3':200}):
        self.T = algo_params['T'] - (2*algo_params['window_length'])
        self.X = num_shares
        self.T_0 = algo_params['T_0']
        self.carry_forward = 0
        self.window_length = algo_params['window_length']
        self.p1 = algo_params['p1']
        self.p2 = algo_params['p2']
        self.p3 = algo_params['p3']
        self.var = algo_params['VAR_Q']
        self.last_x_t = num_shares
        self.last_order = 0

    #To estiamte : eta(temp impact), sigma(brownian motion),
    def __estimate_parameters(self, price, high, low, volume):
        ##sigma estimation
        price = [np.log(x) for x in price]
        diff = np.diff(price)
        diff = [i**2 for i in diff]
        sigma = np.sqrt(2*(np.sqrt(np.mean(diff)+1)-1))
        ##eta estimation
        bid_ask_spread = np.mean(high) - np.mean(low)
        avg_volume = np.mean(volume)
        x_t_dot = self.p3*avg_volume
        eta = (bid_ask_spread/(self.p1*avg_volume)) - (bid_ask_spread/(2*x_t_dot))
        return(sigma, eta)

    #Lambda hat calculation
    def __calculate_lambda_hat(self, sigma):
        return(1-np.exp((-sigma*np.sqrt(self.T_0)*norm.ppf(self.var))-(pow(sigma,2)*(self.T_0)/2)))

    def order(self, price, high, low, volume, t):
        ##updating params
        sigma, eta = self.__estimate_parameters(price, high, low, volume)
        ##calculating lambda and integral
        lamba = self.__calculate_lambda_hat(sigma)/eta
        integral = np.sum(price[self.window_length-1:])
        ##updation
        x_t = max(((self.T-t)/self.T)*((self.X)-((lamba/4)*integral)),0)
        n_t = self.last_x_t - x_t
        if(n_t>=0):
            self.last_x_t = x_t
        n_t = max(n_t,0)
        #volume correction
        final_order = np.floor(n_t+self.carry_forward)
        self.carry_forward = n_t+self.carry_forward - final_order
        ##assertions
        #assert self.carry_forward<=1
        #assert final_order>=0
        #assert final_order == int(final_order)
        ##Return Final result
        self.last_order = final_order
        #print(t, x_t, n_t, final_order)
        return(final_order)

    def order_update(self, unrealised):
        self.carry_forward += unrealised

data = "../Data/NSE-EOM/ABAN/2018-05-31.csv"
data = pd.read_csv(data)
sim = simulation(data,dump_data=True)
met = gbm(10000)
order_list = []
price_list = []
for i in range(15,361):
    high = list(data['High'])[:i]
    low = list(data['Low'])[:i]
    price = list(data['Close'])[:i]
    volume = list(data['Volume'])[:i]
    ord = met.order(price, high, low, volume, i-15)
    price = sim.order(ord, i)
    met.order_update(0)
    order_list.append(ord)
    price_list.append(price)
print(price_list)
'''
plt.plot(range(0,len(price_list)),data['Close'][15:361],'b')
plt.plot(range(0,len(price_list)),price_list,'g')
#plt.plot(range(0,len(order_list)),order_list)
#plt.ylim([28980,28990])

plt.show()
'''
