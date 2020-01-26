##Important Libraries that are imported
import numpy as np
import matplotlib.pyplot as plt
import os
import numbers
import pandas as pd

class algo:

    '''Class implementing the backtesting of market data.

    First, create an object of the class as follows:
    object_name = algo(folder containing the market data)

    Note : The folder must contain csv files containing minute by minute market data in the format mentioned.

    To call the backtest function, call object_name.backtest(parameters)

    Parameters
    ----------

    method : String, Whether to backtest the static or dynamic version.
    Options : 'static' or 'dynamic'

    algorithm : String, Whether to use the vanilla algorithm or the modified versionself. Options : 'vanilla' or 'modified'

    num_shares : Int, Number of shares to sell - Integer

    start : Int, Default : 1, The number of minutes to wait in the start of the day before executing the algorithm. Note : In the static case, start is 1 while in the dynamic case, start must be greater than or equal to window_length.

    window_length : Int, Default : 15, If methdod is 'dynamic', the window length is the past length from which the parameters will be dynamically estimated.

    print_output : Bool, Default: True, Whether to print the results or not.

    p1 : Float, The p1 estimate. Default : 0.1

    p2 : Float, The p2 estimate. Default : 0.01

    save_output = String, Default : False, Path to save the file. The file will be saved in the given path as Results.csv

    lamba : Float, The lambda estimate. Default : 2.

    Returns
    -------

    vwap_diff : Dictionary of the Difference vwap_market - vwap_algo for all the backtested days

    comp_ratio : Dictionary of Competetive ratio for different days

    opt_diff : Dictionary of the loss (max_price - vwap_algo) for the backtested days

    last_firing : Dictionary of the last last minute fired number of shares.

    Example
    -------

    >>> from algorithm import algo
    >>> obj = algo('./Stock_Data_Processed/')
    >>> obj.backtest('static','modified',5000,print=True,start = 1, p1=0.00001,p2=0.000001,lamba=2)
    >>> obj.plot_n_shares('static','modified')

    '''

    #Some Variables:
    available_algorithms = ['vanilla','modified']
    available_methods = ['static','dynamic']

    #Temporatry Index
    ind = 0


    #Processing the input folder data:


    ##Constructor, creates a class with a folder_name
    def __init__(self, data_folder):
        self.__check_folder_exists(data_folder)
        self.folder_name = data_folder
        self.data = self.__get_files(data_folder)


    ##Check if the given folder exists
    def __check_folder_exists(self, data_folder):
        if(not os.path.isdir(data_folder)):
            raise Exception('No Such Directory')

    #Get required files
    def __get_files(self,data_folder):
        file_dict = {}
        data_dict = {}
        list_dirs = os.listdir(data_folder)
        for ticker in list_dirs:
            dir = data_folder + ticker
            if(not os.path.isdir(dir)):
                continue
            filenames = os.listdir(dir)
            filenames = [os.path.join(dir,file) for file in filenames]
            file_dict[ticker] = filenames
        ##Checking and getting data from CSV Files
        for ticker in file_dict.keys():
            self.__check_files(file_dict[ticker])
            data_dict[ticker] = self.__get_data(file_dict[ticker])
        return(data_dict)

    ##Checks if the csv files are in proper format
    def __check_files(self, files):
        if(len(files)==0):
            raise Exception('No CSV Files')
        for file in files:
            try:
                df = pd.read_csv(file)
                assert len(df) > 0
            except:
                raise Exception('Some csv files cannot be read/ has no data in them')
            if(not set(['Close','Open','High','Low','Volume']).issubset(set(df.columns))):
                raise Exception('Some files are not in the right format')

    #Checks if an argument is correct
    def __check_input(self, arg, avail_list):
        if(not arg in avail_list):
            raise Exception('Check Input : ', arg,'not available in list')

    #Checks if the given objects are decimals
    def __check_float(self, args):
        for arg in args:
            if(not isinstance(arg,numbers.Number)):
                raise Exception('p1, p2 and lambda must be floating point numbers')

    #Returns data in the format that we want:
    def __get_data(self, files):
        datas = []
        for file in files:
            df = pd.read_csv(file)
            data = df.to_dict(orient = 'index')
            val = list(data.values())
            datas.append(val)
        return(datas)

    #Returns the market stats for the data:
    def __get_market_stats(self, data):
        volume_list = []
        price_list = []
        for j in data:
            volume_list.append(j['Volume'])
            price_list.append(j['Close'])
        total_volume = sum(volume_list)
        if(total_volume==0):
            return(-1)
        vwap = (sum([a*b for a,b in zip(volume_list,price_list)]))/total_volume
        assert vwap>0
        return(vwap,total_volume,max(price_list))


    #Function to return volume which are positive integers:
    def __process_volume(self, volume_list, num_shares):
        new_volume_list = []
        for i in range(0,len(volume_list)-1):
            volume_till_now = sum(volume_list[:i+1])
            new_sum_volme = sum(new_volume_list)
            assert int(volume_till_now-new_sum_volme)>=0
            new_volume_list.append(int(np.floor(volume_till_now-new_sum_volme)))
        #assert sum(new_volume_list) == num_shares
        assert sum(new_volume_list) <= num_shares
        last_minute_firing = (num_shares-sum(new_volume_list)-volume_list[-1])
        new_volume_list.append(last_minute_firing+volume_list[-1])
        assert last_minute_firing >= 0
        #print('Last minute firing : ',last_minute_firing)
        return(new_volume_list,last_minute_firing)

    #Function to plot price_figures:
    def __plot_figure(self, price_list, price_impact, price_sold):
        fig = plt.figure()
        plt.plot(range(0,len(price_list)),price_list)
        plt.plot(range(0,len(price_list)),price_impact[1:])
        plt.plot(range(0,len(price_list)),price_sold)
        plt.legend(['Historical Market Price','Permanent Impact Price','Price Sold by Algo'],loc='upper left')
        fig.savefig('./Plots/Price' + str(self.ind) + '.png', dpi=fig.dpi)

    #Function to

    #Function to get the price list:
    def __get_price_list(self, price_list, vol_list, gamma_list, epsilon_list, eta_list, tou_list, day_open_price):
        price_impact = [day_open_price] + [(price_list[i] - (gamma_list[i]*vol_list[i])) for i in range(0,len(price_list))]
        for i in range(1,len(price_impact)):
            print("Epsilon : "+epsilon_list[i-1]+" Eta/Tou : "+(eta_list[i-1]/tou_list[i-1])+"\n")
        price_sold = [(price_impact[i-1] - (epsilon_list[i-1]*(np.sign(vol_list[i-1]))) - ((eta_list[i-1]/tou_list[i-1])*vol_list[i-1])) for i in range(1,len(price_impact))]
        #self.__plot_figure(price_list,price_impact,price_sold)

        ##Return
        return(price_sold)

    #Main Vanilla Algorithm: (Single algo to implement both static and dynamic methods)
    def __algorithm(self, method, data, start, window_length, num_shares,  lamba, p1, p2):
        ##Algo Begins:
        if(method == 'static'): #Even static elements can have start slack
            alpha,sigma,gamma,eta,epsilon = self.__estimate_parameters_current_day(data,p1,p2)
            start = 1
            N = len(data)
            T = 1
            tou = T/N
        else:
            #The start must be greater than or equal to window_length
            if(start<=window_length):
                start = window_length
            N = len(data) - start + 1
            T = 1 - ((start-1)/len(data))
            tou = T/N
        #Variables to store results
        X = num_shares
        volume_list = []
        hist_price_list = []
        ##Lists to store all the necessary information which impacts the price calculation:
        gamma_list = []
        epsilon_list =[]
        eta_list = []
        tou_list = []
        ##Day open Price:
        day_open_price = data[start]['Open']
        #Iterate algorith throughout the day
        for iter in range(start,len(data)+1):
            if(method=='dynamic'): #Update parameters only if the method is dynamic
                alpha,sigma,gamma,eta,epsilon = self.__dynamic_estimate_parameter(data,iter,window_length,p1,p2)
            eta_hat = eta*(1-((gamma*tou)/(2*eta)))
            kappa_hat_squared = (lamba*pow(sigma,2))/eta_hat
            kappa = (np.arccosh(((kappa_hat_squared*(pow(tou,2)))/2)+1))/tou
            x_bar = alpha/(2*lamba*pow(sigma,2))
            if (method=='dynamic'):
                t_j_half = (1-0.5)*tou
            else:
                t_j_half = (iter-0.5)*tou
            a_1 = ((2*np.sinh(0.5*kappa*tou))/(np.sinh(kappa*T)))
            a_2 = np.cosh(kappa*(T-t_j_half))
            a_3 = np.cosh(kappa*t_j_half)
            n_j = (a_1*a_2*X) + (a_1*(a_3-a_2)*x_bar)
            ##List to update the information required to obtain the prices:
            gamma_list.append(gamma)
            epsilon_list.append(epsilon)
            eta_list.append(eta)
            tou_list.append(tou)
            ##Updating volume list:
            volume_list.append(n_j)
            ##Updating historical price:
            hist_price_list.append(data[iter-1]['Close'])
            if(method=='dynamic'):
                X = X - (n_j)
                T = 1 - ((iter)/len(data))
                #tou = T/N
        #print('Volume sold : ',sum(volume_list))
        try:
            assert int(np.around(sum(volume_list)))== num_shares
        except:
            return(-1)
            print('Volume Not Conserved')
        volume_list, last_minute_firing = self.__process_volume(volume_list,num_shares)
        ##Price Updation:
        price_list = self.__get_price_list(hist_price_list,volume_list,gamma_list, epsilon_list, eta_list, tou_list, day_open_price)
        ##Return:
        return(volume_list,price_list,last_minute_firing)


    #Modified Algortihm : (Single algo to implement both static and dynamic methods)
    def __modified_algorithm(self, method, data, start, window_length, num_shares,  lamba, p1, p2):
        if(method == 'static'): #Even static elements can have start slack
            alpha,sigma,gamma,eta,epsilon = self.__estimate_parameters_current_day(data,p1,p2)
            start = 1
            N = len(data)
            T = 1
            tou = T/N
        else:
            #The start must be greater than or equal to window_length
            if(start<window_length):
                start = window_length
            N = len(data) - start + 1
            T = 1 - ((start-1)/(len(data)))
            tou = T/N #(1/(len(data)-start))
        #Variables to store results
        X = num_shares
        volume_list = []
        hist_price_list = []
        ##Lists to store all the necessary information which impacts the price calculation:
        gamma_list = []
        epsilon_list =[]
        eta_list = []
        tou_list = []
        ##Day open Price:
        day_open_price = data[start]['Open']
        #Iterate the function for the entire day
        for iter in range(start,len(data)+1):
            if(method=='dynamic'): #Update parameters only if the method is dynamic
                alpha,sigma,gamma,eta,epsilon = self.__dynamic_estimate_parameter(data,iter,window_length,p1,p2)
            eta_hat = eta*(1-((gamma*tou)/(2*eta)))
            kappa_hat_squared = (lamba*pow(sigma,2))/eta_hat
            kappa = (np.arccosh(((kappa_hat_squared*(pow(tou,2)))/2)+1))/tou
            x_bar = alpha/(2*lamba*pow(sigma,2))
            A=np.cosh(kappa*tou)-np.sinh(kappa*tou)
            B=np.cosh(kappa*tou)+np.sinh(kappa*tou)
            c1=((X*pow(A,N))+x_bar*(1-(pow(A,N))))/(pow(A,N)-pow(B,N))
            c2=((-X*pow(B,N))+(x_bar*(pow(B,N)-1)))/(pow(A,N)-pow(B,N))
            if(method=='dynamic'):
                ##Check this
                n_j=(c1*pow(B,1-1))+(c2*pow(A,1-1))-(c1*pow(B,1))-(c2*pow(A,1))
            else:
                n_j=(c1*pow(B,iter-1))+(c2*pow(A,iter-1))-(c1*pow(B,iter))-(c2*pow(A,iter))
            ##List to update the information required to obtain the prices:
            gamma_list.append(gamma)
            epsilon_list.append(epsilon)
            eta_list.append(eta)
            tou_list.append(tou)
            ##Updating volume list:
            volume_list.append(n_j)
            ##Updating historical price:
            hist_price_list.append(data[iter-1]['Close'])
            if(method=='dynamic'):
                X = X - (n_j)
                #X[i+1] = X[i] - n[i]
                T = 1 -(iter/len(data))
                N = T/(tou)
                #tou = T/N
                '''try:
                    assert tou == (T/N)
                except:
                    print('Tou not conserved')'''
        #print('Volume sold : ',sum(volume_list))
        try:
            assert int(np.around(sum(volume_list)))== num_shares
        except:
            return(-1)
            print('Volume Not Conserved')
        volume_list, last_minute_firing = self.__process_volume(volume_list,num_shares)
        ##Price Updation:
        price_list = self.__get_price_list(hist_price_list,volume_list,gamma_list, epsilon_list, eta_list, tou_list, day_open_price)
        ##Return:
        return(volume_list,price_list,last_minute_firing)

    #Returns the estimate of all parameters for current day:
    def __estimate_parameters_current_day(self,data,p1,p2):
        initial_price = data[0]['Open']
        bid_ask_list = []
        volume_list = []
        returns = []
        for j in data:
            bid_ask_list.append(j['High']-j['Low'])
            volume_list.append(j['Volume'])
            returns.append((j['Close'] - j['Open'])/j['Open'])
        avg_returns = np.mean(returns)
        std_returns = np.std(returns)
        bid_ask_spread = np.mean(bid_ask_list)
        avg_volume = np.mean(volume_list)
        epsilon = bid_ask_spread/2
        eta = bid_ask_spread/(p1*avg_volume)
        gamma = bid_ask_spread/(p2*avg_volume)
        alpha = avg_returns*initial_price
        sigma = std_returns*initial_price
        return(alpha,sigma,gamma,eta,epsilon)

    #Dynamic Estimate of parameters:
    def __dynamic_estimate_parameter(self,data,iter_step,k,p1,p2):
        initial_price = data[iter_step-1]['Close']
        bid_ask_list = []
        volume_list = []
        returns = []
        for i in range(iter_step-k,iter_step):
            bid_ask_list.append(data[i]['High'] - data[i]['Low'])
            volume_list.append(data[i]['Volume'])
            returns.append((data[i]['Close'] - data[i]['Open'])/data[i]['Open'])
        avg_returns = np.mean(returns)
        std_returns = np.std(returns)
        bid_ask_spread = np.mean(bid_ask_list)
        avg_volume = np.mean(volume_list)
        epsilon = bid_ask_spread/2
        eta = bid_ask_spread/(p1*avg_volume)
        gamma = bid_ask_spread/(p2*avg_volume)
        alpha = avg_returns*initial_price
        sigma = std_returns*initial_price
        return(alpha,sigma,gamma,eta,epsilon)

    #Back-test Main Function
    def backtest(self, method, algorithm, num_shares, start = 1, window_length =15, print_output = True,save_output=False,  p1=0.01, p2=0.1, lamba=2):
        #Checks if the inputs are right
        self.__check_input(method,self.available_methods)
        self.__check_input(algorithm,self.available_algorithms)
        self.__check_float([num_shares,p1,p2,lamba])
        #Variables to store results:
        df = pd.DataFrame(columns = ['Ticker','Number Samples','Avg. VWAP Difference','Avg. Competetive Ratio','Average Loss','VWAP % difference','Average Price High','Std. VWAP Difference','Std. Loss','Avg. Last Firing'])
        #Iterate the algorithm over all files
        for ticker in self.data.keys():
            print(ticker)
            vwap_diff = []
            comp_ratio = []
            opt_diff = []
            last_firing = []
            vol_list = []
            vwap_market_dict = []
            price_high_market = []

            for data in self.data[ticker]:
                self.ind+=1
                if(self.__get_market_stats(data) == -1):
                    break
                vwap_market, total_volume, price_high = self.__get_market_stats(data)
                if(algorithm == 'vanilla'):
                    try:
                        volume_list, price_list, last_minute_firing = self.__algorithm(method, data, start, window_length, num_shares, lamba, p1, p2)
                    except:
                        print('Error in Algorithm for '+ticker)
                        continue
                else:
                    try:
                        volume_list, price_list, last_minute_firing = self.__modified_algorithm(method, data, start, window_length, num_shares, lamba, p1, p2)
                    except:
                        print('Error in Algorithm for '+ticker)
                        continue

                #Post getting the volume list and price list, get stats
                total_volume = sum(volume_list)
                total_sell_amount = sum([a*b for a, b in zip(volume_list,price_list)])
                vwap_algo = total_sell_amount/total_volume

                #Update the list which stores the stats for that ticker for all days
                vwap_diff.append(vwap_market - vwap_algo)
                comp_ratio.append(vwap_market/vwap_algo)
                opt_diff.append(price_high - vwap_algo)
                last_firing.append(last_minute_firing)
                vwap_market_dict.append(vwap_market)
                price_high_market.append(price_high)
            df = df.append(pd.Series([ticker,len(self.data[ticker]),np.mean(vwap_diff),np.mean(comp_ratio),np.mean(opt_diff),(np.mean(vwap_diff)/np.mean(vwap_market_dict))*100,np.mean(price_high_market),np.std(vwap_diff),np.std(opt_diff),np.mean(last_firing)],index = df.columns),ignore_index = True)
        df = df.set_index('Ticker')
        if(print_output ==True):
            print(df)
        if(save_output!=False):
            df.to_csv(save_output+'Results.csv')
            print('Saved in and as : ',save_output+'Resuts.csv')
        return(df)
