##Important Libraries that are imported
import numpy as np
import os
import pandas as pd
import datetime


class algo:

    '''Class Implementing the Konishi Algorithm backtested in historical data
    First, create an object of the class as follows:
    object_name = algo(folder containing the market data)

    Note : The folder must contain csv files containing minute by minute market data in the format mentioned.

    To call the backtest function, call object_name.backtest(parameters)

    Parameters
    ----------

    k : (Int) (Default:5 days) The number of previous day samples to compute E(X(t)) from

    N : (Int) (Default:100), Number of shares to sell - Integer

    T : (Int) (Default:350) Number of intervals for a day. In the default case, the data is minute by minute and we would like to have a day as consisiting of 350mins

    save_output = String, Default : False, Path to save the file. The file will be saved in the given path as Results.csv

    print_output : Bool, Default: True, Whether to print the results or not.


    Returns
    -------

    DataFrame Consisting of the following columns : ['Ticker','Number Samples','Avg. VWAP Difference','Avg. Competetive Ratio','Average Loss','VWAP % difference','Average Price High','Std. VWAP Difference','Std. Loss','Avg. Last Firing']. For each ticker all the associated averages are returned.

    Example
    -------

    >>> from algorithm import algo
    >>> obj = algo('./18-Jun-2018-Stock-Data/') or obj = algo('./Sample/')
    >>> obj.backtest(k=5,N=100,T=350,save_output="./",print_output=True)
    '''
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
            filenames = [x[:-4] for x in filenames]
            sorted(filenames, key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
            filenames = [x+'.csv' for x in filenames]
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

    #Get accumulated volume till iter for the day's data:
    def __get_volume(self, data, iter):
        total_volume = 0
        for i in range(0, iter+1):
            total_volume += data[i]['Volume']
        return(total_volume)

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
        last_minute_firing = (num_shares-sum(new_volume_list))
        new_volume_list[-1]+= last_minute_firing
        assert last_minute_firing >= 0
        #print('Last minute firing : ',last_minute_firing)
        return(new_volume_list,last_minute_firing)

    #Returns the estimate of E(X(t))
    def __get_e_x_t(self, data_list, k, iter, t, T):
        e_x_t = np.mean([((self.__get_volume(data,T-1) - self.__get_volume(data,t))/(self.__get_volume(data,T-1))) for data in data_list[(iter-k):iter]])
        return(e_x_t)


    #Main Algorithm:
    def __algorithm(self, data_list, iter, k, N, T):
        v_star_list = [0]
        price_list = []
        for t in range(0, T):
            e_x_t = self.__get_e_x_t(data_list,k,iter,t,T)
            x_star = np.floor(((2*N*e_x_t)+1)/2)/N
            v_star = N*(1-x_star)
            assert v_star >= v_star_list[-1]
            v_star_list.append(v_star)
            price_list.append(data_list[iter][t]['Close'])
        assert (v_star) == N
        buy_list = []
        for i in range(1,len(v_star_list)):
            buy_list.append(v_star_list[i] - v_star_list[i-1])
        volume_list, last_minute_firing = self.__process_volume(buy_list,N)
        return(buy_list,price_list,last_minute_firing)

    #Backtest Main Function:
    def backtest(self, k=5, N=100, T=350, print_output=True,save_output=False):
        #Iterating algo over all possible tickers
        #Variables to store results:
        df = pd.DataFrame(columns = ['Ticker','Number Samples','Avg. VWAP Difference','Avg. Competetive Ratio','Average Loss','VWAP % difference','Average Price High','Std. VWAP Difference','Std. Loss','Avg. Last Firing'])
        for ticker in self.data.keys():
            if(len(self.data[ticker])<=k):#We want aleast k day's data for the ticker
                continue
            #List of variables storing the
            vwap_diff = []
            comp_ratio = []
            opt_diff = []
            last_firing = []
            vol_list = []
            vwap_market_dict = []
            price_high_market = []
            #Get All Data
            data_list = self.data[ticker]
            for iter in range(k,len(data_list)):
                if(self.__get_market_stats(data_list[iter]) == -1):
                        break
                vwap_market, total_volume, price_high = self.__get_market_stats(data_list[iter])
                try:
                    volume_list, price_list, last_minute_firing = self.__algorithm(data_list,iter,k,N,T)
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
                vwap_market_dict.append(vwap_market)
                last_firing.append(last_minute_firing)
                price_high_market.append(price_high)

            df = df.append(pd.Series([ticker,len(self.data[ticker])-k,np.mean(vwap_diff),np.mean(comp_ratio),np.mean(opt_diff),(np.mean(vwap_diff)/np.mean(vwap_market_dict))*100,np.mean(price_high_market),np.std(vwap_diff),np.std(opt_diff),np.mean(last_firing)],index = df.columns),ignore_index = True)
        df = df.set_index('Ticker')
        if(print_output ==True):
            print(df)
        if(save_output!=False):
            df.to_csv(save_output+'Results.csv')
            print('Saved in and as : ',save_output+'Resuts.csv')
        return(df)
