# Order-Execution-Strategy

## About the Project

Literature survey of popuplar trade slicing algorithms. The algorithms are implemented in python and backtested using real time stock data traded on the Indian stock market (NSE). The survey starts with the review of the seminal work of Robert Almgren and Neil Chriss and continues to other works that are built on top of the framework. All the hyper-parameters are tuned using methods proposed by the authors.

## Codes

* simulation_engine.py - Python file implementing the market impact function as proposed in the almgren-chriss framework (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1654151)
* gbm.py - Python file implementing the optimal order execution as given by Jim Gatheral and Alexander Scheid (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1654151) 
* konishi.py - Python file implementing the optimal order execution as given by Konishi (https://www.sciencedirect.com/science/article/abs/pii/S1386418101000234) 
* almgren_chris.py - Python file implementing the optimal orcer execution strategy as given by Robert Almgren and Neil Chris. (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1654151)

## Authors

* Kousik Krishnan - Quantitative Analyst, Credit Suisse (https://www.linkedin.com/in/kousik-krishnan-239073101/)
* Sushant Vijayan - Doctoral Candidate at Tata Institute of Fundamental Research (https://www.linkedin.com/in/sushant-vijayan-3574886a/)
