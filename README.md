# Stocks-Analysis-PCA
In this task I stocks analysis on the S&P 500 in 2016. The rows data was on each stock, it close price of each day.
The data was represented by two datasets, one of the closing price of the stock on each day in 2016. and the second data is the sectors of each symbol. The prices data is represented by features as Symbol, open price, close price, lowest price, highest price and volume.
I did pca for each stock to 2-d dimension and created function that gets sectors list and show scatter plot of the stocks in this sector after the PCA.
I did data scaling with substract the ln values(ln(pi+1)-ln(p)).
With those functions I did stocks analysis by the scatter-plots.
## Prices.csv
The csv file of the first data.
## Securities.csv
The csv file of the second data.
## Stocks.py
The code file with all the functions.
