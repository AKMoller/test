# Coding exercise
The purpose of the following exercise is to assess your modeling and data science skills, how you 
deal with timeseries data, illustrate how some of our activities take place, and be basis for 
discussion during the interview.

## Introduction
An important activity for Ørsted is to buy and sell power on the European markets. Having an
informed view on the future production of power therefore allows for a competitive advantage when
placing bids and offers. 
One of the inputs to this view comes from production forecast for Ørsted's 
windfarms provided by a number of vendors. These vendors take into account the location of the 
windfarms, the placement of the wind turbines, the weather, and other factors and then provide 
daily forecast for the production for the windfarms for the next 24-hour period.

## Data
This repository contains two pickle files, `features.pkl` and `target.pkl`. 

`features.pkl` is a time-indexed Pandas dataframe with three columns, `0`, `1`, and `2`. The values 
represent the forecasted production from three forecast vendors for a portfolio of Ørsted windfarms 
(data has been transformed from its original). The units are MWh for the 30-minute window given by 
the index.
`target.pkl` contains the actual production from the portfolio for the same time range.

##  Task
Your task is to create a Python class with an `sklearn`-style API (at least `fit`, `predict`, and 
`score` methods should be implemented) to predict the actual production based on the forecasted 
productions from the three vendors.

There is no limitation in terms of modeling techniques, feature engineering/cleaning, or extra data 
you may want to include. If you do include extra data, please add this to the submission.
Likewise, please add a `requirements.txt` file for any packages that are needed to run your code.

If the model takes more than a couple of minutes to train on a normal laptop, then please add a
method in the class to load a pre-trained model (e.g. by pickling it) and include the pickled
model in the submission.

We will evaluate:
- Feature engineering and model creation
- Code quality and any unit tests of your code you may wish to add.

You will be asked to talk about and go through your solution which you can do just by going through
your source code or in the form of a Jupyter notebook if that helps you present (we will look at
source code either way).


### Additional questions

Please also be ready to talk about/discuss the following

**Domain**
- What is the average weekly production? Does this number sound fair?
- Would you expect seasonality in this kind of data? If so, why and what kind?
- If you had been able to ask questions about the domain, would that have helped you? What
 questions would you have asked?

**Technical**
- Argue why you chose the model you did, and why you did not choose something else?
- What would be a good baseline prediction? Why?
- Argue why you scored the model how you did.
- If you had had more time, what would you have done?


## Delivery
Please reply to the email, from which you got the link to this repo, with a link to a GitHub or
 GitLab repo containing your code, and please cc: `kahau@orsted.dk`.
