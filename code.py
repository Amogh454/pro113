#importing...
from re import T
from time import daylight
import plotly.express as px
import pandas as pd
import statistics as st
import csv
import plotly.graph_objects as pg
import numpy as np
import plotly.figure_factory as ff
import seaborn as sb
import random as rd

#Read the data
with open('data.csv',newline='') as f:
    r = csv.reader(f)
    savingsData = list(r)


savingsData.pop(0)    


tEntries = len(savingsData)
tReminder = 0

for i in savingsData:
    if int(i[3]) == 1:
        tReminder = tReminder+1

data = pd.read_csv('data.csv')

fig = px.scatter(data, y= "quant_saved" ,color="rem_any")
fig.show()

graph = pg.Figure(pg.Bar(x=['reminded', 'notReminded'], y=[tReminder,(tEntries-tReminder)]))

graph.show()

allSavings = []
for i in savingsData:
    allSavings.append(float(i[0]))
mean = st.mean(allSavings)
median = st.median(allSavings)
mode = st.mode(allSavings)
ST = st.stdev(allSavings)
print(mean)
print(median)
print(mode)
print(ST)

R = []
r = []
for i in savingsData:
    if int(i[3]) == 1:
        R.append(float(i[0]))
    else:
        r.append(float(i[0]))    

Rmean = st.mean(R)
Rmode = st.mode(R)
Rmedian = st.median(R)
rmean = st.mean(r)
rmedian = st.median(r)
rmode = st.mode(r)
RsT = st.stdev(R)
rSt = st.stdev(r)
print(Rmean)
print(Rmode)
print(Rmedian)
print(RsT)
print(rmean)
print(rmedian)
print(rmode)
print(rSt)


age=[]
savings=[]

for i in savingsData:
    if float(i[5])!=0:
        age.append(float(i[5]))
        savings.append(float(i[0]))

C = np.corrcoef(age,savings) 
print(C[0,1])

fig = ff.create_distplot([data['quant_saved'].tolist()],['savings'], show_hist=False)
fig.show()

#sb.boxplot(d= data, x= data['quant_saved'])
q1 = data['quant_saved'].quantile(0.25)
q3 = data['quant_saved'].quantile(0.75)

iqr = q3-q1
print(iqr)


lower=q1-1.5*iqr
upper = q3+1.5*iqr
print('lower =',lower)
print('upper = ',upper)

newData = data[data['quant_saved']<upper]
allSavings = newData['quant_saved'].tolist()
mean = st.mean(allSavings)
median = st.mean(allSavings)
mode = st.mode(allSavings)
ST = st.stdev(allSavings)
print('New Data Mean = ', mean)
print('New Data Median = ', median)
print('New Data Mode = ', mode)
print('New Data Standard Deviation = ', ST)

SmList = []
for i in range(1000):
    templist = []
    for j in range(100):
        templist.append(rd.choice(allSavings))
        SmList.append(st.mean(templist))

mSampling = st.mean(SmList)
fig = ff.create_distplot([SmList], ['savings'], show_hist = False)
fig.add_trace(pg.Scatter(x= [mSampling, mSampling], y=[0, 0.1,], mode='lines', name='MEAN'))
fig.show()


