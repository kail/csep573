import numpy as np
import matplotlib.pyplot as plt
import csv


infiles = [
    ('/Users/mskobov/repos/csep573/reinforcement_mdp/rtdp_avg_reward_all_states.csv','b'),
    #('/Users/mskobov/repos/csep573/reinforcement_mdp/rtdp_avg_reward_to_goal.csv','g'),
    ('/Users/mskobov/repos/csep573/reinforcement_mdp/stack_rtdp_avg_reward_all_states.csv','r'),
    #('/Users/mskobov/repos/csep573/reinforcement_mdp/stack_rtdp_avg_reward_to_goal.csv','c'),
    ('/Users/mskobov/repos/csep573/reinforcement_mdp/vi_avg_reward_all_states.csv','m'),
    #('/Users/mskobov/repos/csep573/reinforcement_mdp/vi_avg_reward_to_goal.csv','k'),
]
for config in infiles:
    with open(config[0], "r") as f:
        data = [row for row in csv.reader(f)]
        xd = [float(row[0]) for row in data]
        yd = [float(row[1]) for row in data]
    
    # sort the data
    reorder = sorted(range(len(xd)), key = lambda ii: xd[ii])
    xd = [xd[ii] for ii in reorder]
    yd = [yd[ii] for ii in reorder]
    
    # make the scatter plot
    #plt.scatter(xd, yd, s=30, alpha=0.15, marker='o')
    
    # determine best fit line
    par = np.polyfit(xd, yd, 1, full=True)
    
    slope=par[0][0]
    intercept=par[0][1]
    xl = [min(xd), max(xd)]
    yl = [slope*xx + intercept  for xx in xl]
    
    # coefficient of determination, plot text
    #variance = np.var(yd)
    #residuals = np.var([(slope*xx + intercept - yy)  for xx,yy in zip(xd,yd)])
    #Rsqr = np.round(1-residuals/variance, decimals=2)
    #plt.text(.9*max(xd)+.1*min(xd),.9*max(yd)+.1*min(yd),'$R^2 = %0.2f$'% Rsqr, fontsize=30)
    
    plt.xlabel("Time (ms)")
    plt.ylabel("Reward")
    
    # error bounds
    yerr = [abs(slope*xx + intercept - yy)  for xx,yy in zip(xd,yd)]
    par = np.polyfit(xd, yerr, 2, full=True)
    
    yerrUpper = [(xx*slope+intercept)+(par[0][0]*xx**2 + par[0][1]*xx + par[0][2]) for xx,yy in zip(xd,yd)]
    yerrLower = [(xx*slope+intercept)-(par[0][0]*xx**2 + par[0][1]*xx + par[0][2]) for xx,yy in zip(xd,yd)]
    
    plt.plot(xd, yd, config[1]+'-')

plt.show()