import matplotlib.pyplot as plt
import numpy as np
import statistics



def plotting_func(y,title):
    #plt.figure(figsize=(12,7))
    plt.hist(y, bins = 20)
    #plt.bar(y, label='label', c="grey", ls=':', marker='s')
    # plt.axhline(np.mean(y),color='r', label = 'mean', linestyle = '--')
    # plt.axhline(np.mean(y)+ statistics.stdev(y),linestyle = 'dotted', label = 'standard deviation')
    # plt.axhline(np.mean(y)- statistics.stdev(y),linestyle = 'dotted')
    # plt.axhline(np.mean(y)- 2* statistics.stdev(y),linestyle = 'dotted')
    plt.title(title)
    #plt.xlabel("Student")
    plt.ylabel('count')
    plt.legend()
    plt.show()


def hist_plot(list1,list2,title):
    # Plotting the histograms
    biggest = max([max(list2), max(list1)])
    print(biggest)
    plt.hist([list1,list2], bins=range(min(list1), biggest + 2), color = ['orange', 'blue'], alpha=0.5, label=['Human','LLM'])
    #plt.hist(list2, bins=range(min(list2), max(list2) + 1), alpha=0.5, label='List 2')

    # Adding labels and title
    plt.xlabel('Error Number')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()

    # Showing the plot
    plt.show()
