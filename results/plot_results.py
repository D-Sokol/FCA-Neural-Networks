#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


datasets = 'breast_cancer', 'mammographic_masses', 'seismic_bumps', 'zoo'
filename_template = "{method}-{dataset}.txt"

if __name__ == '__main__':
    for dataset in datasets:
        plt.figure(figsize=(12,7))

        plt.title('Accuracy of different network architectures')
        plt.xlabel('Neurons')
        plt.xlim(1, 21)
        plt.xticks(range(1, 21))
        plt.ylabel('Accuracy, %')
        plt.ylim(-5, 105)
        plt.yticks(range(0, 110, 10))
        plt.grid()

        data = pd.read_csv(filename_template.format(method='full-1hl', dataset=dataset))
        averaged = data.groupby('neurons').mean()
        plt.plot(averaged.index, averaged['accuracy'], 'r--.', label='Fully connected (1 hidden layer)')

        data = pd.read_csv(filename_template.format(method='full-2hl', dataset=dataset))
        averaged = data.groupby('neurons').mean()
        plt.plot(averaged.index, averaged['accuracy'], 'b--.', label='Fully connected (2 hidden layers)')
        # plt.legend()

        ax = plt.gca().twiny()
        data = pd.read_csv(filename_template.format(method='full-min_supp', dataset=dataset))
        for max_level, subset in data.groupby('max_level'):
            averaged = subset.groupby('min_supp').mean()
            ax.plot(averaged.index, averaged['accuracy'], marker='*', label=f'Min supp (lvl={max_level})')
        else:
            ax.set_xlabel('Minimal concept support')
            ax.set_xlim(0.0, 1.0)
            ax.set_xticks(np.arange(0.0, 1.1, 0.1))
            # ax.legend()
            del ax

        plt.gcf().legend()
        plt.savefig(f'{dataset}-accuracy.pdf')
        plt.clf()