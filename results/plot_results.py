#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


datasets = 'breast_cancer', 'mammographic_masses', 'seismic_bumps', 'zoo', 'car_evaluation', 'titanic'
filename_template = "{method}-{dataset}.txt"

if __name__ == '__main__':
    for dataset in datasets:
        plt.figure(figsize=(12,7))

        # plt.title('Accuracy for dataset "{}"'.format(dataset))
        plt.xlabel('Количество нейронов')
        plt.ylabel('Точность, %')
        plt.ylim(-5, 105)
        plt.yticks(range(0, 110, 10))
        plt.grid()

        data = pd.read_csv(filename_template.format(method='full-1hl', dataset=dataset))
        averaged = data.groupby('neurons').mean()
        plt.plot(averaged.index, averaged['accuracy'], 'r--.', label='Dense (1 hidden layer)')

        data = pd.read_csv(filename_template.format(method='full-2hl', dataset=dataset))
        averaged = data.groupby('neurons').mean()
        plt.plot(averaged.index, averaged['accuracy'], 'b--.', label='Dense (2 hidden layers)')
        # plt.legend()

        # ax = plt.gca().twiny()
        ax = plt.gca()

        data = pd.read_csv(filename_template.format(method='full-min_supp', dataset=dataset))
        for max_level, subset in data.groupby('max_level'):
            averaged = subset.groupby('neurons').mean()
            ax.plot(averaged.index, averaged['accuracy'], marker='*', label=f'Supp; $l_{{\\mathrm{{max}}}}={max_level}$')
        data = pd.read_csv(filename_template.format(method='full-min_cv', dataset=dataset))
        for max_level, subset in data.groupby('max_level'):
            averaged = subset.groupby('neurons').mean()
            ax.plot(averaged.index, averaged['accuracy'], marker='*', label=f'CV; $l_{{\\mathrm{{max}}}}={max_level}$')
        data = pd.read_csv(filename_template.format(method='full-min_cfc', dataset=dataset))
        for max_level, subset in data.groupby('max_level'):
            averaged = subset.groupby('neurons').mean()
            ax.plot(averaged.index, averaged['accuracy'], marker='*', label=f'CFC; $l_{{\\mathrm{{max}}}}={max_level}$')
        # ax.set_xlabel('Threshold for $\\mathcal{M}$-metric')
        # ax.set_xlim(0.0, 1.0)
        # ax.set_xticks(np.arange(0.0, 1.1, 0.1))

        plt.gcf().legend()
        plt.savefig(f'{dataset}-accuracy.png')
        plt.clf()
