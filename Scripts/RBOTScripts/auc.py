import yaml
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np

test_results_dir_name = \
    '/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/results'
test_results_dir = os.fsencode(test_results_dir_name)
tracking_errors_file = os.fsencode('tracking_errors.yml')
plot_file = os.fsencode('auc.png')
for directory in os.listdir(test_results_dir):
    directory_full = os.path.join(test_results_dir, directory)
    if os.path.isdir(directory_full):
        aucs = []
        for test_case_directory in os.listdir(directory_full):
            test_case_directory_full = os.path.join(directory_full, test_case_directory)
            if os.path.isdir(test_case_directory_full):
                tracking_errors_file_full = os.path.join(test_case_directory_full, tracking_errors_file)
                tracking_errors_file_name = os.fsdecode(tracking_errors_file_full)
                with open(tracking_errors_file_name, 'r') as stream:
                    data = yaml.safe_load(stream)
                frames = data['frames'][1:]
                errors = [frame['error'] for frame in frames]
                errors.sort()
                x = [0]
                y = [0]
                n = len(errors)
                test_case_directory_name = os.fsdecode(test_case_directory)
                if test_case_directory_name[:2] == 'ch':
                    pass
                if test_case_directory_name == 'ch_fm_l':
                    print('Here')
                for i in range(len(errors)):
                    if errors[i] > 0.2 or i == len(errors) - 1:
                        x.append(0.2)
                        y.append(100 * i / n)
                        reached02 = True
                        break
                    x.append(errors[i])
                    y.append(100 * i / n)
                    y.append(100 * (i + 1) / n)
                    x.append(errors[i])
                auc = metrics.auc(x, y)
                aucs.append(auc)
                plt.plot(x, y)
                plot_file_full = os.path.join(test_case_directory_full, plot_file)
                plot_file_name = os.fsdecode(plot_file_full)
                plt.savefig(plot_file_name)
                plt.clf()
        directory_name = os.fsdecode(directory)
        print(directory_name + ":", sum(aucs) / len(aucs))
