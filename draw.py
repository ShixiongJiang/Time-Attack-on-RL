import json
import os.path

import numpy as np
import matplotlib.pyplot as plt
import json

task_list = ['Goal', 'Push', 'Button', 'Race']
adv_method_list = ['Optimal Time Attack', 'FGSM Attack', 'Random']


fig, ax = plt.subplots()
fig.set_figheight(4.6)
fig.set_figwidth(12)
plt.yticks(fontsize=36)
plt.xticks(fontsize=36)
plt.xlabel('Perturbation Range', fontsize='36')
plt.ylabel('Time',fontsize='36')
plt.legend(fontsize=26)
epsilon_list = [ 0.01, 0.03, 0.05, 0.07, 0.10]

for task in task_list:
    color_list = ['purple', "red", "green","orange"]
    color_iterator = iter(color_list)
    for adv_method in adv_method_list:
        file_path = f'./figs/{task}{adv_method}-data.json'

        # Load the data from the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        avg_time_list = data['avg_time_list']
        avg_timd_std_list = data['avg_timd_std_list']
        color = next(color_iterator)
        plt.plot(epsilon_list, avg_time_list, color=color, label=f"{adv_method}")
        plt.fill_between(epsilon_list, np.array(avg_time_list) - np.array(avg_timd_std_list), np.array(avg_time_list) + np.array(avg_timd_std_list), color=color, alpha=0.2)

        # plt.xlabel("Perturbation range")
        # plt.ylabel("Y-axis")
        plt.title(f'{task}')
        plt.legend(fontsize=26)

        # plt.show()
        plt.savefig(f'./figs/{task}.pdf',bbox_inches='tight', dpi=500)
