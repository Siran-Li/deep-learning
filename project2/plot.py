import matplotlib.pyplot as plt
import numpy as np
import json


"""
# plot the training loss curves
with open('./loss_relu.json') as f:
    loss_relu = json.load(f)
with open('./loss_tanh.json') as f:
    loss_tanh = json.load(f)
with open('./loss_selu.json') as f:
    loss_selu = json.load(f)

with open('./loss_relu_wo_sig.json') as f:
    loss_relu_wo_sig = json.load(f)
with open('./loss_tanh_wo_sig.json') as f:
    loss_tanh_wo_sig = json.load(f)
with open('./loss_selu_wo_sig.json') as f:
    loss_selu_wo_sig = json.load(f)

x = range(1, len(loss_relu)+1)
plt.plot(x, loss_relu, 'r-', label='ReLU+Sigmoid')
plt.plot(x, loss_tanh, 'g-', label='Tanh+Sigmoid')
plt.plot(x, loss_selu, 'b-', label='SELU+Sigmoid')

plt.plot(x, loss_relu_wo_sig, 'r--', label='ReLU')
plt.plot(x, loss_tanh_wo_sig, 'g--', label='Tanh')
plt.plot(x, loss_selu_wo_sig, 'b--', label='SELU')

plt.xlabel('epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.savefig("loss_curves_compare.png")
plt.show()
"""

# plot the accuracies
x_labels = ['ReLU', 'Tanh', 'SELU']
acc = [0.9864000082015991, 0.9736000061035156, 0.9754000008106232]
std = [0.0048826206661912374, 0.006053099815481821, 0.011637871602138906]

plt.bar(x_labels, acc, width=0.4, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=4)

plt.ylim(0.9, 1.0)
plt.ylabel('Accuracy')
plt.grid(axis='y')
plt.savefig('./test_acc.png')
plt.show()