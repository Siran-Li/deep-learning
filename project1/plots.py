import matplotlib.pyplot as plt
import numpy as np

"""
# plot for MLP activation function choices
x_labels = ['ReLU', 'Tanh', 'Sigmoid', 'SELU']
acc = [0.8069000244140625, 0.7776999473571777, 0.7784000039100647, 0.7620999217033386]
std = [0.0071561806835234165, 0.03965700790286064, 0.02044070139527321, 0.02994607202708721]

plt.bar(x_labels, acc, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=8)
plt.ylim(0.6, 0.9)
plt.ylabel('Accuracy')
plt.xlabel('Activation function')
plt.grid(axis='y')
plt.savefig('./project1/pics/activation_mlp.png')
plt.show()
"""

"""
# plot for ConvNet activation function choices
x_labels = ['ReLU', 'Tanh', 'Sigmoid', 'SELU']
acc = [0.8137999773025513, 0.7789000272750854, 0.6934000849723816, 0.7812999486923218]
std = [0.02263626642525196, 0.05178041383624077, 0.08957453817129135, 0.048108793795108795]

plt.bar(x_labels, acc, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.ylim(0.4, 0.9)
plt.ylabel('Accuracy')
plt.xlabel('Activation function')
plt.grid(axis='y')
plt.savefig('./project1/pics/activation_convnet.png')
plt.show()
"""

"""
# plot for MLP dropout rate
x_labels = ['0.0', '0.2', '0.5', '0.8']
acc = [0.8069000244140625, 0.7980999946594238, 0.8018000721931458, 0.7129999399185181]
std = [0.0071561806835234165, 0.006279588211327791, 0.006663333624601364, 0.029257476329803467]

plt.bar(x_labels, acc, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=4)
plt.ylim(0.6, 0.9)
plt.ylabel('Accuracy')
plt.xlabel('Dropout rate')
plt.grid(axis='y')
plt.savefig('./project1/pics/dropout.png')
plt.show()
"""

"""
# plot for ConvNet BatchNorm
x_labels = ['w/ BN', 'w/o BN']
acc = [0.804599940776825, 0.8137999773025513]
std = [0.04717155173420906, 0.02263626642525196]

plt.bar(x_labels, acc, width=0.3, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=4)
plt.ylim(0.6, 0.9)
plt.ylabel('Accuracy')
plt.grid(axis='y')
plt.savefig('./project1/pics/batchnorm.png')
plt.show()
"""


"""
# plot for weight sharing
x_labels = ['w/o weight share', 'w/ weight share']
acc_mlp = [0.8069000244140625, 0.8407999873161316]
std_mlp = [0.0071561806835234165, 0.006663334555923939]

acc_conv = [0.8137999773025513, 0.8751999735832214]
std_conv = [0.02263626642525196, 0.008456417359411716]

X_axis = np.arange(len(x_labels))

plt.bar(X_axis - 0.2, acc_mlp, width=0.4, yerr=std_mlp, align='center', alpha=0.5, ecolor='black', capsize=4)
plt.bar(X_axis + 0.2, acc_conv, width=0.4, yerr=std_conv, align='center', alpha=0.5, ecolor='black', capsize=4)

plt.xticks(X_axis, x_labels)
plt.ylim(0.6, 0.9)
plt.ylabel('Accuracy')
plt.grid(axis='y')
plt.legend(['MLP', 'ConvNet'])
plt.savefig('./project1/pics/weight_share_acc.png')
plt.show()
"""

# plot for auxiliary loss
x_labels = [0.5000, 0.6000, 0.7000, 0.8000, 0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000]
acc_conv  = [0.9013, 0.9046, 0.9078, 0.9142, 0.9069, 0.9091, 0.9092, 0.9062, 0.9091, 0.9102, 0.9069]
std_conv = [0.008756, 0.011374, 0.005633, 0.007554, 0.008306, 0.003872, 0.008108, 0.006828, 0.007310, 0.004709, 0.008293]

acc_mlp = [0.8758, 0.8784, 0.8817, 0.8794, 0.8840, 0.8795, 0.8756, 0.8807, 0.8821, 0.8807, 0.8812]
std_mlp = [0.003458, 0.007763, 0.004270, 0.007619, 0.004372, 0.006311, 0.005854, 0.008667, 0.005646, 0.008327, 0.011153]


plt.errorbar(x_labels, acc_mlp, yerr=std_mlp, alpha=0.5) #, align='center', alpha=0.5, ecolor='black', capsize=4)
plt.errorbar(x_labels, acc_conv, yerr=std_conv, alpha=0.5) #, align='center', alpha=0.5, ecolor='black', capsize=4)
plt.ylim(0.86, 0.93)
plt.ylabel('Accuracy')
plt.xlabel('$\lambda$')
plt.grid(axis='y')
plt.legend(['MLP', 'ConvNet'])
plt.savefig('./project1/pics/auxiliary_loss.png')
plt.show()