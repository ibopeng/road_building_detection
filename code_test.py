import matplotlib.pyplot as plt
import numpy as np

test_acc = np.loadtxt('./test_prediction/testset_accuracy.txt')

plt.plot(test_acc)
plt.show()
