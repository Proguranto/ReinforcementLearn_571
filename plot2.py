import matplotlib.pyplot as plt
import numpy as np

def normalize(arr, t_min=0, t_max=1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

# 2
epochs = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
loss = [0.00629314, 0.00006553, 0.00006078, 0.00018171, 0.00006441, 0.00007861, 0.00006930, 0.00008106, 0.00007013, 0.00001603]

fig2 = plt.figure()
plt.plot(epochs, loss, color="darkviolet")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("2.3, #2")
plt.savefig("plot/2_2.png")
plt.show()
# Success rate:  0.26
# Average reward (success only):  -5.412916317268375
# Average reward (all):  -9.528158672717584


#
batch_sizes = [5, 10, 25, 50]
loss = [0.00009932, 0.00001711, 0.00013880,  0.00000339]
success_rate = [0.3, 0.38, 0.25, 0.15]
fig3 = plt.figure()
plt.plot(batch_sizes, normalize(loss), label="log loss", color="darkviolet")
plt.plot(batch_sizes, success_rate, label="success rate",  color="red")
plt.xlabel("batch size")
plt.title("2.3, #3, modified batch sizes")
plt.legend()
plt.savefig("plot/2_3.png")
plt.show()
