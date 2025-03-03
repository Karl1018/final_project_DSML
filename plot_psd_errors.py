import matplotlib.pyplot as plt

error_list = [0.4771, 0.4636, 0.8083, 0.3627, 0.2744, 0.2962, 0.2805, 0.2643, 0.2761, 0.3092, 0.2862, 0.3213, 0.2713, 0.2636, 0.2648, 0.2868, 0.2589, 0.3124, 0.2963, 0.3006]
plt.plot(error_list, marker='o')
plt.title("Power Spectrum Error Over Training Rounds")
plt.xticks(range(len(error_list)), range(1, len(error_list)+1)) # Set x-ticks
plt.xlabel("Training Rounds")
plt.ylabel("Error")

plt.savefig("bonus/psd_errors.png", dpi=300)
plt.show()