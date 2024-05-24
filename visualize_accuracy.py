import matplotlib.pyplot as plt


# Imagenette

# accuracy_values = [0.683, 0.841, 0.864, 0.963, 0.958, 0.977]
# epochs = ['7 real', '7 real\n + 3 SDXL-L', '7 real\n + 3 SDXL-L (x3)', '7 real\n + 3 OWLv2', '7 real\n + 3 (OWLv2, SDXL-L)', '10 real',]

# Tin
accuracy_values = [0.636, 0.641, 0.694, 0.693, 0.706]
epochs = ['180 real', '180 real\n + 20 SDXL-L', '180 real\n + 20 OWLv2', '180 real\n + 20 (OWLv2, SDXL-L)', '200 real']


# VOC
# accuracy_values = [0.596, 0.685, 0.715, 0.764, 0.794, 0.816]
# epochs = ['15 real', '15 real\n + 5 SDXL-L', '15 real\n + 5 SDXL', '15 real\n + 5 OWLv2', '15 real\n + 5 (OWLv2, SDXL)', '20 real',]


plt.figure(figsize=(10, 6))

# Calculate differences in accuracy
accuracy_diff = [accuracy_values[i] - accuracy_values[i-1] if i > 0 else 0 for i in range(len(accuracy_values))]

# Plotting
plt.plot(epochs, accuracy_values, marker='o', linestyle='-')

# Adding difference in accuracy above each point
for i, diff in enumerate(accuracy_diff):
    if diff != 0:
      plt.text(epochs[i], accuracy_values[i] + 0.004, f'{diff:+.2f}', ha='center', va='bottom', fontsize=10)

# Adding labels and title
plt.xlabel('Dataset', fontsize=14)
#plt.ylabel('mAP-50', fontsize=14)
#plt.title('PASCAL VOC mAP-50 over different training datasets', fontsize=16)
plt.title('Tiny ImageNet accuracy over different training datasets', fontsize=16)
plt.ylabel('Accuracy', fontsize=14)
#plt.title('Imagenette accuracy over different training datasets', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


# Displaying the plot
plt.grid(True)
plt.tight_layout()
#plt.savefig('imagenette_accuracy.png')
plt.savefig('tin_accuracy.png')
#plt.savefig('voc_accuracy.png')
