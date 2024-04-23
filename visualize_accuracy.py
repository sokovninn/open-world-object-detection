import matplotlib.pyplot as plt


models = ['10 Real', '10 Real + 3 Synth', '10 Real + 3 Synth (x3)', '7 Real + 3 Real (CLIP)']
accuracies = [97.7, 84.1, 86.4, 96.8]

plt.figure(figsize=(10, 8))
plt.bar(models, accuracies)
plt.xlabel('Datasets')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Accuracy Between Different Datasets')
#plt.ylim(0, 100)

for i in range(len(models)):
    plt.text(i, accuracies[i] + 1, str(accuracies[i]), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('accuracy_comparison.png')