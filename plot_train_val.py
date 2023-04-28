import matplotlib.pyplot as plt

train_loss, train_acc1 = [], []
valid_loss, valid_acc1= [], []

with open("searchs/fashionmnist/fashionmnist.log", "r") as f:
    for line in f:
        if "Train:" in line:
            line_parts = line.split("Prec@(1,5) ")
            if len(line_parts) == 2:
                loss_str, acc1_str= line_parts[0].split("Loss ")[1], line_parts[1].split()[0]
                acc1_str = acc1_str.replace('(', '').replace(',', '').replace('%', '')
                train_loss.append(float(loss_str))
                train_acc1.append(float(acc1_str[:-1]))
        elif "Valid:" in line:
            line_parts = line.split("Prec@(1,5) ")
            if len(line_parts) == 2:
                loss_str, acc1_str = line_parts[0].split("Loss ")[1], line_parts[1].split()[0]
                valid_loss.append(float(loss_str))
                acc1_str = acc1_str.replace('(', '').replace(',', '').replace('%', '')
                valid_acc1.append(float(acc1_str[:-1]))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label="Train")
plt.plot(valid_loss, label="Valid")
plt.title("Loss")
plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc1, label="Train Top-1")
plt.plot(valid_acc1, label="Valid Top-1")
plt.title("Accuracy")
plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()
plt.savefig('loss_accuracy.png')
plt.show()