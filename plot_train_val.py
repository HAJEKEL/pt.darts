import matplotlib.pyplot as plt

train_loss, train_acc1, train_acc5 = [], [], []
valid_loss, valid_acc1, valid_acc5 = [], [], []

with open("searchs/fashionmnist/fashionmnist.log", "r") as f:
    for line in f:
        if "Train:" in line:
            line_parts = line.split("Prec@(1,5) ")
            if len(line_parts) == 2:
                loss_str, acc1_str, acc5_str = line_parts[0].split("Loss ")[1], line_parts[1].split()[0], line_parts[1].split()[1]
                acc1_str = acc1_str.replace('(', '').replace(',', '').replace('%', '')
                acc5_str = acc5_str.replace(')', '').replace('%', '')
                train_loss.append(float(loss_str))
                train_acc1.append(float(acc1_str[:-1]))
                train_acc5.append(float(acc5_str[:-1]))
        elif "Valid:" in line:
            line_parts = line.split("Prec@(1,5) ")
            if len(line_parts) == 2:
                loss_str, acc1_str, acc5_str = line_parts[0].split("Loss ")[1], line_parts[1].split()[0], line_parts[1].split()[1]
                valid_loss.append(float(loss_str))
                acc1_str = acc1_str.replace('(', '').replace(',', '').replace('%', '')
                acc5_str = acc5_str.replace(')', '').replace('%', '')
                valid_acc1.append(float(acc1_str[:-1]))
                valid_acc5.append(float(acc5_str[:-1]))

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
plt.plot(train_acc5, label="Train Top-5")
plt.plot(valid_acc5, label="Valid Top-5")
plt.title("Accuracy")
plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()

plt.show()

plt.savefig('loss_accuracy.png')
