import torch
from torchvision import datasets, transforms
from models.search_cells import SearchCNNController # assuming the model is defined in a file called model.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# load the test dataset
test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# get data with meta info
input_size, input_channels, n_classes, train_data = utils.get_data(config.dataset, config.data_path, cutout_length=0, validation=False)

# load the model
model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers, net_crit, device_ids=config.gpus)
model.load_state_dict(torch.load('best.pth.tar')['state_dict']) # assuming the model weights are stored in 'best.pth.tar'

# set the model to evaluation mode
model.eval()

# evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        # move images and labels to device
        images, labels = images.to(device), labels.to(device)

        # forward pass
        outputs = model(images)

        # get predicted classes
        _, predicted = torch.max(outputs.data, 1)

        # update accuracy stats
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# print accuracy
print('Test accuracy: %.2f %%' % (100 * correct / total))
