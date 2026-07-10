# 1. data loader
train_data = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor())

test_data = datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transforms.ToTensor())

# take a glimpse:
image, label = train_data[0]
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Label: {label}')
plt.show()

# shuffle + batch
train_loader = DataLoader(train_data, batch_size=32, shuffle=True) # returns a DataLoader object
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# take a glimpse:
images, labels = next(iter(train_loader)) 
image = images[0]
label = labels[0]

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Label: {label}')
plt.show()
