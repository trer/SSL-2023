from mnist import MNIST



mndata = MNIST('./MNIST')
mndata.gz = True
images, labels = mndata.load_training()

print(len(images[0])//28)

print("done")