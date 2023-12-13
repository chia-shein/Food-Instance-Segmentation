from matplotlib import pyplot as plt
def imshow(img):
    plt.figure(figsize=(16,9))
    plt.imshow(img)
    plt.axis('off')
    plt.show()