# Fashion-MNIST Image Classifier (CNN from Scratch)
 
## What this is
 
I wanted to build and train a convolutional neural network myself, so I picked something small enough to do in a day: classifying clothing images from the Fashion-MNIST dataset into 10 categories (shirts, sandals, bags, etc.) using a CNN I wrote in PyTorch.
 
The model hit 90.56% accuracy on data it had never seen, and the mistakes it made mostly happened between categories that look alike.
 
## The data
 
Fashion-MNIST is 70,000 grayscale images (60,000 train, 10,000 test), each 28x28 pixels, split across 10 clothing categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot. It's built to be a slightly harder drop-in replacement for the classic MNIST digit dataset, same size and shape, but images that are actually a bit trickier to tell apart.
 
## Building the model
 
I wrote a small CNN with two convolutional layers (16 filters, then 32 filters, both 3x3), each followed by a max-pool step to shrink the image down, then two fully connected layers narrowing down to 10 outputs, one per clothing category. ReLU activations after each layer keep the network from just collapsing into one big linear equation.
 
I traced the shape by hand through each layer to make sure the numbers lined up: starting at 28x28, after two rounds of pooling the image shrinks to 7x7, with 32 channels by that point, so the flattened input into the first fully connected layer is 32 x 7 x 7 = 1568.
 
## Training
 
I used cross-entropy loss and the Adam optimizer, trained for 5 epochs. Loss dropped steadily each epoch:
 
| Epoch | Loss   |
|-------|--------|
| 1     | 0.5070 |
| 2     | 0.3284 |
| 3     | 0.2786 |
| 4     | 0.2477 |
| 5     | 0.2279 |
 
## Results
 
Test accuracy came out to 90.56% on the held-out test set.

I also built a confusion matrix to see exactly where it was getting things wrong, instead of just trusting the one accuracy number. The biggest source of error was Shirt getting confused with T-shirt/top, Pullover, Coat, and Dress, and to a lesser extent Pullover and Coat getting mixed up with each other. Meanwhile categories with distinct shapes, Trouser, Sandal, Sneaker, Bag, Ankle boot, were classified almost perfectly (972-990 out of 1000).
 
That pattern makes sense. T-shirts, shirts, pullovers, coats, and dresses all look kinda similar, especially at 28x28 resolution. A lot of the detail that would distinguish them (collar shape, sleeve length, texture) isn't there at that size. The model's mistakes line up with where a person would also have a harder time, which is a good sign it's picking up on visual structure rather than memorizing noise.
 
## Tools
 
Python, PyTorch, torchvision, scikit-learn (confusion matrix), matplotlib