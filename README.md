# Keras Learning Rate Range Test
Plots the change of the loss function according to a varying learning rate. Allows us to find a good range of learning rates for the model. The learning rate can be linear or exponential (preferred).

## Usage
Create and compile a Keras model, then execute this code:

```python
BATCH_SIZE=512
EPOCHS=5

# Learning Rate Finder callback
lr_finder = LRFinder(min_lr=1e-4, max_lr=1e1, steps_per_epoch=int(len(x_train)/BATCH_SIZE), epochs=EPOCHS)

# Train model with batch size 512 for 5 epochs
# Include Learning Rate Finder as callback
hist = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[lr_finder])
```

## Observe results
```python
# Plot the learning rate against iteration & loss
lr_finder.plot_lr()
```
![Learning Rate](/img/learning_rate_loss_iteration.png)

```python
# Plot loss against learning rate (with derivative of loss)
lr_finder.plot(monitor='loss')
```
![Loss](/img/loss_vs_learning_rate.png)

```python
# Plot accuracy against learning rate (with derivative of accuracy)
lr_finder.plot(monitor='acc')
```
![Accuracy](/img/accuracy_vs_learning_rate.png)


## Contributions
Contributions are welcome. Please, file issues and submit pull requests on GitHub, or contact me directly.

## References
This code is based on:
- The method described in section 3.3 of the 2015 paper ["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/abs/1506.01186) by Leslie N. Smith
- The implementation of the algorithm in [fastai library](https://github.com/fastai/fastai) by Jeremy Howard. See [fast.ai deep learning course](http://course.fast.ai/) for details.
- ["Estimating an Optimal Learning Rate For a Deep Neural Network"](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0) by Pavel Surmenok. Implementation code available [here](https://github.com/surmenok/keras_lr_finder).
- ["CLR"](https://github.com/bckenstler/CLR) repository by Brad Kenstler
- ["Learning-Rate"](https://github.com/nathanhubens/Learning-Rate) repository by Nathan Hubens
- [fastaiv2keras](https://github.com/metachi/fastaiv2keras) repository
