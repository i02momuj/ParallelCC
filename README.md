# ParallelCC

Classifier Chains (CC) [] is one of the best-performing methods in multi-label classification.
CC is based on the idea of creating a binary classifier for each of the labels but linked in such way that each binary classifier includes the predictions of previous labels in the chain as extra input features.
Therefore, CC is, by its original design, not parallelizable, since the binary classifier needs the outputs of previous ones to be built.

In this project, we slightly modified the design of CC in order to make it parallelizable.

_More information will be provided soon_.