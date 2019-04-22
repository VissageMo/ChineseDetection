def adjust_learning_rate(init_learning_rate, decay_point, decay_par=0.1, accuracy=None, epoch=None):
    """Change learning_rate in training

    Accuracy mode:
        if decay_point[list], decay learning rate when accuracy is larger than
        each point in decay_point, need param 'accuracy'
    Epoch mode:
        if decay_point[int], decay learning rate when epoch increace
        another decay_point, need param 'epoch'
    """
    if isinstance(decay_point, list):
        learning_rate = init_learning_rate
        for i in range(len(decay_point)):
            if accuracy > decay_point[i]:
                learning_rate = learning_rate * decay_par
    elif isinstance(decay_point, int):
        learning_rate = init_learning_rate * (0.1 ** (epoch // decay_point))
    return learning_rate