"""
Helper function for training scripts
"""

# Author: Joel Ong

import matplotlib.pyplot as plt

def plot_loss_per_generation(loss_vec):
    plt.plot(loss_vec, 'k-')
    plt.title('L2 Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('L2 Loss')
    plt.show()
    
def save_model(saver, sess, file_name):
    saver.save(sess, file_name)
    print('Trained Model saved as ' + file_name)  