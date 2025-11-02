import matplotlib.pyplot as plt
from math import sin, cos

def plot_robot_arm(theta, xp, L):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Robotic Arm Configuration')
    ax.grid(True)
    ax.set_axisbelow(True)

    joint = (L[0] * cos(theta[0]), L[0] * sin(theta[0]))
    hand = ( L[0] * cos(theta[0]) + L[1] * cos(theta[1]), L[0] * sin(theta[0]) + L[1] * sin(theta[1]))
    
    ax.scatter(xp[0], xp[1], c='purple', marker='*', s=250,  zorder=1)
    ax.plot((0, joint[0]), (0, joint[1]), color='blue', lw=3, zorder=2)
    ax.plot((joint[0], hand[0]), (joint[1], hand[1]),  color='blue', lw=3, zorder=2)
    ax.scatter((0, joint[0]), (0, joint[1]), c='black', marker='s', s=100,  zorder=3)
    ax.scatter(hand[0], hand[1], c='green', marker='o', s=100,  zorder=4)

    plt.show()