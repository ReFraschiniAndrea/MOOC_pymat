from math import sin, cos, sqrt
from matplotlib import pyplot as plt

# Input parameters 
L1 = 1                # length of arm1 
L2 = 1.5              # length of arm2 
theta = [2.5, 2.7]    # initial angles 
tol = 0.0001          # tolerance 
alpha = 0.1           # learning rate 
Niter = 100           # max iterations

targets = [
    [0.75, -1],
    [1, 1],
    [2, 2.5]
]

# Compute tip robot position 
def robot_position(theta, L1, L2): 
    theta1, theta2 = theta 
    x = L1 * cos(theta1) + L2 * cos(theta2) 
    y = L1 * sin(theta1) + L2 * sin(theta2) 
    return [x, y]

# Evaluation of J 
def J(theta, xp, L1, L2): 
    [x, y] = robot_position(theta, L1, L2) 
    Jval = (x-xp[0])**2 + (y-xp[1])**2 
    return Jval

# Evaluation of Grad(J) 
def grad_J(theta, xp, L1, L2): 
    [x, y] = robot_position(theta, L1, L2)

    dx_dt1 = - L1 * sin(theta[0]) 
    dx_dt2 = - L2 * sin(theta[1]) 
    dy_dt1 =   L1 * cos(theta[0]) 
    dy_dt2 =   L2 * cos(theta[1]) 

    dJ_dx = 2*(x - xp[0]) 
    dJ_dy = 2*(y - xp[1]) 
            
    DJ_dt1 = dJ_dx*dx_dt1 + dJ_dy*dy_dt1 
    DJ_dt2 = dJ_dx*dx_dt2 + dJ_dy*dy_dt2 

    return [DJ_dt1, DJ_dt2]

plt.figure(figsize=(10, 5))
# Gradient Descent Method 
for xp in targets:
    distances = []
    i = 1 
    while i <= Niter: 
        # Gradient of the objective function 
        grad = grad_J(theta, xp, L1, L2) 
        # Update theta with gradient 
        theta[0] -= alpha * grad[0] 
        theta[1] -= alpha * grad[1] 
        # Compute the current distance 
        Jval = J(theta, xp, L1, L2) 
        distances.append(sqrt(Jval))
        # Check for convergence 
        if Jval < tol: 
            print(f"Converged after {i} iterations.") 
            break 
        i = i + 1

    plt.plot(distances, label = '(x_p, y_p) = '+str(xp), marker='o', markersize=3)

plt.xlabel("Iterations")
plt.ylabel("Distance")
plt.title("Robot tip to target distance")
plt.xlim((0, 100))
plt.ylim((0, 4))
plt.legend()
plt.grid()
plt.show()