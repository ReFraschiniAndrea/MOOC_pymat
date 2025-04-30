from manim import *
import os
# add the path to the latex binaries
env = os.environ
env["PATH"] = r"C:\Users\rfand\AppData\Local\Programs\MiKTeX\miktex\bin\x64;" + env["PATH"]

config.background_color=WHITE
config.renderer="cairo"
# test resolution
# config.pixel_width = 960 
# config.pixel_height = 720 
# release resolution
# config.pixel_width = 1440 
# config.pixel_height = 1080 


def double_arm_kinematics(l1:float, l2:float, theta1:float, theta2:float, alpha=1, center=[0,0,0]):
    return [
        center[0] + l1 * np.cos(theta1) + alpha * l2 * np.cos(theta2),
        center[1] + l1 * np.sin(theta1) + alpha * l2 * np.sin(theta2),
        0
    ]

class DoubleArm():
    def __init__(self, l1=1, l2=1, theta1=0, theta2=0, center=[0,0,0]):
        # internal state
        self.center = center
        self.l1 = l1
        self.l2 = l2
        self.theta1 = theta1
        self.theta2 = theta2
        # create the 2 arms and the jpints
        self.origin = Dot(center, color=GRAY).set_z_index(1)
        self.joint = Dot(center + RIGHT*l1, color=GRAY).set_z_index(1)
        self.hand =  Dot(center + RIGHT*(l1+l2), color=GREEN).set_z_index(1)
        self.arm1 = Line(center, center + RIGHT*l1, stroke_color=BLUE)
        self.arm2 = Line(center+ RIGHT*l1, center + RIGHT*(l1+l2), stroke_color=BLUE)
        self.ARM1 = VGroup(self.arm1, self.origin).rotate(theta1, about_point=center)
        self.ARM2 = VGroup(self.arm2, self.joint, self.hand).move_to(double_arm_kinematics(l1,l2,theta1, theta2, alpha=0.5, center=center)).rotate(theta2)

    def animate_by_angle(self, scene, theta1, theta2, **kwargs):
        self.ARM2.save_state()
        def second_arm_anim(arm, t):
            arm.restore()
            arm.become(
                arm.copy().move_to(double_arm_kinematics(self.l1, self.l2, theta1*t+self.theta1, theta2*t+self.theta2, alpha=0.5, center=self.center)).rotate(theta2*t)
                )
        scene.play(
            Rotate(self.ARM1, theta1, about_point=self.center, **kwargs), 
            UpdateFromAlphaFunc(self.ARM2, second_arm_anim, **kwargs),
            )
        self.theta1 += theta1
        self.theta2 += theta2

    def animate_to_angle(self, scene, theta1, theta2, **kwargs):
        # if theta1-self.theta1 > 2*PI:
        self.animate_by_angle(scene, theta1-self.theta1, theta2-self.theta2, **kwargs)


class NewDB():
    def __init__(self,axes: Axes, l1, l2, theta1=0, theta2=0, color=BLUE, z_index=0):
        # internal attributer
        self.ax = axes
        self.l1 = l1
        self.l2 = l2
        self.theta1 = ValueTracker(theta1)
        self.theta2 = ValueTracker(theta2)
        # create the 2 arms and the joints
        self.foot = Dot(self.center(), color=GRAY).set_z_index(z_index+1)
        self.joint = Dot(self.ax.c2p(l1, 0, 0), color=GRAY).set_z_index(z_index+1)
        self.hand =  Dot(self.ax.c2p(l1+l2, 0, 0), color=GRAY).set_z_index(z_index+1)
        self.arm1 = Line(self.center(), self.ax.c2p(l1, 0, 0), stroke_color=color).set_z_index(z_index)
        self.arm2 = Line(self.ax.c2p(l1, 0, 0), self.ax.c2p(l1+l2, 0, 0), stroke_color=color).set_z_index(z_index)
        self.ARM1 = VGroup(self.arm1, self.foot).rotate(theta1, about_point=self.center())
        self.ARM2 = VGroup(self.arm2, self.joint, self.hand).move_to(
            0.5*(self._hand_coord(theta1, theta2) + self._joint_coord(theta1))
        ).rotate(theta2)
        # updaters
        self.foot.add_updater(
            lambda mobj: mobj.move_to(self.center())
        )
        self.joint.add_updater(
            lambda mobj: mobj.move_to(self._joint_coord(self.theta1.get_value()))
        )
        self.hand.add_updater(
            lambda mobj: mobj.move_to(self._hand_coord(self.theta1.get_value(), self.theta2.get_value()))
        )
        self.arm1.add_updater(
            lambda mobj: mobj.put_start_and_end_on(self.center(),self._joint_coord(self.theta1.get_value()))
            )
        self.arm2.add_updater(
            lambda mobj: mobj.put_start_and_end_on(
                self._joint_coord(self.theta1.get_value()),
                self._hand_coord(self.theta1.get_value(), self.theta2.get_value())
            ))
        
        self.add(axes)
        self.add(self.foot, self.arm1, self.joint, self.arm2, self.hand)
        

    def center(self):
        return self.ax.c2p(0,0,0)
    
    def _joint_coord(self, t1, t2=None):
        return self.ax.c2p(
            self.l1*np.cos(t1), self.l1*np.sin(t1), 0
            )
    
    def _hand_coord(self, t1, t2):
        return self.ax.c2p(
            self.l1*np.cos(t1) + self.l2*np.cos(t2),
            self.l1*np.sin(t1) + self.l2*np.sin(t2),
              0)
    
    def MoveToAngles(self, t1, t2, **kwargs):
        return AnimationGroup(
            self.theta1.animate(**kwargs).set_value(t1),
            self.theta2.animate(**kwargs).set_value(t2)
        )

    def MoveByAngles(self, t1, t2, **kwargs):
        current_t1 = self.theta1.get_value()
        current_t2 = self.theta2.get_value()
        return self.MoveToAngles(t1 + current_t1, t2+current_t2, **kwargs)



from math import cos, sin, sqrt

# Definition of robotic arm variables and function
l1 = 1.3
l2 = 1.8
def GD(
    starting_point,
    target,
    l1, l2,
    max_iter = 1000,
    learning_rate = 0.01,
    tol = 1e-2
    ):

    def current_distance(angles, theta):
        return np.sqrt(objective_func_2(theta[0], theta[1], target))

    def objective_func_2(t1, t2, target):
        return np.square(l1*np.cos(t1) +l2*np.cos(t2)- target[0]) + np.square(l1*np.sin(t1) +l2*np.sin(t2)- target[1])

    def gradient_x_robot(theta):
        dxr_dx = - l1 * sin(theta[0])
        dxr_dy = - l2 * sin(theta[1])
        return [dxr_dx, dxr_dy]

    def gradient_y_robot(theta):
        dyr_dx = l1 * cos(theta[0])
        dyr_dy = l2 * cos(theta[1])
        return [dyr_dx, dyr_dy]

    def gradient_obj_function(theta, target):
        x_robot, y_robot = double_arm_kinematics(l1, l2, theta[0], theta[1])[:2]

        x_comp = 2 * ((x_robot - target[0]) * gradient_x_robot(theta)[0] + (y_robot - target[1]) * gradient_y_robot(theta)[0])
        y_comp = 2 * ((x_robot - target[0]) * gradient_x_robot(theta)[1] + (y_robot - target[1]) * gradient_y_robot(theta)[1])

        return [x_comp, y_comp]

    result = np.zeros((max_iter+1, 3))
    result[0] = [starting_point[0], starting_point[1], objective_func_2(starting_point[0], starting_point[1], target)]
    angles = starting_point.copy()
    for i in range(max_iter):
        grad = gradient_obj_function(angles, target)

        # Update theta with gradient
        angles[0] -= learning_rate * grad[0]
        angles[1] -= learning_rate * grad[1]

        # Check for convergence
        distance = current_distance(angles, target)

        result[i+1] = [angles[0], angles[1], objective_func_2(angles[0], angles[1], target)]

        if distance < tol:
            # print(f"Converged after {i} iterations.")
            break

    result = np.resize(result, (i, 3))
    return result


def GD_D(
    starting_point,
    target,
    l1, l2,
    max_iter = 1000,
    learning_rate = 0.01,
    tol = 1e-2
    ):

    def current_distance(angles, theta):
        return np.sqrt(objective_func_2(theta[0], theta[1], target))

    def objective_func_2(t1, t2, target):
        return np.square(l1*np.cos(t1) +l2*np.cos(t2)- target[0]) + np.square(l1*np.sin(t1) +l2*np.sin(t2)- target[1])

    def gradient_x_robot(theta):
        dxr_dx = - l1 * sin(theta[0])
        dxr_dy = - l2 * sin(theta[1])
        return [dxr_dx, dxr_dy]

    def gradient_y_robot(theta):
        dyr_dx = l1 * cos(theta[0])
        dyr_dy = l2 * cos(theta[1])
        return [dyr_dx, dyr_dy]

    def gradient_obj_function(theta, target):
        x_robot, y_robot = double_arm_kinematics(l1, l2, theta[0], theta[1])[:2]

        x_comp = 2 * ((x_robot - target[0]) * gradient_x_robot(theta)[0] + (y_robot - target[1]) * gradient_y_robot(theta)[0])
        y_comp = 2 * ((x_robot - target[0]) * gradient_x_robot(theta)[1] + (y_robot - target[1]) * gradient_y_robot(theta)[1])

        return [x_comp, y_comp]

    result = np.zeros((max_iter+1, 3))
    result[0] = [starting_point[0], starting_point[1], objective_func_2(starting_point[0], starting_point[1], target)]
    angles = starting_point.copy()
    for i in range(max_iter):
        grad = gradient_obj_function(angles, target)

        # Update theta with gradient
        angles[0] -= learning_rate * grad[0]
        angles[1] -= learning_rate * grad[1]

        # Check for convergence
        J = objective_func_2(angles[0], angles[1], target)

        result[i+1] = [angles[0], angles[1], J]

        if J < tol:
            # print(f"Converged after {i} iterations.")
            break

    result = np.resize(result, (i+2, 3))
    return result

def objective_function_level_curve(theta1, k, l1, l2, x0, y0, clamp=True):
    D = np.sqrt(x0**2+ y0**2)
    thetastar= np.arctan2(y0, x0)
    alpha = theta1 - thetastar
    # linear trigonometric equation
    # a*sin(x) + b*sin(x) = c -> R sin(x + phi) = c
    # with R = sqrt(a**2 + b**2), phi=atan2(b/a)
    a = np.sin(alpha)
    b = np.cos(alpha) - D/l1
    c = (k**2 - (l1**2 + l2**2 + D**2))/(2*l1*l2) - D/l2*np.cos(alpha)
    R = np.sqrt(1+(D*l1)**2 -2*D/l1*np.cos(alpha))
    phi = np.arctan2(b, a)
    beta = np.arcsin(c/R) - phi
    return beta + thetastar
    