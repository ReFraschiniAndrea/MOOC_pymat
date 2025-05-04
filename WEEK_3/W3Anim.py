from manim import *
import os


env = os.environ
env["PATH"] = r"C:\Users\rfand\AppData\Local\Programs\MiKTeX\miktex\bin\x64;" + env["PATH"]


def double_arm_kinematics(l1:float, l2:float, theta1:float, theta2:float, alpha=1, center=[0,0,0]):
    return [
        center[0] + l1 * np.cos(theta1) + alpha * l2 * np.cos(theta2),
        center[1] + l1 * np.sin(theta1) + alpha * l2 * np.sin(theta2),
        0
    ]

class NewDB(VMobject):
    def __init__(self,axes: Axes, l1, l2, theta1=0, theta2=0, color=BLUE, z_index=0):
        super().__init__()
        # internal attributer
        self.ax = axes
        self.l1 = l1
        self.l2 = l2
        self.theta1 = ValueTracker(theta1)
        self.theta2 = ValueTracker(theta2)
        # create the 2 arms and the joints
        self.foot = Dot(self.orig(), color=GRAY).set_z_index(z_index+1)
        self.joint = Dot(self.ax.c2p(l1, 0, 0), color=GRAY).set_z_index(z_index+1)
        self.hand =  Dot(self.ax.c2p(l1+l2, 0, 0), color=GRAY).set_z_index(z_index+1)
        self.arm1 = Line(self.orig(), self.ax.c2p(l1, 0, 0), stroke_color=color).set_z_index(z_index)
        self.arm2 = Line(self.ax.c2p(l1, 0, 0), self.ax.c2p(l1+l2, 0, 0), stroke_color=color).set_z_index(z_index)
        self.ARM1 = VGroup(self.arm1, self.foot).rotate(theta1, about_point=self.orig())
        self.ARM2 = VGroup(self.arm2, self.joint, self.hand).move_to(
            0.5*(self._hand_coord(theta1, theta2) + self._joint_coord(theta1))
        ).rotate(theta2)
        # updaters
        self.foot.add_updater(
            lambda mobj: mobj.move_to(self.orig())
        )
        self.joint.add_updater(
            lambda mobj: mobj.move_to(self._joint_coord(self.theta1.get_value()))
        )
        self.hand.add_updater(
            lambda mobj: mobj.move_to(self._hand_coord(self.theta1.get_value(), self.theta2.get_value()))
        )
        self.arm1.add_updater(
            lambda mobj: mobj.put_start_and_end_on(self.orig(),self._joint_coord(self.theta1.get_value()))
            )
        self.arm2.add_updater(
            lambda mobj: mobj.put_start_and_end_on(
                self._joint_coord(self.theta1.get_value()),
                self._hand_coord(self.theta1.get_value(), self.theta2.get_value())
            ))
        
        self.add(axes)
        self.add(self.foot, self.arm1, self.joint, self.arm2, self.hand)
        

    def orig(self):
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


class RobotGradientDescent():
    def __init__(self, l1, l2, target):
        self.l1 = l1
        self.l2 = l2
        self.target = np.asarray(target)

    def robot_position(self, t1, t2):
        x = self.l1*np.cos(t1) + self.l2*np.cos(t2)
        y = self.l1*np.sin(t1) + self.l2*np.sin(t2)
        return x, y

    def J(self, t1, t2):
        x, y = self.robot_position(t1, t2)
        return np.square(x - self.target[0]) + np.square(y - self.target[1])
    
    def distance(self, t1, t2):
        return np.sqrt(self.J(t1, t2))
    
    def gradient_x_robot(self, t1, t2):
        dxr_dt1 = - self.l1 * np.sin(t1)
        dxr_dt2 = - self.l2 * np.sin(t2)
        return dxr_dt1, dxr_dt2

    def gradient_y_robot(self, t1, t2):
        dyr_dt1 = self.l1 * np.cos(t1)
        dyr_dt2 = self.l2 * np.cos(t2)
        return dyr_dt1, dyr_dt2

    def gradient_obj_function(self, t1, t2):
        x, y = self.robot_position(t1, t2)
        dxr_dt1, dxr_dt2 = self.gradient_x_robot(t1, t2)
        dyr_dt1, dyr_dt2 = self.gradient_y_robot(t1, t2)

        dJ_dt1 = 2*(x - self.target[0])*dxr_dt1 + 2*(y - self.target[1])*dyr_dt1
        dJ_dt2 = 2*(x - self.target[0])*dxr_dt2 + 2*(y - self.target[1])*dyr_dt2

        return dJ_dt1, dJ_dt2
    
    def run(
        self,
        starting_point,
        max_iter = 1000,
        learning_rate = 0.01,
        tol = 1e-2
    ):
        result = np.zeros((max_iter + 1, 3))
        result[0] = [starting_point[0], starting_point[1], 
                     self.J(starting_point[0], starting_point[1])]
        t1, t2 = starting_point.copy()
        for i in range(max_iter):
            dx, dy = self.gradient_obj_function(t1, t2)

            # Update theta with gradient
            t1 -= learning_rate * dx
            t2 -= learning_rate * dy
            Jval = self.J(t1, t2)
            result[i+1] = [t1, t2, Jval]

            # Check for convergence
            if Jval < tol:
                break

        result = np.resize(result, (i+2, 3)) # weird I know
        return result

def objective_function_level_curve(
        theta1: np.ndarray, 
        G, 
        l1, 
        l2, 
        x0: float, 
        y0: float, 
        clamp=True
        ):
    D = np.sqrt(x0**2+ y0**2)
    thetastar= np.arctan2(y0, x0)
    alpha = theta1 - thetastar
    # linear trigonometric equation
    # a*sin(x) + b*sin(x) = c -> R sin(x + phi) = c
    # with R = sqrt(a**2 + b**2), phi=atan2(b/a)
    a = np.sin(alpha)
    b = np.cos(alpha) - D/l1
    # c = (k**2 - (l1**2 + l2**2 + D**2))/(2*l1*l2) + D/l2*np.cos(alpha)
    c = G + D/l2*np.cos(alpha)
    R = np.sqrt(1+(D*l1)**2 -2*D/l1*np.cos(alpha)) # argument is always positive
    phi = np.arctan2(b, a)
    beta1 = np.arcsin(c/R) - phi
    beta2 = np.pi - np.arcsin(c/R) - phi
    return beta1 + thetastar, beta2+thetastar