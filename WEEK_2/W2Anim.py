from manim import *

class RegressionLine(DashedLine):
    def __init__(self, m: float, q: float, axes: Axes, dash_length = ..., dashed_ratio = 0.5, **kwargs):
        self.axes=axes
        axes.p2c
        self.X1 = 0
        self.X2 = 1
        self.slope = ValueTracker(m)
        self.intercept = ValueTracker(q)
        super().__init__(self.eval_to_point(self.X1), 
                         self.eval_to_point(self.X2), 
                         dash_length=dash_length, dashed_ratio=dashed_ratio, **kwargs)
    
    def eval(self, x):
        return self.slope.get_value()*x +self.intercept.get_value()
    
    def eval_to_point(self, x):
        return self.axes.c2p(x, self.eval(x), 0)

    def add_data_point(self, point: Point):
        point_coords = self.axes.p2c(point.get_center())
        proj_point = Point([point_coords[0], self.eval(point.get_x()), 0])
        proj_line = Line(point, proj_point)
        proj_point.add_updater(
            lambda p:
        )
        proj_line.add_updater(
            lambda line:
        )