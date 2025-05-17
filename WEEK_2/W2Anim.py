from manim import *

class RegressionLine(VGroup):
    def __init__(self, m: float, q: float, axes: Axes, dash_length = 0.5, dashed_ratio = 0.5, **kwargs):
        super().__init__()
        self.ax=axes
        axes.p2c
        self.X1 = 0
        self.X2 = 1
        self.slope = ValueTracker(m)
        self.intercept = ValueTracker(q)
        self.line = DashedLine(self.eval_to_point(self.X1), 
                         self.eval_to_point(self.X2), 
                         dash_length=dash_length, dashed_ratio=dashed_ratio, **kwargs)
        self.line.add_updater(
            lambda l: l.put_start_and_end_on(self.eval_to_point(self.X1), self.eval_to_point(self.X2))
        )
        self.add(self.line)
        self.proj_points = []
        self.proj_lines = []
    
    def eval(self, x):
        return self.slope.get_value()*x +self.intercept.get_value()
    
    def eval_to_point(self, x):
        return self.ax.c2p(x, self.eval(x), 0)

    def add_data_point(self, point: Dot):
        point_coords = self.ax.p2c(point.get_center())
        x = point_coords[0]
        proj_point = Dot(self.eval_to_point(x))
        proj_point.add_updater(
            lambda p: p.move_to(self.eval_to_point(x))
        )
        self.proj_points.append(proj_point)
        self.add(proj_point)

        proj_line = Line(point, proj_point)
        proj_line.add_updater(
            lambda line: line.put_start_and_end_on(point.get_center(), proj_point.get_center())
        )
        self.proj_lines.append(proj_line)
        self.add(proj_line)

    def add_dataset(self, points):
        for point in points:
            self.add_data_point(point)


def E(m, q, data_points):
    x = data_points[:, 0]
    y = data_points[:, 1]
    E = np.sum(np.square((m*x +q) - y))
    return E


class ECounter(Variable):
    def __init__(self, regression_line: RegressionLine, data: np.ndarray, num_decimal_places = 2, **kwargs):
        super().__init__(
            E(regression_line.slope.get_value(), regression_line.intercept.get_value(), data),
            'E',
            num_decimal_places,
            **kwargs)
        self.tracker.add_updater(
            lambda tracker: tracker.set_value(
                E(regression_line.slope.get_value(), regression_line.intercept.get_value(), data)
            )
        )

def GenerateDataset(
    n: int,
    m: float,
    q: float,
    sigma: float = 1,
    seed: int = 0
) -> np.ndarray:
    RNG= np.random.default_rng(seed)
    x = RNG.uniform(0,1, size=n)
    y = m*x + q + RNG.normal(0, sigma, size=n)
    return np.column_stack((x,y))


def PointsFromData(data: np.ndarray, ax: Axes, **kwargs):
    return [Dot(ax.c2p(data[i, 0], data[i, 1]), **kwargs) for i in range(len(data))]


class Test(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0,2,0.5],
            y_range=[0,2,0.5],
            x_length=8,
            y_length=8,
            x_axis_config={'stroke_color':WHITE, 'include_ticks':True},
            y_axis_config={'stroke_color':WHITE, 'include_ticks':True}
        )
        m, q = 1, 0.5
        data = GenerateDataset(20, m, q, sigma=0.25)
        points = PointsFromData(data, ax, color=RED)
        self.add(ax, *points)
        l = RegressionLine(m,q, ax, color=GREEN, dash_length=0.25, dashed_ratio=0.25)
        self.add(l)
        l.add_dataset(points)
        self.wait()
        self.play(l.slope.animate.set_value(0), l.intercept.animate.set_value(0))
        self.wait()
        ecounter = ECounter(l, data).to_edge(UP)
        self.add(ecounter)
