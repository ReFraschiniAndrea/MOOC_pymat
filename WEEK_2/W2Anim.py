from manim import *

class RegressionLine(VGroup):
    def __init__(
        self,
        m: float,
        q: float,
        axes: Axes,
        x_range: tuple[float, float] = (0, 1),
        color=BLUE,
        dash_length = 0.5,
        dashed_ratio = 0.5,
        **kwargs
    ):
        super().__init__()
        self.ax=axes
        self.X1 = x_range[0]
        self.X2 = x_range[1]
        self.slope = ValueTracker(m)
        self.intercept = ValueTracker(q)
        self.line = DashedLine(self.eval_to_point(self.X1), 
                         self.eval_to_point(self.X2),
                         color=color,
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

    def add_data_point(
        self,
        point: Dot,
        point_config: dict = {'color':TEAL},
        line_config: dict = {'color': TEAL_D}
    ):
        point_coords = self.ax.p2c(point.get_center())
        x = point_coords[0]
        proj_point = Dot(self.eval_to_point(x), **point_config)
        proj_point.add_updater(
            lambda p: p.move_to(self.eval_to_point(x))
        )
        self.proj_points.append(proj_point)
        self.add(proj_point)

        proj_line = Line(point, proj_point, **line_config)
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
            color=BLACK,
            **kwargs)
        self.tracker.add_updater(
            lambda tracker: tracker.set_value(
                E(regression_line.slope.get_value(), regression_line.intercept.get_value(), data)
            )
        )

def generate_regression_dataset(
    func: callable,
    n: int,
    x_range = (0,1),
    sigma: float = 1,
    seed: int = 0
) -> np.ndarray:
    RNG= np.random.default_rng(seed)
    x = RNG.uniform(*x_range, size=n)
    x = np.sort(x) # random, but in increasing order for convenience
    y = func(x) + RNG.normal(0, sigma, size=n)
    return np.column_stack((x,y))


def points_from_data(data: np.ndarray, ax: Axes, **kwargs):
    return VGroup(Dot(ax.c2p(data[i, 0], data[i, 1]), **kwargs) for i in range(len(data)))

class LinearRegressionEquations(VMobject):
    def __init__(self, x_i_color = BLUE, y_i_color=ORANGE):
        super().__init__()
        self.m_eq = MathTex(
            r'\hat{m} = '
            r'\frac{n \sum\limits_{i=1}^n x_i y_i - \sum\limits_{i=1}^n x_i \sum\limits_{i=1}^n y_i}'
            r'{ n \sum\limits_{i=1}^n x_i^2 - \left( \sum\limits_{i=1}^n x_i \right)^2}',
            color=BLACK
        )
        for i in [9, 10, 19, 20, 35, 37, 45, 46]:
            self.m_eq[0][i].set_color(x_i_color)
        for j in [11, 12, 26, 27]:
            self.m_eq[0][j].set_color(y_i_color)

        self.q_eq = MathTex(
            r'\hat{q} = '
            r'\frac{\sum\limits_{i=1}^n y_i - \hat{m} \sum\limits_{i=1}^n x_i}{n}',
            color=BLACK
        )
        self.q_eq[0][8:10].set_color(y_i_color)
        self.q_eq[0][18:20].set_color(x_i_color)
        self.q_eq.next_to(self.m_eq, RIGHT, buff=0.5).align_to(self.m_eq, UP)
        self.add(self.m_eq, self.q_eq)
        self.center()
 
        # add utilities to access certain terms
        self.m_sum_x = VGroup(
            self.m_eq[0][14:21],
            self.m_eq[0][40:47],
        )
        self.m_sum_y = self.m_eq[0][21:28]
        self.m_sum_x_y = self.m_eq[0][4:13]
        self.m_sum_x_sq = self.m_eq[0][30:38]

        self.q_sum_x = self.q_eq[0][13:20]
        self.q_sum_y = self.q_eq[0][3:10]


config.background_color = WHITE
config.pixel_width=960
config.pixel_height=720
class Test(Scene):
    def construct(self):
        # ax = Axes(
        #     x_range=[0,2,0.5],
        #     y_range=[0,2,0.5],
        #     x_length=8,
        #     y_length=8,
        #     x_axis_config={'stroke_color':WHITE, 'include_ticks':True},
        #     y_axis_config={'stroke_color':WHITE, 'include_ticks':True}
        # )
        # m, q = 1, 0.5
        # data = GenerateDataset(20, m, q, sigma=0.25)
        # points = PointsFromData(data, ax, color=RED)
        # self.add(ax, *points)
        # l = RegressionLine(m,q, ax, color=GREEN, dash_length=0.25, dashed_ratio=0.25)
        # self.add(l)
        # l.add_dataset(points)
        # self.wait()
        # self.play(l.slope.animate.set_value(0), l.intercept.animate.set_value(0))
        # self.wait()
        # ecounter = ECounter(l, data).to_edge(UP)
        # self.add(ecounter)

        a =  LinearRegressionEquations()
        self.add(a)#, index_labels(m[0]), index_labels(q[0]))
        self.play(Indicate(a.m_eq.sum_x_sq))
        # a= MathTex(r'\sum_{i=1}^n x_i', color=BLACK)
        # self.add(a)

