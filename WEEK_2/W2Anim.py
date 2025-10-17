from manim import *

class RegressionLine(VGroup):
    def __init__(
        self,
        m: float,
        q: float,
        axes: Axes,
        x_range: tuple[float, float] = (0, 1),
        color=BLUE,
        **kwargs
    ):
        super().__init__()
        self.ax=axes
        self.X1 = x_range[0]
        self.X2 = x_range[1]
        self.slope = ValueTracker(m)
        self.intercept = ValueTracker(q)
        self.line = Line(self.eval_to_point(self.X1), 
                         self.eval_to_point(self.X2),
                         color=color,
                         **kwargs)
        self.line.add_updater(
            lambda l: l.put_start_and_end_on(self.eval_to_point(self.X1), self.eval_to_point(self.X2))
        )
        self.add(self.line)
        self.proj_points = VGroup()
        self.proj_lines = VGroup()
    
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
        self.proj_points.add(proj_point)
        # self.add(proj_point)

        proj_line = Line(point, proj_point, **line_config)
        proj_line.add_updater(
            lambda line: line.put_start_and_end_on(
                point.get_center(),
                self.eval_to_point(x))
        )
        self.proj_lines.add(proj_line)
        # self.add(proj_line)

    def add_dataset(self, points):
        for point in points:
            self.add_data_point(point)


def E(m, q, data_points):
    x = data_points[:, 0]
    y = data_points[:, 1]
    E = np.sum(np.square((m*x +q) - y))
    return E

def R2(m, q, data_points):
    SSres = E(m, q, data_points)
    y = data_points[:, 1]
    SStot = np.sum(np.square(y - np.mean(y)))
    r2 = 1 - SSres/SStot
    return max(0, min(1, r2))


class ECounter(Variable):
    def __init__(self, regression_line: RegressionLine, data: np.ndarray, num_decimal_places = 2, **kwargs):
        super().__init__(
            E(regression_line.slope.get_value(), regression_line.intercept.get_value(), data),
            'E',
            num_decimal_places=num_decimal_places,
            color=BLACK,
            **kwargs)
        self.value.add_updater(
            lambda v: v.set_value(
                E(regression_line.slope.get_value(), regression_line.intercept.get_value(), data)
            )
        )

class R2Counter(Variable):
    def __init__(self, regression_line: RegressionLine, data: VGroup, num_decimal_places = 2, **kwargs):
        self.data = data
        self.regLine = regression_line
        super().__init__(
            R2(regression_line.slope.get_value(), regression_line.intercept.get_value(), self._data_coords()),
            'R^2',
            num_decimal_places=num_decimal_places,
            color=BLACK,
            **kwargs)
        self.value.add_updater(
            lambda v: v.set_value(
                R2(regression_line.slope.get_value(), regression_line.intercept.get_value(), self._data_coords())
            )
        )
    
    def _data_coords(self):
        return np.stack([self.regLine.ax.p2c(p.get_center()) for p in self.data], axis=0)[:,:2]

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
    def __init__(self, x_i_color = BLUE, y_i_color=ORANGE, central_buff=1):
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
        self.q_eq.next_to(self.m_eq, RIGHT, buff=central_buff).align_to(self.m_eq, UP)
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

    
    def get_sums_without_repetition(self) -> VGroup:
        return VGroup(
            self.m_sum_x[0], self.m_sum_y, self.m_sum_x_y, self.m_sum_x_sq
            ).copy().arrange(RIGHT).center()
    
    def _get_sums(self):
        return VGroup(
            *self.m_sum_x, self.m_sum_y, self.m_sum_x_y, self.m_sum_x_sq,
            self.q_sum_x, self.q_sum_y
        ) 

    def ExtractSumTerms(self, target: VGroup) -> Succession:
        sums = self._get_sums()
        sums2target_map = {0:0, 1:0, 2:0, 3:1, 4:1, 5:2, 6:3}
        return Succession(
            sums.animate(run_time=0).set_opacity(0),
            AnimationGroup(
                FadeOut(self),
                *[ReplacementTransform(sums[s].copy(), target[t])
                for s, t in sums2target_map.items()]
            )
        )

def mq_throgh_points(p1, p2):
    m = (p2[1]-p1[1])/(p2[0]-p1[0])
    q = p1[1] -m*p1[0]
    return m, q

def linear_reg_coeffs(dataset: np.ndarray):
    coeffs = np.polynomial.polynomial.Polynomial.fit(dataset[:, 0], dataset[:, 1], 1).convert().coef
    return coeffs[1], coeffs[0]

class WildFireIcon(VGroup):
    def __init__(self):
        self.forest_icon = SVGMobject(r'Assets\W2\pine_trees_icon.svg')
        u = self.forest_icon.height
        self.fire_icon = SVGMobject(r"Assets\W2\flame_icon.svg").scale_to_fit_height(u*0.6).shift(UP*u*0.3 + LEFT*u*0.2)
        self.fire_icon_2= SVGMobject(r"Assets\W2\flame_icon.svg").flip().scale_to_fit_height(u*0.45).shift(DOWN*u*0 + RIGHT*u*0.4)
        super().__init__(self.forest_icon, self.fire_icon, self.fire_icon_2)

class WildfireFactorsScheme(VGroup):
    def __init__(self, icons_height, **kwargs):
        self.tri = Triangle().scale(3).flip(axis=RIGHT).center()
        self.circles = VGroup(Circle(stroke_color=BLACK, fill_color=WHITE, radius= icons_height*2 + 0.5, stroke_width=6, fill_opacity=0).move_to(self.tri.get_vertices()[i]) for i in range(3))
        
        # self.wildfire_icon = SVGMobject(r'Assets\W2\forest_fire_icon.svg').scale_to_fit_height(icons_height*4).move_to(self.tri.get_vertices()[0])
        self.wildfire_icon = WildFireIcon().scale_to_fit_height(icons_height*4).move_to(self.tri.get_vertices()[0] + UP*0.1)
        self.high_temp_icon = SVGMobject(r'Assets\W2\high_temperature_icon.svg').set_color(RED).scale_to_fit_height(icons_height*4).move_to(self.tri.get_vertices()[1])
        self.humidty_icon = SVGMobject(r'Assets\W2\humidity_icon.svg').set_color(BLUE).scale_to_fit_height(icons_height*4).move_to(self.tri.get_vertices()[2])
        
        self.causal_arrows = VGroup(
            Line(self.tri.get_vertices()[0], self.tri.get_vertices()[1], color=BLACK, stroke_width=6, buff=icons_height*2 + 0.5),
            Line(self.tri.get_vertices()[0], self.tri.get_vertices()[2], color=BLACK, stroke_width=6, buff=icons_height*2 + 0.5),
        )

        super().__init__(self.causal_arrows, self.circles, self.wildfire_icon, self.high_temp_icon, self.humidty_icon, **kwargs)

class Test(Scene):
    def construct(self):
        w = WildfireFactorsScheme(0.6)
        self.add(w)
        