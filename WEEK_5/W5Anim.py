import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from mooc_utils.Generic_mooc_utils import SANS_SERIF_FONT, CODE_FONT, PixelArray, custom_get_axis_labels
from typing import List, Iterable
import skimage
import scipy.ndimage

SATURATED_RED = ManimColor("#FF1500")

class ValueDisplay(VMobject):
    """Value with a tracker for animataion (`Variable` without the label)"""
    def __init__(
        self,
        var: float,
        num_decimal_places: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.tracker = ValueTracker(var)
        self.value = DecimalNumber(
            self.tracker.get_value(),
            num_decimal_places=num_decimal_places,
            **kwargs
        )
        self.value.add_updater(lambda v: v.set_value(self.tracker.get_value()))
        self.add(self.value)


class DigitRecognitionOutputCircle(VMobject):
    """Single outlined circle containing a digit"""
    _DIGIT_TEXT_KWARGS = {
        'font':SANS_SERIF_FONT, 'font_size':30, 'weight':MEDIUM,
        'fill_color': BLACK, 'stroke_width':1, 'stroke_color':BLACK
    }
    def __init__(
        self,
        digit: int,
        radius: float = 0.3, fill_color=RED,
        stroke_width: float = 4, stroke_color=BLACK,
        text_kwargs : dict = {}
    ):
        super().__init__()
        self.circle = Circle(radius=radius, fill_color=fill_color, fill_opacity=1, stroke_color=stroke_color, stroke_width=stroke_width)
        kwargs = self._DIGIT_TEXT_KWARGS.copy()
        kwargs.update(text_kwargs)
        self.digit = Text(str(digit), **kwargs)
        self.digit.move_to(self.circle)
        self.add(self.circle, self.digit)


class DigitRecognitionOutputLayer(VGroup):
    """Digit recognition output (0 to 9) illsutration."""
    def __init__(
        self,
        buff: float = 0.08,
        **kwargs
    ):
        self.circles = [DigitRecognitionOutputCircle(i, **kwargs) for i in range(10)]
        super().__init__(self.circles)
        self.arrange(DOWN, buff=buff).center()
    
    def activate(self, digit: int, color=GREEN, **kwargs):
        self.circles[digit].circle.set_fill(color)

    def Activate(self, digit: int, color=GREEN, **kwargs) -> Animation:
        return ApplyMethod(self.circles[digit].circle.set_fill, color, **kwargs)
    
    
class LayerTitle(Paragraph):
    def __init__(self, *text, line_spacing = -1, alignment = 'center', font_size=32):
        super().__init__(*text, line_spacing=line_spacing, alignment=alignment, 
                         color=BLACK, stroke_width=0, font=SANS_SERIF_FONT, font_size=font_size, weight=NORMAL, t2w={text[0]:BOLD})
        self.scale(0.8)
        

class DenseLayer(VMobject):
    """Illsutration of a dense layer of a neural network."""
    def __init__(
        self,
        n_input_nodes: int = 16,
        input_nodes_radius: float = 0.1,
        input_nodes_color = DARK_BLUE,
        input_nodes_buff: float = 0.1,
        n_output_nodes:int = 3,
        output_nodes_radius = 0.2,
        output_nodes_color = GOLD,
        output_nodes_buff = 1,
        horizonatl_buff: float = 1,
        line_color = BLUE,
        line_width = 1,
        add_vdots: bool = False
    ):
        super().__init__()
        self.n_input_nodes = n_input_nodes
        self.n_output_nodes = n_output_nodes
        self.input_nodes = VGroup(Circle(radius=input_nodes_radius, color=input_nodes_color, fill_opacity=1, stroke_width=0) for _ in range(n_input_nodes))
        self.input_nodes.arrange(DOWN, buff=input_nodes_buff)
        self.output_nodes = VGroup(Circle(radius=output_nodes_radius, color=output_nodes_color, fill_opacity=1, stroke_width=0) for _ in range(n_output_nodes))
        self.output_nodes.arrange(DOWN, buff=output_nodes_buff)

        self.output_nodes.next_to(self.input_nodes, RIGHT, buff= horizonatl_buff)

        # create the lines between input and output nodes
        self.vlines = VGroup()
        for output in self.output_nodes:
            for input in self.input_nodes:
                line = Line(input.get_center(), output.get_center(), color=line_color, stroke_width=line_width)
                self.vlines.add(line)

        self.add(self.vlines, self.input_nodes, self.output_nodes)
        if add_vdots:
            self.add_vdots()
        self.center()

    def add_vdots(self):
        vdots = MathTex(r'\vdots', color=BLACK, font_size=32)
        self.vdots_in = vdots.copy().next_to(self.input_nodes, DOWN)
        self.vdots_out = vdots.copy().next_to(self.output_nodes, DOWN)
        self.add(self.vdots_in, self.vdots_out)

    def get_lines_from_input_node(self, n: int):
        return VGroup(
            self.vlines[n + i*self.n_output_nodes] for i in  range(self.n_output_nodes)
        )

    def get_lines_from_output_node(self, n: int):
        return self.vlines[n*self.n_input_nodes : (n+1)*self.n_input_nodes]
    
    def EdgePropagationAnimation(self, color=BLUE_B, stroke_width=None, run_time=1, lag_ratio=0.5, **kwargs):
        stroke_width = self.vlines.stroke_width * 1.5 if stroke_width is None else stroke_width
        edges_group = self.vlines.copy().set_stroke(color, stroke_width)
        return AnimationGroup(
            *[ShowPassingFlash(edge, **kwargs)
            for edge in edges_group],
            lag_ratio=lag_ratio,
            run_time=run_time
        )

        
class NodeScheme(VGroup):
    """Illustration for explanation of how a single NN node works."""
    def __init__(
        self,
        circle_color = GOLD,
        arrow_color = BLUE,
        n_inputs: int = 3,
        label_font_size = 48,
        arrow_width = 4
    ):
        super().__init__()
        circle_radius = 1.8
        arrow_size = 1.8
        max_angle= PI/5
        buff=0.3
        arrow_config = {'color':arrow_color, 'stroke_width':arrow_width}
        # arrow_config = {'color':arrow_color, 'stroke_width':arrow_width, 'max_tip_length_to_length_ratio': 0.4, 'max_stroke_width_to_length_ratio':20}

        self.circle = Circle(radius=circle_radius, color=circle_color, fill_opacity=1, stroke_width=0)
        # Input labels, arrows and weights
        self.input_labels = VGroup(MathTex(f'x_{i}', font_size=label_font_size, color=BLACK) for i in range(n_inputs))
        self.weight_labels = VGroup(MathTex(f'w_{i}', font_size=label_font_size, color=BLACK) for i in range(n_inputs))

        self.input_arrows = VGroup()
        angles = np.linspace(PI-max_angle, PI + max_angle, n_inputs)
        r = circle_radius + arrow_size
        for label, angle, weight in zip(self.input_labels, angles, self.weight_labels):
            label.move_to([r*np.cos(angle), r*np.sin(angle), 0])
            arrow = Line(start=[(r-buff)*np.cos(angle), (r-buff)*np.sin(angle), 0], end=[circle_radius*np.cos(angle), circle_radius*np.sin(angle), 0],
                                 buff = 0, **arrow_config)
            weight.add_to_back(BackgroundRectangle(weight, color=WHITE, fill_opacity=1, stroke_width=0))
            start, end = arrow.get_start_and_end()
            weight.move_to((end - start) * 0.5 + start)
            self.input_arrows.add(arrow)

        # Output label and arrow
        self.output_label = MathTex(f'y', font_size=label_font_size, color=BLACK)
        self.output_label.move_to([r, 0, 0])
        self.output_arrow = Line(start=[circle_radius, 0, 0], end=[(r-buff), 0, 0],
                                  buff = 0, **arrow_config)

        # Activation Function Scheme
        self.activation_function = ActivationFunctionScheme(sigma_font_size=label_font_size).move_to(self.circle).shift(RIGHT*circle_radius/2)
        self.sum = MathTex(r'\Sigma', color=BLACK).scale_to_fit_height(circle_radius*0.65).move_to(self.circle).shift(LEFT*circle_radius/2)
        self.middle_arrow = Line(self.sum.get_right(), self.activation_function.get_left(), buff=0.05, **arrow_config)

        # bias b
        self.b = MathTex('b', color=BLACK, font_size=label_font_size).move_to(self.circle).shift(UP*circle_radius/2*1.25).shift(LEFT*0.1)
        self.top_middle_arrow = Line(self.b.get_bottom()+DOWN*0.1, self.circle.get_center() + LEFT*0.1 , buff=1, **arrow_config)
        
        # Add tips of teh arrows at the end manually so they are all equal
        tip = self.output_arrow.create_tip(tip_length=0.25, tip_width=0.25)
        for arrow in [self.output_arrow, self.middle_arrow, *self.input_arrows]:
            arrow.add_tip(tip.copy())

        self.add(self.circle, self.input_arrows, self.input_labels, self.weight_labels, self.output_label, self.output_arrow,
                 self.sum, self.activation_function, self.b, self.middle_arrow, self.top_middle_arrow)
        
class SigmoidFunctionBezier(CubicBezier):
    def __init__(self, color=BLACK, stroke_width=4):
        p1 = np.array((2,2,0))
        p1_handle = p1 - 3*RIGHT
        super().__init__(p1, p1_handle, -p1_handle, -p1, color=color, stroke_width=stroke_width)

class ActivationFunctionScheme(VMobject):
    def __init__(self, text: str = 'Activation\nFunction', color=DARK_BLUE, stroke_width=4, sigma_font_size=48):
        super().__init__()
        self.function = SigmoidFunctionBezier(color=color, stroke_width=stroke_width).scale_to_fit_height(0.7)
        self.sigma = MathTex(r'\sigma', color=BLACK, font_size=sigma_font_size).align_to(self.function, UL)
        self.text = Paragraph(text, alignment='center', color=color, font=SANS_SERIF_FONT, font_size=50
                              ).scale_to_fit_width(1).next_to(self.function, DOWN, buff=SMALL_BUFF)  # need to use bigger font size and scale it down due to bad kerning issue
        self.background_rectangle = SurroundingRectangle(
            self.function, self.sigma, self.text,
            fill_color=WHITE, fill_opacity=1, stroke_color=color, stroke_width=stroke_width,
            corner_radius=0.1
        )
        self.add(self.background_rectangle, self.function, self.text, self.sigma)

class SoftmaxLayer(VMobject):
    def __init__(self, color=DARK_BLUE, font_size=64, stroke_width=4):
        super().__init__()
        self.text = Text('Softmax', font=SANS_SERIF_FONT, font_size=font_size, color=color, weight=BOLD)
        self.text.rotate(PI/2)
        self.background_rectangle = SurroundingRectangle(
            self.text, stroke_color=color, stroke_width=stroke_width, fill_color=WHITE, fill_opacity=1,
            corner_radius=0.15, buff=0.25)
        self.add(self.background_rectangle, self.text)


class BrokenArrow(Line):
    """An arrow that bends with 90° turns (either 1 or 2 turns possible)."""
    def __init__(self, start, end, first_direction='h', n_turns=1, add_tip: bool=True, **kwargs):
        super().__init__(start, end, **kwargs)
        dir = 1 if first_direction=='h' else 0
        match n_turns:
            case 1:
                keypoint = [start[dir], end[1-dir], 0]
                keypoints = [start, keypoint, end]
            case 2:
                middle_point = (start+end)/2
                if dir:
                    keypoint1 = [middle_point[0], start[1], 0]
                    keypoint2 = [middle_point[0], end[1], 0]
                else:
                    keypoint1 =  [start[0], middle_point[1], 0]
                    keypoint2 =  [end[0], middle_point[1],0]
                keypoints = [start, keypoint1, keypoint2, end]
            case _:
                raise NotImplementedError
        self.keypoints = keypoints
        self.set_points_as_corners(keypoints)
        if add_tip:
            self.add_tip()

    def add_tip(self, tip=None, tip_length=None, tip_width=None):
        if tip is None:
            self.tip = self.create_tip(tip_length=tip_length, tip_width=tip_width)
        else:
            self.tip=tip
        self.position_tip(self.tip)
        #readjust to take into account tip
        self.keypoints[-1] = self.tip.base
        self.set_points_as_corners(self.keypoints)
        self.add(self.tip)

class PixelImage(ImageMobject):
    def __init__(
        self,
        filename_or_array: str | np.ndarray,
        pixel_size: float = None,
        add_outline:bool = False,
        outline_kwargs: dict = {},
        **kwargs
    ):
        super().__init__(filename_or_array, **kwargs)
        self.set_resampling_algorithm(RESAMPLING_ALGORITHMS['nearest'])

        # extract size of the image
        self.im_height, self.im_width = self.get_pixel_array().shape[:2]

        # Create axes for accessing image-space coordinates
        self.axes = Axes(
            [0, self.im_width], [0, -self.im_height], self.width, self.height,
            tips=False, axis_config={'include_ticks':False, 'stroke_width':0},  y_axis_config={"scaling": LinearBase(scale_factor=-1)})
        self.axes.move_to(self, aligned_edge=UL)
        self.axes.set_opacity(0)
        self.add(self.axes)
        
        self.outline: Rectangle = None
        if pixel_size is not None:
            self.scale_to_fit_height(self.im_height*pixel_size)
        if add_outline:
            self.add_outline(**outline_kwargs)
        
    def c2p(self,  *coords):
        "Image space coordinates (x (horiz),y(vetical, 0))"
        return self.axes.c2p(*coords)
    
    def i2p(self, i:int, j:int):
        """Get center of the (i,j) pixel (i row, j column)"""
        return self.c2p(0.5 + j, 0.5 + i, 0)
    # NOTE: these getters break if the image is rotated
    def get_pixel_width(self):
        return self[0].width / self.im_width
    def get_pixel_height(self):
        return self[0].height / self.im_height
    
    def get_pixel_copy(self, i,j, add_value=False, match_z_index: bool = True, stroke_width=0, **kwargs):
        value = self.pixel_array[i, j, 0]
        color = rgb_to_color((value, value, value))
        pixel = Rectangle(color, self.get_pixel_height(), self.get_pixel_width(), stroke_width=stroke_width, fill_opacity=1)
        pixel.move_to(self.i2p(i,j))
        if match_z_index:
            pixel.set_z_index(self.z_index)
        return pixel

    def get_highlight(self, height: float, width:float, color=YELLOW, stroke_width=4, fill_opacity=0, **kwargs) -> Rectangle:
        highlight = Rectangle(
            height=height, width=width,
            color=color, stroke_width=stroke_width, fill_opacity=fill_opacity,
            **kwargs
        ).set_z_index(self.z_index + 1)
        return highlight
    
    def get_pixel_highlight(self, position=(0,0), **kwargs):
        highlight = self.get_highlight(
            height=self.get_pixel_height(), width=self.get_pixel_width(),
            **kwargs
        )
        highlight.move_to(self.i2p(*position))
        return highlight
    
    def add_outline(self, color=DARK_BLUE, stroke_width=4):
        self.outline = SurroundingRectangle(self[0], buff=0, fill_opacity=0, color=color, stroke_width=stroke_width)
        self.set_z_index(1)
        self.outline.move_to(self[0]).set_z_index(0)
        self.add_to_back(self.outline)

    def get_size_brace(self, direction, label: str, brace_config: dict = {}, label_config: dict = {}):
        return get_labeled_brace(self, direction, label, brace_config, label_config)


class ImageStack(Group):
    """Stack of images that are partially overlayed.
    The first one is the one on top, i.e completely visible."""
    def __init__(self, images: List[str | np.ndarray], offset= UL*0.5, **kwargs):
        self.top_z_index: int = 2*len(images)
        self.images: List[PixelImage] = [PixelImage(im, **kwargs) for im in images]
        for i, im, in enumerate(self.images):
            im.shift(i*offset)
            im.set_z_index(self.top_z_index-2*i)
            im.outline.set_z_index(im.z_index-1)
        
        super().__init__(*self.images)
        self.center()

    def top(self):
        return self[0]
    
    def restore(self):
        for mob in self.submobjects:
            mob.restore()
        return self
    
    def save_state(self):
        for mob in self.submobjects:
            mob.save_state()

    def get_stack_brace(self, direction, label, brace_config: dict = {}, label_config: dict = {}):
        b_config = {'color':DARK_BLUE}; b_config.update(brace_config)
        l_config = {'color': BLACK, 'font':SANS_SERIF_FONT, 'font_size':48}; l_config.update(label_config)
        brace = BraceBetweenPoints(self[-1].get_corner(direction), self[0].get_corner(direction), direction, **b_config)
        label_mob = Text(label, **l_config)
        brace.put_at_tip(label_mob)
        return VGroup(brace, label_mob)

def create_gizmo(start: Rectangle,  end: Rectangle):
    line_config = {'stroke_width': start.stroke_width, 'stroke_color':start.stroke_color}
    # Naive and lazy way to determine direction: only "diagonal" ones are considered
    dir = normalize(end.get_center() - start.get_center())
    ortho_dir = np.array((-dir[1], dir[0], 0))
    # clamp to one of the 4 diagonals
    # ortho_dir = np.sign(ortho_dir)

    line_1 = Line(start.get_critical_point(ortho_dir), end.get_critical_point(ortho_dir), **line_config)
    line_2 = Line(start.get_critical_point(-ortho_dir), end.get_critical_point(-ortho_dir), **line_config)
    return VGroup(line_1, line_2).set_z_index(max(start.z_index, end.z_index))


class CNNDigitRecognitionScheme(Group):
    def _filter(image: np.ndarray, kernel:np.ndarray):
        f = scipy.ndimage.convolve(image.astype(np.float32), kernel.astype(np.float32))
        f = np.clip((f+600)/(1200), 0, 1)
        return (f*255).astype(np.uint8)
    
    def __init__(
        self,
        input: np.ndarray,
        input_label: int,
        n_filters: int = 5,
        pooling_factor: int = 2,
        pixel_size: float  | Iterable[float]= 0.05,
        horizontal_spacing: float = 0.8,  # space between main layer blocks
        outline_kwargs: dict = {},
        dense_layer_kwargs: dict = {},
        highlights_kwargs: dict = {}
    ):
        super().__init__()
        self.input_array_: np.ndarray = input
        self.pooling_factor = pooling_factor
        self.CONV_HIGHLIGHT_POS = (7,7)
        self.POOLING_HIGHLIGHT_POS = (2,2)

        # Create intermediate values by generating random filters and then pooling
        np.random.seed(0)
        self.filters_ =  [
            np.array([[-0.82,  0.01,  0.79],
                      [-0.88,  0.05,  0.91],
                      [-0.84,  0.02,  0.86],]),  # vertical edge
            np.array([[-0.77, -0.81, -0.74],
                      [ 0.02,  0.04,  0.01],
                      [ 0.83,  0.88,  0.80],]),  # horizontal edge
            np.array([[-0.65, -0.12,  0.05],
                      [-0.10, -0.72,  0.11],
                      [ 0.04,  0.13,  0.69],]),  # diagonal edge
            np.array([[-0.48, -0.61, -0.50],
                      [-0.59,  2.31, -0.62],
                      [-0.52, -0.58, -0.47],]),  # laplacian (stroke width)
            np.array([[ 0.71,  0.69, -0.03],
                      [ 0.68, -0.92, -0.81],
                      [-0.04, -0.78, -0.75],])  # corner detection
        ]
        self.filtered_ = [CNNDigitRecognitionScheme._filter(self.input_array_, filter) for filter in self.filters_]
        self.pooled_ = [skimage.measure.block_reduce(filtered, block_size=pooling_factor, func=np.max) for filtered in self.filtered_]
        self.flattened_ = np.concatenate([pooled.reshape((-1, 1)) for pooled in self.pooled_[::-1]], axis=0)  # reverse order

        # Create The actual mobjects
        if not isinstance(pixel_size, Iterable):
            pixel_size = [pixel_size for _ in range(4)]
        self.input = PixelImage(input, pixel_size=pixel_size[0], add_outline=True, outline_kwargs=outline_kwargs)
        self.conv_layer = ImageStack(
            self.filtered_,  offset=UL*4*pixel_size[1], pixel_size=pixel_size[1],
            add_outline=True, outline_kwargs=outline_kwargs
        )
        self.pooling_layer = ImageStack(
            self.pooled_, offset=UL*1.5*pixel_size[2], pixel_size=pixel_size[2],
            add_outline=True, outline_kwargs=outline_kwargs
        )
        self.flattened_vector = PixelImage(self.flattened_, pixel_size=pixel_size[3], add_outline=True, outline_kwargs=outline_kwargs)
        self.dense_layer = DenseLayer(**dense_layer_kwargs)
        self.softmax = SoftmaxLayer(font_size=48)
        self.output_layer = DigitRecognitionOutputLayer()
        self.flattened_vector.stretch_to_fit_height(self.dense_layer.height*1.1)

        # Arrange Everything
        # self.flattened_vector.next_to(self.dense_layer, LEFT, buff=0.25)
        self.softmax.next_to(self.dense_layer, RIGHT, buff=0.15)
        Group(
            self.input, self.conv_layer, self.pooling_layer,
            self.flattened_vector, Group(self.dense_layer, self.softmax),
            self.output_layer
        ).arrange(buff=horizontal_spacing)
        # Add dense layer vdots afterwards, so that it remains centered
        self.dense_layer.add_vdots()
        self.add(
            self.input, self.conv_layer, self.pooling_layer,
            self.flattened_vector, self.dense_layer, self.softmax,
            self.output_layer
        )

        # Create Labels
        self.input_title = LayerTitle('Input').next_to(self.input, UP)
        self.dense_layer_title = LayerTitle('Dense','layer').next_to(VGroup(self.dense_layer, self.softmax), UP, buff=0.5)
        self.conv_layer_title = LayerTitle('Convolutional','layer').next_to(self.conv_layer, UP).align_to(self.dense_layer_title, UP)
        self.flattening_layer_title = LayerTitle('Flatten','layer').next_to(self.flattened_vector, UP).align_to(self.dense_layer_title, UP)
        self.pooling_layer_title = LayerTitle('Pooling','layer').next_to(self.pooling_layer, UP).align_to(self.dense_layer_title, UP)
        self.output_layer_title = LayerTitle('Output').next_to(self.output_layer, UP)
        self.add(self.input_title, self.conv_layer_title, self.pooling_layer_title, self.flattening_layer_title, self.dense_layer_title, self.output_layer_title)

        # Create Gizmos for highlight
        self.input_highlight = self.input.get_pixel_highlight(position=self.CONV_HIGHLIGHT_POS, **highlights_kwargs).scale(3)
        self.conv_highlight_1 = self.conv_layer.images[0].get_pixel_highlight(position=self.CONV_HIGHLIGHT_POS, **highlights_kwargs)
        self.conv_highlight_2 = self.conv_highlight_1.copy().scale(pooling_factor).move_to(
            self.conv_layer.images[0].c2p(pooling_factor*self.POOLING_HIGHLIGHT_POS[1], pooling_factor*self.POOLING_HIGHLIGHT_POS[0],0), aligned_edge=UL)
        self.pooling_highlight_1 = self.pooling_layer.images[0].get_pixel_highlight(position=self.POOLING_HIGHLIGHT_POS, **highlights_kwargs)
        self.pooling_highlight_2 = self.pooling_layer.images[-1].get_pixel_highlight(position=(0,0), **highlights_kwargs)
        self.flattened_highlight = self.flattened_vector.get_pixel_highlight(position=(0,0), **highlights_kwargs)
        self.in_conv_gizmo = create_gizmo(self.input_highlight, self.conv_highlight_1)
        self.conv_pool_gizmo = create_gizmo(self.conv_highlight_2, self.pooling_highlight_1)
        self.pool_flat_gizmo = create_gizmo(self.pooling_highlight_2, self.flattened_highlight)
        self.add(
            self.input_highlight, self.conv_highlight_1, self.conv_highlight_2,
            self.pooling_highlight_1, self.pooling_highlight_2, self.flattened_highlight,
            self.in_conv_gizmo, self.conv_pool_gizmo, self.pool_flat_gizmo
        )

        self.output_arrow = self.get_output_arrow(input_label)
        self.add(self.output_arrow)
        
        self.center()

    def get_output_arrow(self, digit: int):
        output_arrow = BrokenArrow(self.softmax.get_right(), self.output_layer[digit].get_left(),
                                   color=DARK_BLUE, stroke_width=4,
                                   n_turns=2, first_direction='h', add_tip=False)
        tip = output_arrow.create_tip(tip_length=0.2, tip_width=0.2)
        output_arrow.add_tip(tip)
        return output_arrow
    
    def get_layer_highlight(
        self,
        layer_to_highlight: str,
        stroke_width=4, color=GOLD,
        height=8, v_buff=0.5, h_buff=0.2
    ):
        match layer_to_highlight:
            case 'conv' | 'convolutional':
                layer = Group(self.conv_layer, self.conv_layer_title)
            case 'pool' | 'pooling':
                layer = Group(self.pooling_layer, self.pooling_layer_title)
            case 'flat' | 'flattening':
                layer = Group(self.flattened_vector, self.flattening_layer_title)
            case 'dense' | 'NN':
                layer = Group(self.dense_layer, self.dense_layer_title, self.softmax)
            case 'all':
                layer = Group(self.conv_layer, self.conv_layer_title, self.dense_layer_title, self.softmax)
            case _:
                raise ValueError()
            
        highlight = RoundedRectangle(
            width=layer.width + 2*h_buff,
            height=height,
            color=color, stroke_width=stroke_width, fill_opacity=0,
            corner_radius=0.25
        ).set_z_index(20)
        highlight.move_to(layer).align_to(self.dense_layer_title.get_top() + v_buff*UP, UP)

        return highlight

    def restore(self):
        for mob in self.submobjects:
            mob.restore()
        return self
    
    def save_state(self):
        for mob in self.submobjects:
            mob.save_state()

class LearnableCoefficientsScheme(VMobject):
    matrix_default_config =  {'v_buff':0.6, 'h_buff':0.6, 'bracket_h_buff':0.1, 'bracket_v_buff':0.1}
    def __init__(
        self,
        font_size=28,
        kernel_colors=[RED, GREEN, ORANGE],
        matrix_config: dict = {}
    ):
        super().__init__()
        self.filter_entries_t = Text('Filter entries:', font=SANS_SERIF_FONT, font_size=font_size, color=BLACK)
        self.weights_and_biases_t =  Text('Weights and biases of the dense layer:', font=SANS_SERIF_FONT, font_size=font_size, color=BLACK)

        # Filter entries
        matrix_cfg = self.matrix_default_config.copy()
        matrix_cfg.update(matrix_config)
        filter_matrix = Matrix([[f'l_{3*i +j}' for j in range(3)] for i in range(3)], color=BLACK, **matrix_cfg)
        self.filters: VGroup[Matrix] = VGroup(filter_matrix.copy().set_color(c) for c in kernel_colors)
        self.filters.add(MathTex(r'\dots', color=BLACK, font_size=48))
        self.filters.arrange(RIGHT, buff=0.5)

        # NN parameters
        self.weights_eq = MathTex(r'\mathbf{W}=[w_0,w_1,w_2,w_0,w_1,w_2,w_0,w_1,w_2,\dots]', color=BLACK, font_size=48,
                             substrings_to_isolate=[f'w_{i}' for i in range(3)])
        substr_bias_eq = [r'\mathbf{b}','=', '[', ']', r'\dots', ',']  # Needed for later
        self.bias_eq = MathTex(r'\mathbf{b}=[b,b,b,\dots]', color=BLACK, font_size=48,
                          substrings_to_isolate=substr_bias_eq)  # isolating 'b' breaks here
        
        # Arrange vertically
        self.filters.next_to(self.filter_entries_t, DOWN, buff=0.3)
        self.weights_and_biases_t.next_to(self.filters, DOWN, buff=0.5)
        VGroup(self.weights_eq, self.bias_eq).arrange(DOWN, buff=0.5).next_to(self.weights_and_biases_t, DOWN, buff=0.25)

        # Add surrounding rectangle
        self.add(self.filter_entries_t, self.filters, self.weights_and_biases_t, self.weights_eq, self.bias_eq)
        self.background_rectangle = SurroundingRectangle(self, stroke_color=BLUE, stroke_width=4, fill_opacity=0, buff=0.5, corner_radius=0.25)
        self.add(self.background_rectangle)
        self.center()

def UpdateLearnableCoefficients(coefficients, updates=None, u_range=1, dt_per_inc=0.3):
    if updates is  None:
        updates = u_range*(2*np.random.random(len(coefficients)) - 1)
        updates = np.round(updates, 1)
        updates[updates==0]=0.1
        run_times = np.abs(updates)/0.1 * dt_per_inc
    return AnimationGroup(
        value.tracker.animate(rate_func=linear, run_time = dt).set_value(value.tracker.get_value() + delta)
        for value, delta, dt in zip(coefficients, updates, run_times) 
    )


class ReLUPlot(VMobject):
    def __init__(self, color=BLUE, stroke_width=4, label_config = {'color':BLACK, 'font_size':48,'font': SANS_SERIF_FONT}):
        super().__init__()
        self.axes = Axes(x_range=[-3.5,3.5], y_range=[-0.5, 3.5], x_length=10, y_length=10*4/7, axis_config={'color': BLACK})
        relu = lambda x: np.maximum(0,x)
        self.relu = self.axes.plot(relu, x_range=[-3.5,3.25], color=color, stroke_width=stroke_width, use_smoothing=False)
        x_label = Text('x', **label_config)
        y_label = Text('ReLU(x)', **label_config)
        self.labels = custom_get_axis_labels(self.axes, x_label, y_label)
        self.add(self.axes, self.relu, self.labels)

def get_labeled_brace(mob: Mobject, direction, label: str, brace_config: dict = {}, label_config: dict = {}):
        b_config = {'color':DARK_BLUE}; b_config.update(brace_config)
        l_config = {'color': BLACK, 'font':SANS_SERIF_FONT, 'font_size':48}; l_config.update(label_config)
        brace = Brace(mob, direction, **b_config)
        label_mob = Text(label, **l_config)
        brace.put_at_tip(label_mob)
        return VGroup(brace, label_mob)

def get_labeled_braces(mob, dir1, lab1, dir2, lab2, brace_config: dict = {}, label_config: dict = {}):
    return VGroup(
        get_labeled_brace(mob, dir1, lab1, brace_config=brace_config, label_config=label_config),
        get_labeled_brace(mob, dir2, lab2, brace_config=brace_config, label_config=label_config)
    )

class KerasCNNSummary(VMobject):
    # Keras-like colors
    KERAS_BLUE = ManimColor('#50acff')
    KERAS_GREEN = ManimColor('#00af00')
    KERAS_AQUA = ManimColor('#00d7ff')

    def __init__(self, font=CODE_FONT, font_size=18):
        col_labels_config={'font':font, 'font_size':font_size, 'color':BLACK, 'weight':ULTRABOLD}
        entry_config={
            'font':font, 'font_size':font_size, 'color':BLACK, 'weight':MEDIUM,
            't2c':{
                **{layer: self.KERAS_BLUE for layer in ['Conv2D', 'MaxPooling2D', 'Flatten', 'Dense']},
                'None':self.KERAS_AQUA,
                **{str(digit): self.KERAS_GREEN for digit in [28,32,4,512,10]}}
        }
        super().__init__()
        # Table data
        self.table = Table(
            [
                ["conv2d (Conv2D)", "(None, 28, 28, 32)", "320"],
                ["max_pooling2d (MaxPooling2D)", "(None, 4, 4, 32)", "0"],
                ["flatten (Flatten)", "(None, 512)", "0"],
                ["dense (Dense)", "(None, 10)", "5,130"],
            ],
            col_labels=[
                Text("Layer (type)", **col_labels_config),
                Text("Output Shape", **col_labels_config),
                Text("Param #", **col_labels_config),
            ],
            include_outer_lines=True,
            line_config={"stroke_width": 2, 'color':BLACK},
            arrange_in_grid_config={'cell_alignment':LEFT},
            v_buff=0.4, h_buff=1,
            element_to_mobject_config=entry_config
        )
        # Column 3 (green)
        self.table.get_entries_without_labels()[2::3].set_color(self.KERAS_GREEN)
    
        # Title
        self.title = Text('Model: "sequential"', **col_labels_config)
        self.title.next_to(self.table, UP).align_to(self.table, LEFT)

        # Footer
        self.footer = Paragraph(
            'Total params: 5,450 (21.29 KB)',
            'Trainable params: 5,450 (21.29 KB)',
            'Non-trainable params: 0 (0.00 B)',
            font=font, color=BLACK, font_size=font_size,
            alignment='left', line_spacing=0.5,
            t2c={'5,450':self.KERAS_GREEN},
            t2w={'Total params:':BOLD, 'Trainable params:':BOLD, 'Non-trainable params:':BOLD},
        )
        self.footer[2][20].set_color(self.KERAS_GREEN)
        self.footer.next_to(self.table, DOWN).align_to(self.table, LEFT)

        self.add(self.title, self.table, self.footer)
        self.center()