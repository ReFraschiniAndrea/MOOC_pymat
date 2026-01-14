import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from mooc_utils.Generic_mooc_utils import SANS_SERIF_FONT, PixelArray
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
        self.center()

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
    
    def save_state(self):
        for mob in self.submobjects:
            mob.save_state()

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
        self.filters_ =  [np.random.random((3,3)) for _ in range(n_filters)]
        self.filtered_ = [scipy.ndimage.convolve(self.input_array_, filter).astype(np.uint8) for filter in self.filters_]
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
        self.softmax = SoftmaxLayer(font_size=54)
        self.output_layer = DigitRecognitionOutputLayer()
        self.flattened_vector.stretch_to_fit_height(self.dense_layer.height*1.3)

        # Arrange Everything
        self.flattened_vector.next_to(self.dense_layer, LEFT, buff=0.25)
        self.softmax.next_to(self.dense_layer, RIGHT, buff=0.15)
        Group(
            self.input, self.conv_layer, self.pooling_layer,
            Group(self.flattened_vector, self.dense_layer, self.softmax),
            self.output_layer
        ).arrange(buff=horizontal_spacing)
        self.add(
            self.input, self.conv_layer, self.pooling_layer,
            self.flattened_vector, self.dense_layer, self.softmax,
            self.output_layer
        )

        # Create Labels
        self.input_title = LayerTitle('Input').next_to(self.input, UP)
        self.dense_layer_title = LayerTitle('Dense','layer').next_to(self.dense_layer, UP, buff=0.5)
        self.conv_layer_title = LayerTitle('Convolutional','layer').next_to(self.conv_layer, UP).align_to(self.dense_layer_title, UP)
        self.pooling_layer_title = LayerTitle('Pooling','layer').next_to(self.pooling_layer, UP).align_to(self.dense_layer_title, UP)
        self.output_layer_title = LayerTitle('Output').next_to(self.output_layer, UP)
        self.add(self.input_title, self.conv_layer_title, self.pooling_layer_title, self.dense_layer_title, self.output_layer_title)

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
    
    def get_layer_highlight(self, layer_to_highlight: int, stroke_width=4, color=GOLD,):
        match layer_to_highlight:
            case 0:
                layer = Group(self.conv_layer, self.conv_layer_title)
            case 1:
                layer = Group(self.pooling_layer, self.pooling_layer_title)
            case 2:
                layer = Group(self.dense_layer, self.dense_layer_title, self.flattened_vector, self.softmax)
            case _:
                raise ValueError()
            
        highlight = RoundedRectangle(
            width = layer.width + 2*0.35,
            height = 8,
            color=color, stroke_width=stroke_width, fill_opacity=0,
            corner_radius=0.25
        ).set_z_index(20)
        highlight.move_to(layer).align_to(self.dense_layer_title.get_top() + 0.5*UP, UP)

        return highlight

    def restore(self):
        for mob in self.submobjects:
            mob.restore()
    
    def save_state(self):
        for mob in self.submobjects:
            mob.save_state()