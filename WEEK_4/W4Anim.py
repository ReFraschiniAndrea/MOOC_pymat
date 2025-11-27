from manim import *
from PIL import Image
import scipy.ndimage
import skimage.transform
from typing import Sequence
from mooc_utils import PixelArray, HALF_SCREEN_LEFT, HALF_SCREEN_RIGHT

SATURATED_RED = ManimColor("#FF1500")
SATURATED_BLUE = ManimColor("#00D0FF")

class DiscreteConvolutionPseudoCode(Tex):
    def __init__(self):
        super().__init__(
            r"{{\textbf{Algorithm:} Local Convolution \newline}}"
            r"{{\textbf{Require:} $ A, K, (i, j) $ \newline}}"
            r"{{1: $v \leftarrow 0$ \newline}}"
            r"{{2: \textbf{for} $m \leftarrow 0 $ to $2$ \textbf{do}: \newline}}"
            r"{{3: \quad \textbf{for} $n \leftarrow 0$ to $2$ \textbf{do}: \newline}}"
            r"{{4: \quad \quad $v \leftarrow v + A(i-1 +m, j-1 + n) \times K(m, n)$ \newline}}"
            r"{{5: \quad \textbf{end for} \newline}}"
            r"{{6: \textbf{end for}}}",
            color=BLACK
        )
        for i in range(len(self)):
            self[i].align_on_border(LEFT)
        self.center()

class ReferenceSystemImageMobject(Group):
    def __init__(self, image: ImageMobject):
        self.image = image
        height, width = self.image.get_pixel_array().shape[:2]
        self.axes = Axes(
            [0, width], [0, -height], self.image.width, self.image.height,
            tips=False, axis_config={'include_ticks':False, 'stroke_width':0},  y_axis_config={"scaling": LinearBase(scale_factor=-1)})
        self.axes.move_to(self.image, aligned_edge=UL)
        self.axes.set_opacity(0)
        super().__init__(self.image, self.axes)
        
    def c2p(self,  *coords: float | Sequence[float] | Sequence[Sequence[float]] | np.ndarray):
        return self.axes.c2p(coords)

def separable_box_blur(
    input: np.ndarray,
    kernel_size: int = 3,
    **kwargs
) -> np.ndarray:
    """2D box blur filter written as two subsequent 1D convolutions."""
    output = np.empty(input.shape)
    kernel1d = np.ones(kernel_size)/kernel_size
    scipy.ndimage.correlate1d(input, kernel1d, 0, output, **kwargs)
    scipy.ndimage.correlate1d(output, kernel1d, 1, output, **kwargs)
    return output

class LiveConvolution():
    """Class for managing live convolution animation.
    Index is referred to the resulting image
    """
    def __init__(
        self,
        source_image: PixelArray,
        result_image: PixelArray,
        sliding_kernel: VMobject,
        pixel_highlight : VMobject,
        kernel_size: int = 3,
        move_images=True,
        background_color=None
    ):
        self.source_image = source_image
        self.result_image = result_image
        self.sliding_kernel = sliding_kernel
        self.pixel_highlight = pixel_highlight

        self.n_ = kernel_size
        self.index_tracker = ValueTracker(0)

        # Set its pixels to transparent
        self.result_image.pixel_array.set_fill(opacity=0)

        if background_color is not None:
            back = Square(self.result_image.height, stroke_width=0).set_fill(color=background_color, opacity=1).move_to(self.result_image).set_z_index(-1)
            self.result_image.add(back)
        
        if move_images:
            self.source_image.move_to(HALF_SCREEN_LEFT)
            self.result_image.move_to(HALF_SCREEN_RIGHT)
            kernel_radius = (self.n_ - 1)//2
            self.sliding_kernel.move_to(self.source_image.pixel_array[kernel_radius, kernel_radius])
            self.pixel_highlight.move_to(self.result_image.pixel_array[0,0])

    def setup(self, full_result=False):
        # setup updaters
        def get_result_index(): return int(self.index_tracker.get_value())
        if full_result:  # source and result have same size:
            get_source_index = get_result_index
        else:
            self.index_tracker.set_value(0)
            row_source = self.source_image.array.shape[1]
            row_result = self.result_image.array.shape[1]
            kernel_radius = (self.n_ - 1)//2
            def get_source_index(): t =int(self.index_tracker.get_value()); return kernel_radius*(row_source+1) + t + 2*kernel_radius*(t//row_result)
        
        self.sliding_kernel.add_updater(lambda m: m.move_to(self.source_image.pixel_array[get_source_index()]))
        self.pixel_highlight.add_updater(lambda m: m.move_to(self.result_image.pixel_array[get_result_index()]))
        self.result_image.pixel_array.add_updater(lambda m: m[get_result_index()].set_fill(opacity=1))

        self.sliding_kernel.update()
        self.pixel_highlight.update()

    def SlideKernel(self, end, run_time=1) -> Animation:
        return self.index_tracker.animate(run_time=run_time, rate_func=linear).set_value(end)

    def clear_updaters(self):
        self.sliding_kernel.clear_updaters()
        self.result_image.pixel_array.clear_updaters()
        self.pixel_highlight.clear_updaters()

    
class Kernel3X3IndexAnimation(VMobject):
    def __init__(
        self,
        input_portion,
        kernel,
        highlight_color = SATURATED_BLUE,
        highlight_stroke_width = 6
    ):
        self.three_by_three = input_portion
        self.kernel = kernel.next_to(self.three_by_three, RIGHT)
        self.kernel[1].set_z_index(1)
        self.i_labels = VGroup(MathTex(lab, color=BLACK) for lab in ['i-1', 'i', 'i+1']).arrange(DOWN ).next_to(self.three_by_three, LEFT)
        self.j_labels = VGroup(MathTex(lab, color=BLACK) for lab in ['j-1', 'j', 'j+1']).arrange(RIGHT).next_to(self.three_by_three, UP)
        for k in range(3):
            self.i_labels[k].match_y(self.three_by_three.pixel_array[k,0])
            self.j_labels[k].match_x(self.three_by_three.pixel_array[0,k])
        self.m_counter = Variable(var=0, label='m', var_type=Integer).set_color(BLACK).next_to(self.kernel[0][0,0], LEFT)
        self.n_counter = Variable(var=0, label='n', var_type=Integer).set_color(BLACK).next_to(self.kernel[0][0,0], UP).match_y(self.j_labels)
        
        VGroup(VGroup(self.three_by_three, self.i_labels, self.j_labels), VGroup(self.kernel, self.m_counter, self.n_counter)).arrange(buff=0.5)

        self.pixel_highlight = self.three_by_three.get_pixel_highlight(color=highlight_color, stroke_width=highlight_stroke_width)
        self.kernel_highlight = self.pixel_highlight.copy().move_to(self.kernel[0][0,0])
        super().__init__()
        self.add(self.three_by_three, self.kernel, self.i_labels, self.j_labels, self.m_counter, self.n_counter, self.pixel_highlight, self.kernel_highlight)

    def setup(self):
        side_length = self.three_by_three.pixel_array[0,0].height
        self.pixel_highlight.add_updater(
            lambda m: m.move_to(self.three_by_three.pixel_array[0,0].get_center() + side_length*(DOWN*self.m_counter.tracker.get_value() + RIGHT*self.n_counter.tracker.get_value()))
        )
        self.kernel_highlight.add_updater(
            lambda m: m.move_to(self.kernel[0][0,0].get_center() + side_length*(DOWN*self.m_counter.tracker.get_value() + RIGHT*self.n_counter.tracker.get_value()))
        )
        self.m_counter.add_updater(
            lambda m: m.set_y(self.kernel[0][0,0].get_y() - side_length*self.m_counter.tracker.get_value())
        )
        self.n_counter.add_updater(
            lambda m: m.set_x(self.kernel[0][0,0].get_x() + side_length*self.n_counter.tracker.get_value())
        )

    def IndexAnimation(self, slide_dt=0.5, wait_dt=0.5):
        return Succession(
            ApplyMethod(self.m_counter.tracker.set_value, 1, run_time=slide_dt),
            Wait(wait_dt),
            AnimationGroup(
                ApplyMethod(self.m_counter.tracker.set_value, 2, run_time=slide_dt),
                # Write(i_labels[2]),
            ),
            Wait(wait_dt),
            ApplyMethod(self.n_counter.tracker.set_value, 1, run_time=slide_dt),
            Wait(wait_dt),
            AnimationGroup(
                ApplyMethod(self.n_counter.tracker.set_value, 2, run_time=slide_dt),
                # Write(j_labels[2])
            )
        )
    
    def clear_updaters(self):
        for obj in (self.m_counter, self.n_counter, self.pixel_highlight, self.kernel_highlight):
            obj.clear_updaters()
               
