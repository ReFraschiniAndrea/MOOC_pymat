import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from mooc_utils import *
import skimage
from PIL import Image
from W4Anim import ReferenceSystemImageMobject, DiscreteConvolutionPseudoCode, separable_box_blur

# Highly suggested to run with --disable_caching
config.update(RELEASE_CONFIG)

NUMBERS_COLOR = RED
HIGHLIGHT_COLOR = BLUE
INDICATE_COLOR = BLUE_B

class W4Theory_slides(MOOCSlide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # IMAGE OF CAR PLATE APPEARS
        self.next_slide(
            notes=
            '''Nowadays, for privacy reasons, we often need to hide certain
            details in photos - like faces or car plates.
            '''
        )
        car_plate_array = np.array(Image.open(r'Assets\W4\car_plate.png'))  # uint8
        car_plate_original = ImageMobject(car_plate_array)
        self.play(FadeIn(car_plate_original))

        # SLIDE 02:  ===========================================================
        # IMAGE GETS BLURRED AND GOES BACK TO NORMAL
        self.next_slide(
            notes=
            '''This is usually done using a particular filter, the so-called
            blurring effect. But how does blur actually work? Did you know there
            are matrices behind it? Let's have a look together.
            '''
        )
        cp_blur = (skimage.filters.gaussian(car_plate_array, sigma=40, mode='reflect', channel_axis=-1)*255).astype(np.uint8)
        car_plate_blur = ImageMobject(cp_blur)
        self.play(FadeIn(car_plate_blur), runtime=2)
        self.wait(3)
        self.play(FadeOut(car_plate_blur), runtime=2)

        # SLIDE 03:  ===========================================================
        # BACK TO ORIGINAL IMAGE
        # IMAGE IS CONVERTED TO GRAYSCALE
        self.next_slide(
            notes=
            '''Let us consider this grayscale picture of the car plate. Where is
            the matrix?
            '''
        )
        cp_gray = (skimage.color.rgb2gray(car_plate_array)*255).astype(np.uint8)
        car_plate_grayscale = ImageMobject(cp_gray)
        self.play(ReplacementTransform(car_plate_original, car_plate_grayscale))

        # SLIDE 04:  ===========================================================
        # SMALL PORTION OF THE IMAGE IS HIGHLIGHTED
        # THE REST DISAPPEARS WHILE THE PORTION IS ZOOMED
        # PORTION IS REPLACED BY A 12 x 12 DOWNSCALED VERSION
        self.next_slide(
            notes=
            '''For simplicity, let's focus on this small portion and downscale
            it. This grayscale image is composed of 144 tiny squares called
            pixels, organized in a 12 by 12 grid.
            '''
        )
        # Add reference system to the image
        referenced_car_plate = ReferenceSystemImageMobject(car_plate_grayscale)
        por_xmin, por_ymin, por_L = 248, 108, 336
        portion = cp_gray[por_xmin:por_xmin+por_L, por_ymin:por_ymin+por_L]
        portion_image = ImageMobject(portion).scale_to_fit_height(car_plate_grayscale.height * portion.shape[0]/car_plate_array.shape[0])
        portion_image.move_to(referenced_car_plate.axes.c2p(por_ymin, por_xmin, 0), aligned_edge=UL)
        self.add(portion_image)
        
        self.play(Circumscribe(portion_image, color=HIGHLIGHT_COLOR), runtime=2)
        self.play(FadeOut(car_plate_grayscale), run_time=0.5)
        self.play(portion_image.animate.scale_to_fit_height(0.75*FRAME_HEIGHT).center(), run_time=0.5)

        downscaled_portion = (skimage.transform.resize(portion, (12, 12))) # float64
        downscaled_portion = ((downscaled_portion - downscaled_portion.min())/(downscaled_portion.max()-downscaled_portion.min())*255).astype(np.uint8)
        sample_12: PixelArray = PixelArray(downscaled_portion, stroke_width=0.75).set_height(portion_image.height)

        self.play(
            AnimationGroup(
                *[GrowFromCenter(pixel) for pixel in sample_12.pixel_array],
                lag_ratio=1,
                run_time=1
            )
        )
        self.remove(portion_image)

        # SLIDE 05:  ===========================================================
        # GRAYSCALE VALUES ARE WRITTEN ON TOP OF THE PIXELS OF THE IMAGE
        self.next_slide(
            notes=
            '''To each pixel of the image, we associate a number, where 0 is
            black, 255 is white and in between we have different shades of gray.
            '''
        )
        sample_12.add_pixel_values(color=NUMBERS_COLOR)
        self.play(
            AnimationGroup(
                Create(sample_12.pixel_values),
                lag_ratio=0.1
            )
        )

        # SLIDE 06:  ===========================================================
        # GRAYSCALE VALUES ARE MOVED TO THE SIDE AND ENCLOSED BY SQUARE BRACKETS
        self.next_slide(
            notes=
            '''At this point, the image can be represented with a matrix. Each
            entry is the grayscale value of the corresponding pixel. In
            particular, each pixel and its value can be uniquely identified by
            its row and column position.
            '''
        )
        sample_12.pixel_array.save_state()
        sample_12.pixel_values.save_state()
        self.play(
            AnimationGroup(
                sample_12.pixel_array.animate.move_to(HALF_SCREEN_LEFT).scale(0.6),
                sample_12.pixel_values.animate.move_to(HALF_SCREEN_RIGHT).scale(0.6),
            )
        )

        sample_12.add_brackets(color=BLACK)
        self.play(
            sample_12.pixel_values.animate.set_color(BLACK),
            FadeIn(sample_12.brackets)
        )

        # SLIDE 07:  ===========================================================
        # PIXEL IN THE IMAGE IS HIGHLIGHTED
        # CORRESPONDING ROW/COLUMN IN MATRIX ARE HIGHLITED IN SUCCESSION
        # I, J LABELS APPEAR FOR ROW/COLUMN
        self.next_slide(
            notes=
            '''For instance, this pixel is row x and column y.
            '''
        )
        qi, qj = 6, 6  # pixel used for example
        pixel_highlight = sample_12.get_pixel_highlight(position=(qi, qj), color=HIGHLIGHT_COLOR)
        
        row_highlight = HighlightRectangle(sample_12.pixel_values.get_row(qi), color=BLUE)
        column_highlight = HighlightRectangle(sample_12.pixel_values.get_column(qj), color=ORANGE)
        x_label = MathTex('x', color=BLACK).next_to(row_highlight, LEFT, buff=0.75)
        y_label = MathTex('y', color=BLACK).next_to(column_highlight, UP)

        self.play(Create(pixel_highlight))
        self.play(
            Create(row_highlight),
            Write(x_label)
        )
        self.play(
            Create(column_highlight),
            Write(y_label)
        )

        # SLIDE 08:  ===========================================================
        # GRAYSCALE VALUES MOVE BACK ON TOP OF THE IMAGE
        # A SINGLE PIXEL IS HIGHLIGHTED
        # ITS VALUE CHANGES TO 255-0-ORIGINAL TO SHOWCASE EFFECT
        self.next_slide(
            notes=
            '''To understand how to blur an image, it is important to highlight
            that changing the values of the matrix affects the image.
            '''
        )
        pi, pj = 5, 8  # pixel used for further examples
        self.play(FadeOut(pixel_highlight, sample_12.brackets, column_highlight, row_highlight, x_label, y_label))
        self.play(sample_12.pixel_array.animate.restore(), sample_12.pixel_values.animate.restore())
        sample_12.remove_brackets()

        pixel_to_highlight : Square = sample_12.pixel_array[pi, pj]
        pixel_highlight.match_height(pixel_to_highlight).set_stroke(width=4).move_to(pixel_to_highlight)
        value_to_highlight : CustomDecimalNumber = sample_12.pixel_values[pi, pj]
        tracker = ValueTracker(value_to_highlight.get_value())
        value_to_highlight.add_updater(
            lambda v: v.set_value(tracker.get_value()).move_to(pixel_to_highlight)
        )
        pixel_to_highlight.save_state()
        tracker.save_state()
        
        self.play(Create(pixel_highlight))
        self.wait(0.5)
        self.play(pixel_to_highlight.animate.set_color(WHITE), tracker.animate.set_value(255), run_time=0.5)
        self.wait(0.5)
        self.play(pixel_to_highlight.animate.set_color(BLACK), tracker.animate.set_value(0), run_time=0.5)
        self.wait(0.5)
        self.play(pixel_to_highlight.animate.restore(), tracker.animate.set_value(sample_12.array[pi, pj]), run_time=0.5)
        
        value_to_highlight.clear_updaters()
        
        # SLIDE 09:  ===========================================================
        # HIGHLIGHTED THE 3 X 3 NEIGHBOURS BY CREATING THE KERNEL
        self.next_slide(
            notes=
            '''Indeed, the easiest way of blurring a portion of the image is to
            average out the values of the corresponding pixels.
            '''
        )
        self.play(FadeOut(pixel_highlight))
        kernel = np.ones((3,3))/9
        kernel_highlight = sample_12.get_kernel_array(kernel, kernel_color=HIGHLIGHT_COLOR, kernel_stroke_width=4, add_values=False).move_to(sample_12.pixel_array[pi, pj])
        kernel_highlight.set_z_index(1)
        self.play(Create(kernel_highlight))

        # SLIDE 10:  ===========================================================
        # AVERAGE FORMULA IS WRITTEN ONE TERM AT A TIME,
        # WHILE CORRESPONDING PIXELS ARE HIGHLIGHTED
        self.next_slide(
            notes=
            '''For instance we replace the pixel with the average of the
            neighbours.
            '''
        )
        # write formula on top and highlight terms while writing
        sample_12.pixel_values.set_z_index(1)  # need to have numbers on top
        initial =  r"\frac{1}{9}\Big(" + " + ".join([f"v_{i}" for i in range(1, 10)]) + r"\Big)"
        substrings_to_isolate = [f"v_i" for i in range(1,10)] + [r"\frac{1}{9}", "+", r"\Big(", r"\Big)"]
        expr_initial = MathTex(
            initial,
            substrings_to_isolate=substrings_to_isolate,
            color=BLACK
        ).to_edge(UP).shift(UP*0.5)

        self.play(VGroup(sample_12, kernel_highlight).animate.set_height(0.6*FRAME_HEIGHT).shift(DOWN))
        dt_per_char = 0.15
        self.play(
            AnimationGroup(
                Write(
                    expr_initial,
                    rate_func=linear,
                    run_time= 20 * dt_per_char
                ),
                Succession(
                    *[Indicate(sample_12.pixel_array[pi-1 + i, pj-1+j], scale_factor=1, color=INDICATE_COLOR) for i in range(3) for j in range(3)],
                    lag_ratio=0.5,
                    run_time = 16 *dt_per_char,
                ),
                lag_ratio=0.2,
            )
        )

        # SLIDE 11:  ===========================================================
        # VALUES IN THE FORMULA ARE REPLACED WITH THE CORRESPONDING PIXELS
        # RESULT APPEARS AND IS PUT IN THE FIANL IMAGE
        self.next_slide(
            notes=
            '''For the given pixel (X,Y) we replace its value with the average
            of these 9 values. This operation has to be repeated for each pixel!
            How can we automate it?
            '''
        )
        # Replace v_i with squares form the image
        v_terms = VGroup(*expr_initial[2:19:2])
        phony_length = 0.6
        pixels_to_sum = VGroup()
        for i in range(3):
            for j in range(3):
                pixel_copy = sample_12.pixel_array[pi-1 + i, pj-1+j].copy()
                pixel_copy.set_stroke(opacity=0).scale(0.91)
                pixel_copy.add(sample_12.pixel_values[pi-1 + i, pj-1+j].copy())
                pixel_copy.set_z_index(2)
                pixels_to_sum.add(pixel_copy)

        self.play(
            AnimationGroup(
                *[AnimationGroup(
                    FadeOut(v_terms[i]),
                    pixels_to_sum[i].animate.scale_to_fit_height(phony_length).move_to(v_terms[i]).match_y(expr_initial[3]),
                    lag_ratio=0.5
                ) for i in range(9)],
                lag_ratio = 0.2,
                runt_time=2
            )
        )

        # Create the blur result (show initially empty)
        blur_array = separable_box_blur(downscaled_portion, 3).astype(np.uint8)
        blur_result : PixelArray = PixelArray(blur_array[1:-1, 1:-1], stroke_color=WHITE, stroke_width=1).set_height(10/12*sample_12.height)
        blur_result.set_fill(opacity=0)
        blur_result.move_to(HALF_SCREEN_RIGHT).shift(DOWN)
        back = Square(blur_result.height, stroke_width=0).set_fill(color=LIGHTER_GRAY, opacity=1).move_to(blur_result).set_z_index(-1)
        blur_result.add(back)
        
        # Formula shifts slightly right, then resulting pixel appears
        self.play(VGroup(expr_initial, pixels_to_sum).animate.shift(LEFT*0.5))

        equal = MathTex("=", color=BLACK).next_to(expr_initial[-1], RIGHT)
        result_pixel = pixels_to_sum[0].copy().next_to(equal, RIGHT)
        cl = blur_array[pi-1, pj-1]
        result_pixel.set_fill(color=rgb_to_color((cl, cl, cl)), family=False)  # exlude the number
        result_pixel[1].set_value(cl).move_to(result_pixel).set_z_index(5)  # the order gets inverrted for some reason, bring back to front
        self.play(FadeIn(equal, result_pixel))

        # Put the pixel in the resulting image
        self.play(
            AnimationGroup(
                VGroup(sample_12, kernel_highlight).animate.move_to(HALF_SCREEN_LEFT).shift(DOWN),
                FadeIn(back, blur_result),
                lag_ratio=0.5
            )
        )
        self.play(
            result_pixel.animate.move_to(blur_result.pixel_array[4, 7]).set_stroke(width=1, opacity=0, family=False).match_height(blur_result.pixel_array[pi-1, pj-1])
        )

        # SLIDE 12:  ===========================================================
        # THE 1/9 COEFFICIENTS IS DISTRIBUTED TO ALL TERMS IN THE AVERAGE FORMULAS
        # THE 1/9 ARE ORGANIZED INTO A 3 X 3 KERNEL
        self.next_slide(
            notes=
            '''It's convenient to organize the weights used to compute the
            average in a 3 by 3 matrix. This matrix is called kernel or filter,
            or, to be more precise, it's the kernel rotated by 180 degrees.
            '''
        )
        # Distribute the coefficients
        expr_distributed = VGroup(
            obj for _ in range(9) for obj in
            [MathTex(r'\frac{1}{9}', color=BLACK), Square(phony_length), MathTex('+', color=BLACK)]
            )
        expr_distributed.remove(expr_distributed[-1])
        expr_distributed.arrange(buff=0.1).match_y(expr_initial)
        one_over_nines = expr_distributed[::3]
        phony_squares = expr_distributed[1::3]
        pluses_new = expr_distributed[2::3]
        pluses_old = expr_initial[3:18:2]
        parenthesis = VGroup(expr_initial[1], expr_initial[-1])

        self.add(v_terms); self.remove(v_terms)  # they are still behind the rectangles
        self.play(
            AnimationGroup(
                FadeOut(equal, parenthesis),
                AnimationGroup(
                    ReplacementTransform(expr_initial[0], one_over_nines[0][0]),
                    *[pixels_to_sum[i].animate.move_to(phony_squares[i]) for i in range(len(pixels_to_sum))],
                    *[pluses_old[i].animate.move_to(pluses_new[i]) for i in range(len(pluses_old))],
                ),
                FadeIn(one_over_nines[1:]),
                lag_ratio=0.5
            )
        )

        # get things out of the way to make space
        self.play(
            FadeOut(pixels_to_sum, pluses_old),
            VGroup(sample_12, kernel_highlight, blur_result, result_pixel).animate.shift(DOWN)
        )

        # Finally, transform the 1/9s in a kernel
        kernel = np.ones((3,3))/9
        kernel_array, kernel_values = sample_12.get_kernel_array(kernel, kernel_color=BLACK, kernel_stroke_width=4, add_values=True, kernel_tex = " 1 / 9", values_size_fator=0.5)
        VGroup(kernel_array, kernel_values).scale(2).center().to_edge(UP).shift(UP*1.5)
        kernel_values.set_z_index(2)

        self.play(AnimationGroup(TransformMatchingShapes(one_over_nines[i], kernel_values[i]) for i in range(9)))
        self.play(FadeIn(kernel_array))

        # SLIDE 13:  ===========================================================
        # KERNEL IS POSITIONED ON TOP OF PREVIOUSLY HIGHLIGHTED AREA,
        # SO THAT VALUE*WEIGHT IS ON TOP OF EACH PIXEL
        self.next_slide(
            notes=
            '''If we position this kernel over the image centered on the pixel
            (x,y), we can perform the average by doing element wise
            multiplication of the two matrices and summing the result up.
            '''
        )
        self.play(FadeOut(blur_result, result_pixel))

        # Empty 3 by 3 to store kernel multiplication result
        local_result_array = PixelArray(downscaled_portion[pi-1:pi+2, pj-1:pj+2]/9, stroke_color=BLACK, stroke_width=4)
        local_result_array.scale_to_fit_height(kernel_array.height).move_to(HALF_SCREEN_RIGHT).shift(DOWN*2)
        local_result_array.pixel_array.set_fill(opacity=0)
        local_result_array.add_pixel_values(num_decimal_places=1, color=BLACK)
        local_result_array.pixel_values.set_opacity(0)

        # Create jig to illustrate procedure: 3 squares + 2 connecting rectangles
        for numbers in [sample_12.pixel_values, kernel_values, local_result_array.pixel_values]:
            numbers.set_z_index(2)  # so that the jig is below
        high1 : Square = sample_12.get_pixel_highlight(position=(pi-1, pj-1), color=HIGHLIGHT_COLOR).set_fill(color=HIGHLIGHT_COLOR, opacity=0.4)
        high3 : Square = local_result_array.get_pixel_highlight(position=(0,0), color=HIGHLIGHT_COLOR, stroke_width=6).set_fill(color=HIGHLIGHT_COLOR, opacity=0.4)
        high2 : Square = high3.copy().move_to(kernel_array[0,0])

        hv1, hv2, hv3 = high1.get_vertices(), high2.get_vertices(), high3.get_vertices()
        poly_1 = Polygon(hv1[1], hv1[0], hv1[3], hv2[3], hv2[2], hv2[1], fill_color=HIGHLIGHT_COLOR, fill_opacity=0.4, stroke_width=0)
        poly_2 = Polygon(hv2[0], hv2[3], hv2[2], hv3[2], hv3[1], hv3[0], fill_color=HIGHLIGHT_COLOR, fill_opacity=0.4, stroke_width=0)
        def poly_1_updater(m: Polygon):
            hv1, hv2 = high1.get_vertices(), high2.get_vertices()
            m.become(
                Polygon(
                hv1[1], hv1[0], hv1[3], hv2[3], hv2[2], hv2[1],
                fill_color = m.fill_color, fill_opacity=m.fill_opacity, stroke_width=m.stroke_width
                )
            )
        def poly_2_updater(m: Polygon):
            hv2, hv3 = high2.get_vertices(), high3.get_vertices()
            m.become(
                Polygon(
                hv2[0], hv2[3], hv2[2], hv3[2], hv3[1], hv3[0],
                fill_color = m.fill_color, fill_opacity=m.fill_opacity, stroke_width=m.stroke_width
                )
            )
        poly_1.add_updater(poly_1_updater)
        poly_2.add_updater(poly_2_updater)
        
        self.play(FadeIn(local_result_array.pixel_array))
        self.play(FadeIn(high1, high2, poly_1))
        self.play(FadeIn(high3, poly_2))
        self.play(
            Succession(
                Succession(
                    AnimationGroup(
                        high1.animate.move_to(sample_12.pixel_array[pi-1+i, pj-1+j]),
                        high2.animate.move_to(kernel_array[i,j]),
                        high3.animate.move_to(local_result_array.pixel_array[i,j]),
                        run_time=0.3
                    ),
                    local_result_array.pixel_values[i,j].animate(run_time=0.1).set_opacity(1),
                    Wait(0.1)
                )
                for i in range(3) for j in range(3)
            )
        )

        result_pixel.match_height(local_result_array.pixel_array[0, 0]).next_to(local_result_array, DOWN)
        self.play(FadeIn(result_pixel, shift=DOWN))

        # SLIDE 14:  ===========================================================
        # SLIDING KERNEL WHILE BLURRED IMAGE FILLS IN (3b1b ANIMATION)
        self.next_slide(
            notes=
            '''At this point, we slide the kernel over the internal pixels of
            the original image and we build the blurred image.
            '''
        )
        # get everything in the correct position
        self.play(FadeOut(local_result_array, result_pixel, high1, high2, high3, poly_1, poly_2, kernel_highlight))

        blur_result.move_to(HALF_SCREEN_RIGHT)
        pixel_highlight = blur_result.get_pixel_highlight(color=HIGHLIGHT_COLOR, stroke_width=4)
        kernel_array = VGroup(kernel_array, kernel_values)  # CAREFUL!!!
        self.play(
            sample_12.pixel_array.animate.move_to(HALF_SCREEN_LEFT),
            sample_12.pixel_values.animate.move_to(HALF_SCREEN_LEFT).set_opacity(0),
            kernel_array.animate.match_height(kernel_highlight).set_color(HIGHLIGHT_COLOR).move_to(sample_12.pixel_array[1,1]).shift(UP*2),
            FadeIn(blur_result),
        )
        self.play(FadeIn(pixel_highlight))

        # Setup main tracker and the updaters for the slide
        index_tracker = ValueTracker(0)
        def get_index(): t =int(index_tracker.get_value()); return 13 + t + 2*(t//10)
        kernel_array.add_updater(lambda m: m.move_to(sample_12.pixel_array[get_index()]))
        pixel_highlight.add_updater(lambda m: m.move_to(blur_result.pixel_array[int(index_tracker.get_value())]))
        blur_result.pixel_array.add_updater(lambda m: m[int(index_tracker.get_value())].set_fill(opacity=1))

        self.play(index_tracker.animate.set_value(99), run_time=5, rate_func=linear)

        # At the end the effect disappears, so we need to reapply it
        blur_result.pixel_array.set_fill(opacity=1)
        kernel_array.clear_updaters()
        blur_result.pixel_array.clear_updaters()
        pixel_highlight.clear_updaters()

        # SLIDE 15:  ===========================================================
        # DISCRETE CONVOLUTION TITLE APPEARS
        # MOVE TERMS S.T. ORIGINAL IMAGE * CONVOLUTION KERNEL = BLURRED IMAGE
        self.next_slide(
            notes=
            '''This process is called discrete convolution between the large
            matrix A, representing the image, and the small matrix, 3X3, the
            rotated kernel, or filter, called K.
            '''
        )

        DC_title = Text('Discrete Convolution', font_size=64, color=BLACK, font=SANS_SERIF_FONT, weight=LIGHT).to_edge(UP).shift(UP*1)
        self.play(Write(DC_title))

        # Compute final arrangemente with phony targets
        star = MathTex("*", color=BLACK)
        equal = MathTex("=", color=BLACK)
        phony1 = Square(sample_12.height*0.7)
        phony2 = Square(blur_result.height*0.7)
        phony3 = Square(kernel_array.height*2)
        VGroup(phony1, star, phony3, equal, phony2).arrange(buff=0.5)
        A_label = MathTex("A", color=BLACK).next_to(phony1, DOWN, buff=1)
        K_label = MathTex("K", color=BLACK).next_to(phony3, DOWN).align_to(A_label, DOWN)

        self.play(FadeOut(pixel_highlight))
        self.play(
            sample_12.animate.match_height(phony1).move_to(phony1),
            blur_result.animate.match_height(phony2).move_to(phony2),
            kernel_array.animate.match_height(phony3).move_to(phony3).set_color(BLACK)
        )
        self.play(FadeIn(star, equal))
        self.play(Write(A_label), Write(K_label))

        # SLIDE 16:  ===========================================================
        # IMAGES DISAPPEAR, START WRITING ALGORITHM
        self.next_slide(
            notes=
            '''The algorithm to compute each filtered pixel looks like this:
            '''
        )
        self.play(FadeOut(DC_title, blur_result, sample_12, kernel_array, star, equal, A_label, K_label))
        pc = DiscreteConvolutionPseudoCode()
        self.play(Write(pc[0]))

        # SLIDE 17:  ===========================================================
        # REQUIRE LINE IS WRITTEN
        self.next_slide(
            notes=
            '''We need the image A, the kernel K and the position (i, j) of the
            pixel that we want to filter.
            '''
        )
        self.play(Write(pc[1]))

        # SLIDE 18:  ===========================================================
        # DOUBLE FOR LOOP IS WRITTEN
        self.next_slide(
            notes=
            '''We loop over the rows and the columns of the kernel, ...
            '''
        )
        self.play(Write(pc[2]))
        self.play(Write(pc[3]), Write(pc[4]))

        # SLIDE 19:  ===========================================================
        # CONVOLUTION LINE IS WRITTEN
        self.next_slide(
            notes=
            '''...to assemble the element wise multiplication of the kernel and
            the portion of the image around the pixel (i,j).
            '''
        )
        self.play(Write(pc[5]))
        self.play(Write(pc[6]), Write(pc[7]))

        # SLIDE 20:  ===========================================================
        # HIGHLIGHT THE ACCUMULATES SUM
        self.next_slide(
            notes=
            '''As we go, we sum these products up in the variable v.
            '''
        )
        partial_sum_highlight = HighlightRectangle(pc[2][2:])
        self.play(Create(partial_sum_highlight))

        # SLIDE 21:  ===========================================================
        # HIGHLIGHT THE INDEXED MATRICES
        self.next_slide(
            notes=
            '''It is worth focusing on the indices here.
            '''
        )
        indices_highlight = HighlightRectangle(pc[5][6:])
        self.play(ReplacementTransform(partial_sum_highlight, indices_highlight))

        # SLIDE 22:  ===========================================================
        # 3 X 3 IMAGE AND KERNEL APPEAR
        # CENTER PIXEL IS HIGHLIGHTED, WITH (i, j) LABELS
        self.next_slide(
            notes=
            '''The idea is to go around the pixel (i,j).
            '''
        )
        indices_line = pc[5][6:].copy()
        indices_line.save_state()
        self.add(indices_line)
        pc[5][6:].set_opacity(0)


        three_by_three: PixelArray = PixelArray(sample_12.array[:3,:3], stroke_width=2, stroke_color=WHITE).set_height(0.45*FRAME_HEIGHT)
        # kernel_array = three_by_three.get_kernel_array(kernel, add_values=True, kernel_tex = " 1 / 9", values_size_fator=0.5, kernel_color=BLACK)
        kernel_array.set_color(BLACK).match_height(three_by_three)
        VGroup(three_by_three, kernel_array).arrange(buff=1).shift(DOWN*0.5)
        A_label.next_to(three_by_three, DOWN)
        K_label.next_to(kernel_array, DOWN)

        ij_labels = VGroup(MathTex(lab, color=BLACK) for lab in ['i-1', 'j-1', 'i', 'j'])
        ij_labels[0].next_to(three_by_three.pixel_array[0,0], LEFT)
        ij_labels[1].next_to(three_by_three.pixel_array[0,0], UP)
        ij_labels[2].next_to(three_by_three.pixel_array[1,0], LEFT).match_x(ij_labels[0])
        ij_labels[3].next_to(three_by_three.pixel_array[0,1], UP).match_y(ij_labels[1])

        self.play(
            AnimationGroup(
                FadeOut(pc, indices_highlight),
                indices_line.animate.scale(1.3).center().to_edge(UP).shift(UP*0.5),
                FadeIn(three_by_three, kernel_array, A_label, K_label),
                lag_ratio=0.5
            )
        )
        self.play(
            Indicate(three_by_three.pixel_array[1,1], color=INDICATE_COLOR, scale_factor=1, run_time=2),
            Write(ij_labels[2]), Write(ij_labels[3])
        )

        # SLIDE 23:  ===========================================================
        # I, J, M, N COUNTERS APPEAR AND FIRST PIXEL IS HIGHLIGHTED
        self.next_slide(
            notes=
            '''This means that, in the matrix A, the pixel in the left corner
            has position i-1, j-1, ...
            '''
        )
        pixel_highlight = three_by_three.get_pixel_highlight(color=HIGHLIGHT_COLOR, stroke_width=6)

        self.play(Create(pixel_highlight), Write(ij_labels[0]), Write(ij_labels[1]))

        # SLIDE 24:  ===========================================================
        # M, N COUNTERS APPEAR NAD FIRST KERNEL ELEMENT HIGHLIGHTED
        self.next_slide(
            notes=
            '''...corresponding to m=0, n=0 in the kernel.
            '''
        )
        m_counter = Variable(var=0, label='m', var_type=Integer).set_color(BLACK)
        n_counter = Variable(var=0, label='n', var_type=Integer).set_color(BLACK)
        VGroup(m_counter, n_counter).arrange(buff=1).next_to(kernel_array, UP).match_y(ij_labels[1])
        kernel_highlight = pixel_highlight.copy().move_to(kernel_array[0][0,0])

        self.play(Create(kernel_highlight), Write(m_counter), Write(n_counter))

        # SLIDE 25:  ===========================================================
        # PIXEL SLIDES IN BOTH IMAGE AND KERNEL WHILE UPDATING INDICES
        # FIRST DOWN, THEN RIGHT
        self.next_slide(
            notes=
            '''Increasing m we move down, increasing n we move right in both
            matrices.
            '''
        )
        side_length = three_by_three.pixel_array[0,0].height
        pixel_highlight.add_updater(
            lambda m: m.move_to(three_by_three.pixel_array[0,0].get_center() + side_length*(DOWN*m_counter.tracker.get_value() + RIGHT*n_counter.tracker.get_value()))
        )
        kernel_highlight.add_updater(
            lambda m: m.move_to(kernel_array[0][0,0].get_center() + side_length*(DOWN*m_counter.tracker.get_value() + RIGHT*n_counter.tracker.get_value()))
        )

        self.play(
            Succession(
                ApplyMethod(m_counter.tracker.set_value, 1, run_time=0.5),
                Wait(0.5),
                ApplyMethod(m_counter.tracker.set_value, 2, run_time=0.5),
                Wait(0.5),
                ApplyMethod(n_counter.tracker.set_value, 1, run_time=0.5),
                Wait(0.5),
                ApplyMethod(n_counter.tracker.set_value, 2, run_time=0.5),
            )
        )

        # SLIDE 26:  ===========================================================
        # PSEUDO-CODE REAPPEARS
        self.next_slide(
            notes=
            '''Keep in mind this algorithm has to be performed for each internal
            pixel of the original image! We need a computer code to do it
            automatically, and in the next videos we will learn how to write it.
            '''
        )
        self.play(FadeOut(three_by_three, kernel_array, pixel_highlight, kernel_highlight, m_counter, n_counter, ij_labels, A_label, K_label))
        self.play(
            AnimationGroup(
                indices_line.animate.restore(),
                FadeIn(pc),
                lag_ratio=0.5
            )
        )