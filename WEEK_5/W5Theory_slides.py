import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from mooc_utils import *
from PIL import Image
from W5Anim import *

config.update(RELEASE_CONFIG)
config.max_files_cached = 200

class w5intro(Scene):
    def construct(self):
        digit = skimage.transform.resize(np.array(Image.open(r'Assets\W5\mnist8.png')), (15,15))
        digit = ((digit - digit.min())/(digit.max()-digit.min())*255).astype(np.uint8)
        digit_PA = PixelArray(digit, stroke_width=1, stroke_color=WHITE)
        digit_PA.scale_to_fit_height(0.3*FRAME_HEIGHT)

        output_layer = DigitRecognitionOutputLayer()

        input_label =  LayerTitle('Input')
        output_label = LayerTitle('Output')

        digit_recognition_algo = Paragraph('Digit Recognition\nAlgorithm', font=SANS_SERIF_FONT, weight=BOLD, font_size=48, color=DARK_BLUE, alignment='center')
        digit_recognition_algo.background_rectangle = SurroundingRectangle(
            digit_recognition_algo,
            fill_color=WHITE, fill_opacity=1, stroke_color=DARK_BLUE, stroke_width=4,
            corner_radius=0.25, buff=0.5
        )
        digit_recognition_algo.add_to_back(digit_recognition_algo.background_rectangle)
        # digit_recognition_algo.background_rectangle.set_fill(color=WHITE, opacity=1)


        VGroup(digit_PA, digit_recognition_algo, output_layer).arrange(RIGHT, buff=0.75)
        input_label.next_to(digit_PA, UP, buff=0.5)
        output_label.next_to(output_layer, UP, buff=0.5)

        left_arr = Arrow(digit_PA.get_right(), digit_recognition_algo.get_left(), color=DARK_BLUE, stroke_width=4, buff=0)
        right_arr = BrokenArrow(digit_recognition_algo.get_right(), output_layer[8].get_left(), n_turns=2, color=DARK_BLUE, add_tip=False)
        right_arr.add_tip(left_arr.tip.copy())

        self.add(digit_PA, input_label, output_layer, output_label, digit_recognition_algo, left_arr)
        self.play(Create(right_arr, run_time=0.5))
        self.play(output_layer.Activate(8), run_time=0.5)


        self.wait(0.5)
        self.play(FadeOut(output_layer, input_label, output_label, digit_recognition_algo, left_arr, right_arr))
        self.play(digit_PA.animate.scale_to_fit_height(0.75*FRAME_HEIGHT).center())
        digit_PA.add_pixel_values(color=SATURATED_RED)  # TODO: use saturated red
        self.play(Create(digit_PA.pixel_values, lag_ratio=0.1))

        self.play(
            AnimationGroup(
                digit_PA.pixel_array.animate.move_to(HALF_SCREEN_LEFT).scale(0.6),
                digit_PA.pixel_values.animate.move_to(HALF_SCREEN_RIGHT).scale(0.6),
            )
        )

        digit_PA.add_brackets(color=BLACK)
        self.play(
            digit_PA.pixel_values.animate.set_color(BLACK),
            FadeIn(digit_PA.brackets)
        )
        

class nodeexp(Scene):
    def construct(self):
        dense_layer = DenseLayer()
        dense_layer_t = LayerTitle('Dense','layer').next_to(dense_layer, UP, buff=0.5)
        self.add(dense_layer)
        self.play(VGroup(dense_layer, dense_layer_t).animate.shift(LEFT*5))

        node_scheme = NodeScheme().shift(RIGHT*2)
        node_scheme_rect = SurroundingRectangle(node_scheme, stroke_color=DARK_BLUE, stroke_width=4, fill_opacity=0, corner_radius=0.25, buff=0.5)
        # create pleasant curved arrow
        start = dense_layer.output_nodes[0].get_center()
        end = node_scheme_rect.get_left()
        alpha = Line(end, start).get_angle() -PI/2
        curve_arrow = ArcBetweenPoints(dense_layer.output_nodes[0].point_at_angle(2*alpha-PI), node_scheme_rect.get_left(), color=DARK_BLUE, stroke_width=4, angle=PI - 2*alpha)
        curve_arrow.add_tip(node_scheme.output_arrow.tip.copy().set_color(DARK_BLUE))
        # Formula below
        substrings = ['y=', r'\sigma', '(', ')', '+', 'b'] + [f'w_{i}' for i in range(3)] + [f'x_{i}' for i in range(3)]
        # NOTE: Putting spaces in the Tex strings makes numbering unreliable due to adding invisible mobjects!
        node_equation_2 = MathTex(r'y=\sigma(w_0x_0+w_1x_1+w_2x_2+b)', color=BLACK, substrings_to_isolate=substrings).next_to(node_scheme_rect, DOWN, buff=0.5)
        node_equation_1 = MathTex(r'y=w_0x_0+w_1x_1+w_2x_2+b', color=BLACK, substrings_to_isolate=substrings)
        node_equation_1.shift(node_equation_2[0].get_center()-node_equation_1[0].get_center())

        self.play(ReplacementTransform(dense_layer.output_nodes[0].copy(), node_scheme.circle))
        self.play(FadeIn(node_scheme_rect, curve_arrow))
        self.play(FadeIn(node_scheme.input_labels, node_scheme.input_arrows))
        self.play(FadeIn(node_scheme.output_label, node_scheme.output_arrow))

        self.play(FadeIn(node_scheme.weight_labels))
        self.play(FadeIn(node_scheme.sum, node_equation_1[:-2]))
        self.play(FadeIn(node_scheme.top_middle_arrow, node_scheme.middle_arrow, node_scheme.b, node_equation_1[-2:]))

        self.play(FadeIn(node_scheme.activation_function))
        self.play(TransformMatchingTex(node_equation_1, node_equation_2))
        weights_highlight = VGroup(
            HighlightRectangle(node_equation_2[3], color=GOLD),
            HighlightRectangle(node_equation_2[6], color=GOLD),
            HighlightRectangle(node_equation_2[9], color=GOLD),
        )
        weights_label = Text('Weights', font=SANS_SERIF_FONT, font_size=32,color=GOLD).next_to(weights_highlight, DOWN)
        bias_highlight =  HighlightRectangle(node_equation_2[-2], color=BLUE)
        bias_label = Text('Bias', font=SANS_SERIF_FONT, font_size=32,color=BLUE).next_to(bias_highlight, DOWN)
        self.play(FadeIn(weights_highlight, weights_label))
        self.play(FadeIn(bias_highlight, bias_label))


class adjustablecoeff(Scene):
    def construct(self):
        dense_layer = DenseLayer().shift(LEFT*5)
        dense_layer_t = LayerTitle('Dense','layer').next_to(dense_layer, UP, buff=0.5)
        self.add(dense_layer, dense_layer_t)


        NN_equation = MathTex(r'\mathbf{y} = \mathcal{NN}(\mathbf{x}; \mathbf{W}, \mathbf{b})', color=BLACK, font_size=48)
        adjustable_coeff_t =  LayerTitle('Adjustable coefficients:')
        weights_eq = MathTex(r'\mathbf{W} = [w_0,w_1,w_2,w_0,w_1,w_2,w_0,w_1,w_2,\dots]', color=BLACK, font_size=48, substrings_to_isolate=[f'w_{i}' for i in range(3)])
        bias_eq = MathTex(r'\mathbf{b} = [b,b,b,\dots]', color=BLACK, font_size=48) # isolating 'b' breaks here
        
        adj_coeff_expl = VGroup(NN_equation, adjustable_coeff_t, weights_eq, bias_eq).arrange(DOWN, buff=0.8).shift(RIGHT*1.5)
        adj_coeff_rect = SurroundingRectangle(adj_coeff_expl, stroke_color=GOLD, stroke_width=4, fill_opacity=0, buff=0.5, corner_radius=0.25)
        self.add(adj_coeff_expl, adj_coeff_rect)
        # self.add(index_labels(weights_eq), index_labels(bias_eq))
        node_colors = [RED, ManimColor("#83C167"), ORANGE]

        self.play(
            Succession(
                AnimationGroup(
                    dense_layer.output_nodes[i].animate.set_color(node_colors[i]),
                    weights_eq[1+6*i:1+6*(i+1):2].animate.set_color(node_colors[i]),
                    bias_eq[0][3+i*2].animate.set_color(node_colors[i])
                )
            for i in range(3)
            )
        )


class filterscene(Scene):
    def construct(self):
        input = skimage.transform.resize(np.array(Image.open(r'Assets\W5\mnist8.png')), (15,15))
        input = ((input - input.min())/(input.max()-input.min())*255).astype(np.uint8)
        ms = CNNDigitRecognitionScheme(input, input_label=8, pixel_size=0.1, horizontal_spacing=0.8, pooling_factor=3,
                                      highlights_kwargs ={'color': GOLD, 'stroke_width':3},
                                      outline_kwargs={'stroke_width':4})
        self.add(ms)
        ms.save_state()
        
        # for mob in ms.submobjects:
        #     print(f'z_index: {mob.z_index} ', mob)
        new_highlight_config = {'color':GOLD, 'stroke_width':6}
        new_input = ms.input.copy()
        new_conv = ms.conv_layer.top().copy()
        new_input_highlight = ms.input_highlight.copy().set(**new_highlight_config)
        new_conv_highlight = ms.conv_highlight_1.copy().set(**new_highlight_config)
        Group(new_input, new_input_highlight).scale(3).move_to(HALF_SCREEN_LEFT + 2*UP)
        Group(new_conv, new_conv_highlight).scale(3).move_to(HALF_SCREEN_RIGHT + 2*UP)

        side=1
        filter_kernel = VGroup(
            Square(side, fill_opacity=0, **new_highlight_config).add(
                MathTex(f'l_{i}', color=GOLD).scale_to_fit_height(side*0.5)
            ) for i in range(9)
        ).arrange_in_grid(3,3, buff=0).move_to(3*DOWN)

        phony_filter_square = Square(filter_kernel.width, **new_highlight_config).set_opacity(0).move_to(filter_kernel)
        new_giz1 = create_gizmo(new_input_highlight, phony_filter_square)
        new_giz2 = create_gizmo(phony_filter_square, new_conv_highlight)

        self.play(
            FadeOut(*[mob for mob in ms.submobjects if mob not in (ms.input, ms.input_highlight, ms.conv_layer, ms.conv_highlight_1)]),
            FadeOut(ms.conv_layer[1:])
        )
        self.play(
            ReplacementTransform(ms.input, new_input),
            ReplacementTransform(ms.input_highlight, new_input_highlight),
            ReplacementTransform(ms.conv_layer.top(), new_conv),
            ReplacementTransform(ms.conv_highlight_1, new_conv_highlight),
        )
        self.play(
            Create(filter_kernel, run_time=1),
            Succession(Wait(0.3),  Create(new_giz1,lag_ratio=0,run_time=0.5)),
            Succession(Wait(1),  Create(new_giz2,lag_ratio=0,run_time=0.5)),
        )
        self.wait(0.5)

        star_symbol = MathTex('*', color=BLACK, font_size=64)
        equal_symbol = MathTex('=', color=BLACK, font_size=64)

        filter_colors = [GOLD, RED, GREEN, ORANGE, PURPLE]
        phony = Group( 
            *[m for i in range(5) for m in [
                ms.input.copy().restore(),
                star_symbol.copy(),
                filter_kernel.copy().scale(0.5).set_color(filter_colors[i]),
                equal_symbol.copy(),
                ms.conv_layer[i].copy().restore()
            ]]
        ).arrange_in_grid(5,5, (1, 0.25)).center()
        self.play(
            AnimationGroup(
                FadeOut(VGroup(new_input_highlight, new_conv_highlight, new_giz1, new_giz2)),
                AnimationGroup(
                    ReplacementTransform(new_input, phony[0]),
                    ReplacementTransform(filter_kernel, phony[2]),
                    ReplacementTransform(new_conv, phony[4]),
                ),
                FadeIn(phony[1], phony[3]),
                Succession(
                    *[FadeIn(phony[5*i:5*(i+1)], shift=DOWN, run_time=0.3) for i in range(1,5)],
                ),
                lag_ratio=0.5
            )
        )

        kernels = VGroup(phony[2+i*5] for i in range(5))
        self.play(
            AnimationGroup(
                FadeOut(*[mob for mob in phony if mob not in kernels]),
                kernels.animate.arrange(RIGHT, buff=0.5)
            )
        )
        kernels_highlight = SurroundingRectangle(
            kernels,
            buff=0.5, corner_radius = 0.25,
            color=GOLD, stroke_width=4, fill_opacity=0
        )
        adj = LayerTitle('Adjustable Coefficients').next_to(kernels_highlight, UP)
        self.play(Create(kernels_highlight), FadeIn(adj))


class poolscene(Scene):
    def construct(self):
        downscaled_portion = skimage.transform.resize(np.array(Image.open(r'Assets\W5\mnist8.png')), (28,28))
        downscaled_portion = ((downscaled_portion - downscaled_portion.min())/(downscaled_portion.max()-downscaled_portion.min())*255).astype(np.uint8)
        ms= CNNDigitRecognitionScheme(downscaled_portion, input_label=8, n_filters=5, pooling_factor=7,
                                      pixel_size=(1.5/28, 1.5/28, 0.125, 0.125), horizontal_spacing=0.8,
                                      highlights_kwargs={'color': GOLD, 'stroke_width': 3},
                                      outline_kwargs={'color': DARK_BLUE, 'stroke_width': 6})
        self.add(ms)
        pool_highlight = ms.get_layer_highlight(1)
        max_pooling_title = SlideTitle('Max Pooling')
        self.play(Create(pool_highlight))
        self.wait(0.5)

        self.play(
            AnimationGroup(
                FadeOut(
                    *[mob for mob in ms.submobjects if mob not in (
                        ms.conv_layer, ms.pooling_layer, ms.conv_highlight_2, ms.pooling_highlight_1, ms.conv_pool_gizmo)],
                    ms.conv_layer[1:], ms.pooling_layer[1:], pool_highlight  
                ),
                Group(
                    ms.conv_layer[0], ms.pooling_layer[0], ms.conv_highlight_2, ms.pooling_highlight_1, ms.conv_pool_gizmo
                ).animate.scale_to_fit_height(3.75).center(),
                lag_ratio=0.5
            )
        )
        self.play(Write(max_pooling_title))

        pool_f, pool_x, pool_y = ms.pooling_factor, ms.POOLING_HIGHLIGHT_POS[0], ms.POOLING_HIGHLIGHT_POS[1]
        to_pool = PixelArray(ms.filtered_[0][pool_f*pool_x:pool_f*(pool_x+1), pool_f*pool_y:pool_f*(pool_y+1)], stroke_width=2, stroke_color=WHITE)
        to_pool.match_height(ms.conv_highlight_2).move_to(ms.conv_highlight_2)
        to_pool.add_pixel_values(color=RED)
        to_pool.set_z_index(ms.conv_highlight_2.z_index + 1)
        

        self.play(Create(to_pool.pixel_array), Create(to_pool.pixel_values))
        self.play(
            Group(
                ms.conv_layer[0], ms.pooling_layer[0], ms.conv_highlight_2, ms.pooling_highlight_1, ms.conv_pool_gizmo
            ).animate.next_to(max_pooling_title, DOWN, buff=0.5),
            to_pool.animate.scale_to_fit_height(3.75).set_x(0).to_edge(DOWN, buff=0.5),
        )

        to_pool.add_brackets(left=r"(", right=r")", color=BLACK)
        max_label = MathTex(r'\max', color=BLACK, font_size=64).next_to(to_pool.brackets, LEFT)
        equal_label = MathTex(r'=', color=BLACK, font_size=64).next_to(to_pool.brackets, RIGHT)
        max_id = np.argmax(to_pool.array)  # index into the flattened array
        max_result = to_pool.pixel_array[max_id].copy().set_stroke(GOLD,3).set_fill(opacity=0)
        
        self.play(
            Succession(
                FadeIn(max_label, to_pool.brackets, equal_label),
                Wait(0.5),
                Create(max_result)
            )
        )
        max_result.set_fill(opacity=1).add(to_pool.pixel_values[max_id].copy())
        
        self.play(max_result.animate.next_to(equal_label, RIGHT))
        self.wait(0.5)
        self.play(max_result.animate.scale_to_fit_height(ms.pooling_highlight_1.height*0.95).move_to(ms.pooling_highlight_1))
        self.wait(0.5)
        
        # DO it a second time
        ms.conv_pool_gizmo[0].add_updater(
            lambda mob: mob.put_start_and_end_on(
                ms.conv_highlight_2.get_corner(UL), ms.pooling_highlight_1.get_corner(UL)
            )
        )
        ms.conv_pool_gizmo[1].add_updater(
            lambda mob: mob.put_start_and_end_on(
                ms.conv_highlight_2.get_corner(DR), ms.pooling_highlight_1.get_corner(DR)
            )
        )

        self.play(
            FadeOut(to_pool.pixel_array, to_pool.pixel_values, max_result),
            ms.conv_highlight_2.animate.shift(LEFT*ms.conv_highlight_2.width),
            ms.pooling_highlight_1.animate.shift(LEFT*ms.pooling_highlight_1.width)
        )

        pool_f, pool_x, pool_y = ms.pooling_factor, ms.POOLING_HIGHLIGHT_POS[0], ms.POOLING_HIGHLIGHT_POS[1]-1
        to_pool_2 = PixelArray(ms.filtered_[0][pool_f*pool_x:pool_f*(pool_x+1), pool_f*pool_y:pool_f*(pool_y+1)], stroke_width=2, stroke_color=WHITE)
        to_pool_2.match_height(ms.conv_highlight_2).move_to(ms.conv_highlight_2)
        to_pool_2.add_pixel_values(color=RED)
        to_pool_2.set_z_index(ms.conv_highlight_2.z_index + 1)
        
        self.play(Create(to_pool_2.pixel_array), Create(to_pool_2.pixel_values))
        self.play(to_pool_2.animate.match_height(to_pool.pixel_array).move_to(to_pool.pixel_array))

        max_id = np.argmax(to_pool_2.array)  # index into the flattened array
        max_result = to_pool_2.pixel_array[max_id].copy().set_stroke(GOLD,3).set_fill(opacity=0)
        
        self.play(Create(max_result))
        max_result.set_fill(opacity=1).add(to_pool_2.pixel_values[max_id].copy())
        self.play(max_result.animate.next_to(equal_label, RIGHT))
        self.wait(0.1)
        self.play(max_result.animate.scale_to_fit_height(ms.pooling_highlight_1.height*0.95).move_to(ms.pooling_highlight_1))
        self.wait(0.5)

        ms.conv_pool_gizmo.clear_updaters()
        

class flattening2(Scene):
    def construct(self):
        digit = np.array(Image.open(r'Assets\W5\mnist8.png'))
        ms = CNNDigitRecognitionScheme(digit, input_label=8, n_filters=5, pooling_factor=7,
                                       pixel_size=(1.5/28, 1.5/28, 0.125, 0.125), horizontal_spacing=0.8,
                                       highlights_kwargs={'color': GOLD, 'stroke_width': 3},
                                       outline_kwargs={'color': DARK_BLUE, 'stroke_width': 6})
        ms.save_state()
        self.add(ms)

        self.play(
            FadeOut(*[mob for mob in ms.submobjects if mob not in [ms.pooling_layer]])
        )
        flattening_title = SlideTitle('Flattening')
        self.play(ms.pooling_layer.animate.scale(3).arrange(UL, buff=0.1).move_to(TITLED_CENTER), Write(flattening_title))
        
        pooled_PA = VGroup(
            PixelArray(ms.pooled_[i], stroke_color=None, stroke_width=0).match_height(ms.pooling_layer[i]).move_to(ms.pooling_layer[i]).set_z_index(ms.pooling_layer[i].z_index)
            for i in range(5)
        )

        self.add(pooled_PA)

        brace_config = {'color': DARK_BLUE}
        brace_width = Brace(pooled_PA[2], UP, **brace_config)
        w = Text("4", color=BLACK, font_size=48, font=SANS_SERIF_FONT)
        brace_width.put_at_tip(w)
        brace_height = Brace(pooled_PA[2], RIGHT, **brace_config)
        h = w.copy()
        brace_height.put_at_tip(h)
        brace_length = BraceBetweenPoints(pooled_PA[-1].get_corner(DL), pooled_PA[0].get_corner(DL), DL, **brace_config)
        l = Text("5", color=BLACK, font_size=48, font=SANS_SERIF_FONT)
        brace_length.put_at_tip(l)

        self.play(
            Succession(
                FadeIn(brace_width, w),
                FadeIn(brace_height, h),
                Wait(0.5),
                FadeIn(brace_length, l),
                Wait(1),
            )
        )
        self.play(FadeOut(brace_width, w, brace_height, h, brace_length, l))

        # self.play(AnimationGroup(Create(p) for p in pooled_PA), FadeOut(*[a.outline for a in ms.pooling_layer]))
        self.play(FadeOut(ms.pooling_layer))
        self.play(
            AnimationGroup(
                p.pixel_array.animate.arrange(RIGHT, buff=0).move_to(p)
                for p in pooled_PA
            )
        )
        self.wait(0.5)
        self.play(pooled_PA.animate.arrange(LEFT, buff=0).scale_to_fit_width(0.9*FRAME_WIDTH).move_to(TITLED_CENTER))
        
        pooled_PA.set_z_index(1)
        pooled_PA_outline = SurroundingRectangle(pooled_PA, color=DARK_BLUE, buff=0, stroke_width=6).set_z_index(pooled_PA.z_index-1)
        brace_tot = Brace(pooled_PA, UP, **brace_config)
        tot = Text("80", color=BLACK, font_size=48, font=SANS_SERIF_FONT)
        brace_length.put_at_tip(l)
        self.play(Create(pooled_PA_outline), FadeIn(brace_tot, tot))
        pooled_PA.add(pooled_PA_outline)
        ms.restore()
        ms.move_to(TITLED_CENTER)
        self.play(
            AnimationGroup(
                FadeOut(brace_tot, tot),
                pooled_PA.animate.rotate(-PI/2).match_height(ms.flattened_vector).stretch_to_fit_width(ms.flattened_vector.width).move_to(ms.flattened_vector),
                FadeIn(*[mob for mob in ms.submobjects if mob not in (ms.flattened_vector)]),
                lag_ratio = 0.5
            )
        )
 

class softmaxexp(Scene):
    def construct(self):
        digit = np.array(Image.open(r'Assets\W5\mnist8.png'))

        # Create the full main scheme for later 
        ms = CNNDigitRecognitionScheme(digit, input_label=8, n_filters=5, pooling_factor=7,
                                       pixel_size=(1.5/28, 1.5/28, 0.125, 0.125), horizontal_spacing=0.8,
                                       highlights_kwargs={'color': GOLD, 'stroke_width': 3},
                                       outline_kwargs={'color': DARK_BLUE, 'stroke_width': 6})
        dense_lay = VGroup(ms.dense_layer, ms.dense_layer_title, ms.softmax).center()
        self.add(dense_lay)
        softmax_title = SlideTitle('Softmax') 
        softmax_eq = MathTex(r'S(y_i)=\frac{e^{y_i}}{\sum_{j=1}^K e^{y_j}}', color=BLACK)
        softmax_eq.add(SurroundingRectangle(softmax_eq, stroke_color=DARK_BLUE, fill_color=WHITE, stroke_width=4, fill_opacity=0, buff=0.5, corner_radius=0.25))
        softmax_eq.move_to(TITLED_CENTER)
        ms.softmax.background_rectangle.set_fill(opacity=0)

        self.play(
            AnimationGroup(
                FadeOut(ms.dense_layer, ms.dense_layer_title),
                AnimationGroup(
                    ReplacementTransform(ms.softmax.background_rectangle, softmax_eq[1]),
                    ReplacementTransform(ms.softmax.text, softmax_eq[0]),
                ),
                Write(softmax_title),
                lag_ratio=0.5
            )
        )

        output_layer = DigitRecognitionOutputLayer()
        def softmax(array):
            exps = np.exp(array)
            return exps/np.sum(exps)
        np.random.seed(42)
        output_values = 5*(2*np.random.random(10)-1)
        output_values[8] = 4.9
        softmaxed = softmax(output_values)
        array_config = {'color':BLUE, 'include_dots':False, 'v_buff':0.4, 'h_buff': 0.3,
                        'elem_to_mob_class': CustomDecimalNumber,
                        'elem_to_mob_config': {'num_decimal_places':2, 'font': SANS_SERIF_FONT, 'color':BLACK, 'font_size':28},
                        'line_config':{'stroke_width':4}}
        example_probabilities = VectorArray(softmaxed, **array_config).match_height(output_layer)
        array_config['elem_to_mob_config'].update({'include_sign':True})
        example_raw_output = VectorArray(output_values, **array_config).match_height(output_layer)
        raw_output_label = LayerTitle('Raw Output')
        raw_output_label_tex = MathTex('y_i', color=BLACK).next_to(raw_output_label, DOWN)
        raw_output_label.add(raw_output_label_tex)
        probabilities_label = LayerTitle('Probabilities')
        probabilities_label_tex = MathTex('S(y_i)', color=BLACK).next_to(probabilities_label, DOWN)
        probabilities_label.add(probabilities_label_tex)
        # probabilites histogram on the side
        PROBABILITY_SCALE=2
        probablities_rects = VGroup(
            Rectangle(height=output_layer[i].height*0.6, width = PROBABILITY_SCALE*softmaxed[i]/np.max(softmaxed),
                      color=BLUE, fill_opacity=1
                      ).next_to(output_layer[i], buff=0.25)
            for i in range(10)
        ) 
        rect_line = Line(UP, DOWN, stroke_width=2, color=DARK_BLUE).match_height(output_layer).next_to(probablities_rects, LEFT, buff=0)

        # Arrange + arrows
        phony_rect = softmax_eq[1].copy()
        softmax_exp_scheme = VGroup(example_raw_output, phony_rect, example_probabilities, VGroup(output_layer, probablities_rects, rect_line)).arrange(buff=0.5).move_to(TITLED_CENTER)
        arrow_config = {'color': DARK_BLUE, 'stroke_width':4, 'max_stroke_width_to_length_ratio': 20, 'buff':0}
        raw_output_label.next_to(example_raw_output, UP)
        probabilities_label.next_to(example_probabilities, UP)
        arrow_1 = Arrow(example_raw_output.get_right(), phony_rect.get_left(), **arrow_config)
        arrow_2 = Arrow(phony_rect.get_right(), example_probabilities.get_left(), **arrow_config)
        softmax_exp_scheme.add(raw_output_label, probabilities_label, arrow_1, arrow_2)
        softmax_exp_scheme.move_to(TITLED_CENTER)

        self.play(
            Succession(
                AnimationGroup(
                    Create(example_raw_output),
                    FadeIn(raw_output_label),
                    softmax_eq.animate.move_to(phony_rect)
                ),
                Create(arrow_1, run_time=0.5),
                Create(arrow_2, run_time=0.5),
                AnimationGroup(
                    Create(example_probabilities),
                    FadeIn(probabilities_label),
                ),
                FadeIn(output_layer),
                AnimationGroup(
                    *[GrowFromEdge(rect, LEFT)
                    for rect in probablities_rects],
                )
            )
        )
        self.play(output_layer.Activate(8))


class Test(Scene):
    def construct(self):
        np.random.seed(0)
        def scale_pulse(m, alpha):
            # alpha goes from 0 → 1 during the animation
            scale = 1 + 0.3 * np.sin(np.pi * alpha)
            m.scale_to_fit_height(0.2* scale)
        matrix_config = {'v_buff':0.6, 'h_buff':0.6, 'bracket_h_buff':0.1, 'bracket_v_buff':0.1}
        initialized_filters = VGroup(
            Matrix(
                np.round(7.5*(2*np.random.random((3,3))-1),1),  **matrix_config,
                element_to_mobject=ValueDisplay, element_to_mobject_config={'include_sign':True, 'font_size':24, 'num_decimal_places':1},
            ).set_color(BLACK) for i in range(3)
        ).arrange(RIGHT)

    
        initialized_weights = VGroup(
            ValueDisplay(7.5*(2*np.random.random()-1), **{'include_sign':True, 'font_size':28, 'num_decimal_places':1}).set_color(BLACK)
        ).arrange().shift(UP*3)  # align with the \dots (weights_eq[-1] is ',\dots]')

        initialized_biases = VGroup(
            ValueDisplay(
                7.5*(2*np.random.random()-1), **{'include_sign':True, 'font_size':28, 'num_decimal_places':1}
            ).set_color(BLACK)
            for i in range(3)
        ).arrange().shift(DOWN*3)

        learnable_coeffs = VGroup(*[elem for filter in initialized_filters for elem in filter.elements], *initialized_weights, *initialized_biases)
        for mob in learnable_coeffs:
            print(mob)

        
        self.add(learnable_coeffs)
        self.play(UpdateLearnableCoefficients(learnable_coeffs, u_range=2))


class W5Theory_slides(MOOCSlide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # EXAMPLES OF OBJECT DETECTION ARE SHOWN.
        self.next_slide(
            notes=
            '''Nowadays, computer vision systems can extract text and numbers
            directly from digital images and videos, and they are also capable
            of detecting objects or recognizing people.
            '''
        )
        self.wait(0.5)

        # SLIDE 02:  ===========================================================
        # SPECIFIC EXMAPLE OF CAR PLATE DIGIT RECOGNITION IS SHOWN SCHEMATICALLY
        self.next_slide(
            notes=
            '''Think about license plates: how do cameras scanning traffic
            instantly read them? They are all powered by sophisticated
            mathematical algorithms.
            '''
        )
        car = SVGMobject(r'Assets\W5\car_front_icon.svg', color=BLACK).scale_to_fit_height(4.3).move_to(3.4*RIGHT+2*DOWN)
        car_plate = Text('196843', color=BLACK, font='Bahnschrift', font_size=26).move_to(car).shift(DOWN*0.27*car.height)
        car_plate_outline = SurroundingRectangle(car_plate, stroke_color=BLACK, stroke_width=4, fill_opacity=0, buff=0.07, corner_radius=0.07)
        car_plate.add(car_plate_outline)
        car.add(car_plate)
        
        cctv = SVGMobject(r'Assets\W5\cctv_icon.svg', color=BLACK
                          ).flip(DOWN).scale_to_fit_height(2).move_to(car.get_center()+6.4*LEFT+5*UP).set_z_index(1)
        VGroup(car, cctv).center()
        recognized_plate = Text('196843', color=GRAY, font=SANS_SERIF_FONT, weight=BOLD).match_x(cctv).set_y(0)
        cctv_arrow = Arrow(cctv.get_bottom(), recognized_plate.get_top(), color=DARK_BLUE, stroke_width=6, max_stroke_width_to_length_ratio=20)

        cctv_point1, cctv_point2 = cctv.get_boundary_point(RIGHT), cctv.get_boundary_point(DOWN)
        cctv_fov = Polygon(
            cctv_point1,
            cctv_point2,
            car_plate.get_critical_point(DL),
            car_plate.get_critical_point(DR),
            car_plate.get_critical_point(UR),
            stroke_width=0, color=BLUE,
            fill_opacity=0.4,
        ).round_corners(radius=0.07)

        self.play(
            Succession(
                FadeIn(car, cctv),
                GrowFromPoint(cctv_fov, (cctv_point1 + cctv_point2)/2),
                AnimationGroup(
                    GrowArrow(cctv_arrow),
                    FadeIn(recognized_plate, shift=DOWN))
            )
        )

        # SLIDE 03:  ===========================================================
        # DIGIT RECOGNITION TITLE APPEARS
        # SINGLE NUMBER BEING HIGHLIGHTED AND DOWNSCALED
        # INPUT OUTPUT SCHEME OF THE DIGIT RECOGNITION ALGORITHM
        self.next_slide(
            notes=
            '''Licence plate recognition is a task known as digit recognition.
            It's basically a classification problem where the goal is to
            associate each portion of an image containing a number with the
            corresponding category: the digits from 0 to 9. But how can an
            algorithm perform this task?
            '''
        )
        self.play(FadeOut(car, cctv, recognized_plate, cctv_arrow, cctv_fov))

        digit = np.array(Image.open(r'Assets\W5\mnist8.png'))
        downscaled_digit = skimage.transform.resize(digit, (15,15))
        downscaled_digit = ((downscaled_digit - downscaled_digit.min())/(downscaled_digit.max()-downscaled_digit.min())*255).astype(np.uint8)
        digit_PA = PixelArray(downscaled_digit, stroke_width=1, stroke_color=WHITE)
        digit_PA.scale_to_fit_height(0.3*FRAME_HEIGHT)

        # Create the full main scheme for later 
        ms = CNNDigitRecognitionScheme(digit, input_label=8, n_filters=5, pooling_factor=7,
                                       pixel_size=(1.5/28, 1.5/28, 0.125, 0.125), horizontal_spacing=0.8,
                                       highlights_kwargs={'color': GOLD, 'stroke_width': 3},
                                       outline_kwargs={'color': DARK_BLUE, 'stroke_width': 6})
        ms.save_state()

        digit_PA.move_to(ms.input)

        digit_recognition_algo = Paragraph('Digit Recognition\nAlgorithm', font=SANS_SERIF_FONT, weight=BOLD, font_size=48, color=DARK_BLUE, alignment='center')
        digit_recognition_algo.background_rectangle = SurroundingRectangle(
            digit_recognition_algo,
            fill_color=WHITE, fill_opacity=1, stroke_color=DARK_BLUE, stroke_width=4,
            corner_radius=0.25, buff=0.5
        )
        digit_recognition_algo.add_to_back(digit_recognition_algo.background_rectangle)
        output_l = ms.output_layer.copy()
        output_l.save_state()

        VGroup(digit_PA, digit_recognition_algo, output_l).arrange(RIGHT, buff=0.75)
        input_t = ms.input_title.copy().next_to(digit_PA, UP, buff=0.25)
        output_layer_t = ms.output_layer_title.copy().next_to(output_l, UP, buff=0.25)

        left_arr = Arrow(digit_PA.get_right(), digit_recognition_algo.get_left(), color=DARK_BLUE, stroke_width=4, buff=0)
        right_arr = BrokenArrow(digit_recognition_algo.get_right(), output_l[8].get_left(), n_turns=2, color=DARK_BLUE, add_tip=False)
        right_arr.add_tip(left_arr.tip.copy())
        intro_scheme = VGroup(digit_PA.pixel_array, input_t, output_l, output_layer_t, digit_recognition_algo, left_arr, right_arr) 
        self.play(FadeIn(intro_scheme[:-1]))  # EVerything but the output arrow
        self.play(Create(right_arr, run_time=0.5))
        self.play(output_l.Activate(8), run_time=0.5)

        # SLIDE 04:  ===========================================================
        # FOCUS ON DIGIT IMAGE, EVERYTHING ELSE DISAPPEARS
        self.next_slide(
            notes=
            '''First, we need to recall how a computer "sees" images. As we have
            already seen, to a computer, an image is a structured grid made of
            tiny squares called pixels.
            '''
        )
        self.play(FadeOut(intro_scheme[1:]))  # everything but the input image
        self.play(digit_PA.animate.scale_to_fit_height(0.75*FRAME_HEIGHT).center())
        
        # SLIDE 05:  ===========================================================
        # GRAYSCALE VALUES APPEARS ON THE IMAGE
        self.next_slide(
            notes=
            '''In greyscale images, each pixel holds a value encoding its
            brightness, ranging from 0 for black to 255 for white.
            '''
        )
        digit_PA.add_pixel_values(color=SATURATED_RED)
        self.play(Create(digit_PA.pixel_values, lag_ratio=0.1))

        # SLIDE 06:  ===========================================================
        # GRAYSCALE VALUES FORM A MATRIX ON THE SIDE
        self.next_slide(
            notes=
            '''The collection of pixel values leads to a matrix of numbers, with
            rows and columns describing the position of the pixel in the image.
            '''
        )
        self.play(
            AnimationGroup(
                digit_PA.pixel_array.animate.move_to(HALF_SCREEN_LEFT).scale(0.6),
                digit_PA.pixel_values.animate.move_to(HALF_SCREEN_RIGHT).scale(0.6),
            )
        )

        digit_PA.add_brackets(color=BLACK)
        self.play(
            digit_PA.pixel_values.animate.set_color(BLACK),
            FadeIn(digit_PA.brackets)
        )

        # SLIDE 07:  ===========================================================
        # THE MATRIX BECOMES THE INPUT THE PREVIOUS 
        self.next_slide(
            notes=
            '''A digit recognition algorithm processes the input matrix to
            associate the image to the correct category.
            '''
        )
        self.play(
            AnimationGroup(
                FadeOut(digit_PA.pixel_array),
                VGroup(digit_PA.pixel_values, digit_PA.brackets).animate.scale_to_fit_width(0.3*FRAME_HEIGHT).next_to(left_arr, LEFT, buff=0),
                FadeIn(intro_scheme[1:]),
                lag_ratio = 0.5
            )
        )

        # SLIDE 08:  ===========================================================
        # CNN TITLE APPEARS
        # GRAPHICAL SCHEME WITH 3 STEPS OF A CNN, NO NAMES YET
        self.next_slide(
            notes=
            '''One of the most popular approaches uses the so-called
            convolutional neural networks, a class of deep learning model
            designed to process images. A typical CNN involves 3 processing
            steps, each handled by different layers of the model:
            '''
        )
        CNN_title = SlideTitle('Convolutional Neural Network')
        ms.move_to(TITLED_CENTER)
        
        self.play(
            FadeOut(intro_scheme.remove(output_l, output_layer_t, digit_PA.pixel_array), digit_PA.pixel_values, digit_PA.brackets),
            ReplacementTransform(output_l, ms.output_layer),
            ReplacementTransform(output_layer_t, ms.output_layer_title),
        )
        self.play(
            Write(CNN_title),
            FadeIn(
                *[mob for mob in ms.submobjects if mob not in (
                    ms.output_layer, ms.output_layer_title, ms.conv_layer_title, ms.pooling_layer_title, ms.flattening_layer_title, ms.dense_layer_title, ms.output_arrow
                )]
            )
        )
        self.play(Create(ms.output_arrow, run_time=0.5))
        self.play(ms.output_layer.Activate(8), run_time=0.5)

        # SLIDE 09:  ===========================================================
        # CONVOLUTIONAL LAYER HIGHLIGHTED AND NAME APPEARS
        self.next_slide(
            notes=
            '''A convolutional layer finds visual patterns. It scans the input
            matrix with filters to highlight important feature of the image,
            such as straight lines, curves or edges.
            '''
        )
        highlight_config = {'color': GOLD, 'stroke_width': 4}
        conv_highlight = ms.get_layer_highlight('conv', **highlight_config)
        self.play(FadeIn(ms.conv_layer_title), Create(conv_highlight))

        # SLIDE 10:  ===========================================================
        # POOLING LAYER HIGHLIGHTED AND NAME APPEARS
        self.next_slide(
            notes=
            '''A pooling layer compresses the information resulting from the
            filters into a vector that captures the essential characteristics of
            the image
            '''
        )
        pool_highlight = ms.get_layer_highlight('pool', **highlight_config)
        self.play(FadeIn(ms.pooling_layer_title), ReplacementTransform(conv_highlight, pool_highlight))

        # SLIDE 11:  ===========================================================
        # FLATTENING LAYER HIGHLIGHTED AND NAME APPEARS
        self.next_slide(
            notes=
            '''The flattening layer then arranges the data in a long linear
            vector;
            '''
        )
        flatten_highlight = ms.get_layer_highlight('flat', **highlight_config)
        self.play(FadeIn(ms.flattening_layer_title), ReplacementTransform(pool_highlight, flatten_highlight))

        # SLIDE 12:  ===========================================================
        # DENSE LAYER HIGHLIGHTED AND NAME APPEARS
        self.next_slide(
            notes=
            '''Finally, a dense neural network completes the classification task
            by mapping the compressed vector to a category, corresponding to the
            predicted digit
            '''
        )
        dense_highlight = ms.get_layer_highlight('dense', **highlight_config)
        self.play(FadeIn(ms.dense_layer_title), ReplacementTransform(flatten_highlight, dense_highlight))

        # SLIDE 13:  ===========================================================
        # FOCUS ON DENSE LAYER
        self.next_slide(
            notes=
            '''Let's now examine each one of these layers in detail, starting
            from the last one.
            '''
        )
        dense_layer_vg = VGroup(ms.dense_layer_title, ms.dense_layer)
        full_dense_layer_g = Group(ms.dense_layer_title, ms.dense_layer, ms.softmax)
        # saved_state = dense_layer_vg.copy()

        self.play(
            FadeOut(
                CNN_title,
                dense_highlight,
                *[mob for mob in ms.submobjects if (mob not in full_dense_layer_g)]
            )
        )
        self.play(full_dense_layer_g.animate.center())
        dense_layer_vg.save_state()

        # SLIDE 14:  ===========================================================
        # HIGHLIGHT NODES OF THE DENSE LAYER
        self.next_slide(
            notes=
            '''A standard dense neural network is made of interconnected nodes.
            '''
        )
        target = dense_layer_vg.copy()
        target[0].match_x(target[1])
        target.set_x(0)
        self.play(
            AnimationGroup(
                FadeOut(ms.softmax),
                dense_layer_vg.animate.become(target),
                lag_ratio=0.5
            )
        )

        VGroup(ms.dense_layer.input_nodes, ms.dense_layer.output_nodes).set_z_index(1)
        self.play(
            ms.dense_layer.EdgePropagationAnimation(color=RED, stroke_width=2, run_time=2.5, lag_ratio=0.01, time_width=0.5)
        )
        VGroup(ms.dense_layer.input_nodes, ms.dense_layer.output_nodes).set_z_index(0)

        # SLIDE 15:  ===========================================================
        # SCHEME OF A SINGLE NODE APPEARS ON THE RIGHT (SHIFT DENSE LAYER LEFT)
        # INPUT ARROWS AND LABELS APPEAR
        # OUTPUT ARROW AND LABEL APPEAR
        self.next_slide(
            notes=
            '''Each node takes in some input numbers (in this example, x_0, x_1
            and x_2) and produces an output y.
            '''
        )
        self.play(dense_layer_vg.animate.shift(LEFT*5))

        node_scheme = NodeScheme().shift(RIGHT*2)
        node_scheme_rect = SurroundingRectangle(node_scheme, stroke_color=DARK_BLUE, stroke_width=4, fill_opacity=0, corner_radius=0.25, buff=0.5)
        # create pleasant curved arrow
        start = ms.dense_layer.output_nodes[0].get_center()
        end = node_scheme_rect.get_left()
        alpha = Line(end, start).get_angle() -PI/2
        curve_arrow = ArcBetweenPoints(ms.dense_layer.output_nodes[0].point_at_angle(2*alpha-PI), node_scheme_rect.get_left(), color=DARK_BLUE, stroke_width=4, angle=PI - 2*alpha)
        curve_arrow.add_tip(node_scheme.output_arrow.tip.copy().set_color(DARK_BLUE))
        # Formula below
        substrings = ['y=', r'\sigma', '(', ')', '+', 'b'] + [f'w_{i}' for i in range(3)] + [f'x_{i}' for i in range(3)]
        # NOTE: Putting spaces in the Tex strings makes numbering unreliable due to adding invisible mobjects!
        node_equation_2 = MathTex(r'y=\sigma(w_0x_0+w_1x_1+w_2x_2+b)', color=BLACK, substrings_to_isolate=substrings).next_to(node_scheme_rect, DOWN, buff=0.5)
        # node_equation_1 = MathTex(r'y=\sigma(w_0x_0+w_1x_1+w_2x_2)', color=BLACK, substrings_to_isolate=substrings)
        # node_equation_1.shift(node_equation_2[0].get_center()-node_equation_1[0].get_center())

        self.play(ReplacementTransform(ms.dense_layer.output_nodes[0].copy(), node_scheme.circle))
        self.play(FadeIn(node_scheme_rect, curve_arrow))
        self.play(FadeIn(node_scheme.input_labels, node_scheme.input_arrows))
        self.play(FadeIn(node_scheme.output_label, node_scheme.output_arrow))

        # SLIDE 16:  ===========================================================
        # WWEIGHT LABELS APPEAR
        # SUM SYMBOL AND BIAS "b" LABEL APPEAR
        # FOMULA FOR Y APPEARS BELOW, HIGHLIGHTING WEIGHTS AND BIAS ACCORDINGLY
        self.next_slide(
            notes=
            '''To do this, it multiplies each input by an adjustable coefficient
            called weight (here w_0, w_1, and w_2, respectively) and then adds
            another adjustable coefficient called bias (here denoted by b)
            '''
        )
        self.play(FadeIn(node_scheme.weight_labels))
        self.play(FadeIn(node_scheme.sum, node_equation_2[:-3], node_equation_2[-1]))
        self.play(FadeIn(node_scheme.top_middle_arrow, node_scheme.middle_arrow, node_scheme.b, node_equation_2[-3:-1]))

        # SLIDE 17:  ===========================================================
        # ACTIVATION FUNCTION ILLUSTRATION APPEARS, SIGMA APPEARS IN THE FORMULA
        # SIGMA HIGHLIGHTED IN FORMULA AND SCHEME
        self.next_slide(
            notes=
            '''Finally, the weighted sum is passed to the function sigma, known
            as "activation function", which further modulates the incoming
            information.
            '''
        )
        self.play(FadeIn(node_scheme.activation_function))
        # self.play(TransformMatchingTex(node_equation_1, node_equation_2))
        
        # SLIDE 18:  ===========================================================
        # HIGHLIGHT WEIGHTS AND BIASES IN FORMULA
        self.next_slide(
            notes=
            '''The choice of the adjustable coefficients is crucial, and it
            allows the neuron's sensitivity to the inputs to be adjusted.
            '''
        )
        weights_highlight = VGroup(
            HighlightRectangle(node_equation_2[3], color=GOLD),
            HighlightRectangle(node_equation_2[6], color=GOLD),
            HighlightRectangle(node_equation_2[9], color=GOLD),
        )
        weights_label = Text('Weights', font=SANS_SERIF_FONT, font_size=32,color=GOLD).next_to(weights_highlight, DOWN)
        bias_highlight =  HighlightRectangle(node_equation_2[-2], color=BLUE)
        bias_label = Text('Bias', font=SANS_SERIF_FONT, font_size=32,color=BLUE).next_to(bias_highlight, DOWN).align_to(weights_label[0], DOWN)
        self.play(FadeIn(weights_highlight, weights_label))
        self.play(FadeIn(bias_highlight, bias_label))
        
        # SLIDE 19:  ===========================================================
        # DENSE LAYER NODE SCHEME DISAPPEARS
        # MATHEMATICAL INTERPRETATION OF NN IS WRITTEN
        # CHANGE NODES AND PARAMETERS TO MATCHING COLORS
        self.next_slide(
            notes=
            '''The dense neural network can be seen as a parametrized function
            that depends on all the weights and biases of all the nodes in the
            network. We will investigate later how to properly find them, but
            before let's see how the dense layer input vector is constructed.
            '''
        )
        self.play(FadeOut(node_scheme, node_scheme_rect, node_equation_2, weights_highlight, weights_label, 
                          bias_highlight, bias_label, curve_arrow))
        
        # Explanation of adjustable coefficients
        NN_equation = MathTex(r'\mathbf{y} = \mathcal{NN}(\mathbf{x}; \mathbf{W}, \mathbf{b})', color=BLACK, font_size=48)
        adjustable_coeff_t =  LayerTitle('Learnable coefficients:')
        weights_eq = MathTex(r'\mathbf{W} = [w_0,w_1,w_2,w_0,w_1,w_2,w_0,w_1,w_2,\dots]', color=BLACK, font_size=48, substrings_to_isolate=[f'w_{i}' for i in range(3)])
        bias_eq = MathTex(r'\mathbf{b} = [b,b,b,\dots]', color=BLACK, font_size=48) # isolating 'b' breaks here
        
        adj_coeff_expl = VGroup(NN_equation, adjustable_coeff_t, weights_eq, bias_eq).arrange(DOWN, buff=0.8).shift(RIGHT*1.5)
        adj_coeff_rect = SurroundingRectangle(adj_coeff_expl, stroke_color=GOLD, stroke_width=4, fill_opacity=0, buff=0.5, corner_radius=0.25)
        
        wees = weights_eq[1:1+3*3*2:2] # 3 groups of 3 weights each
        bees: VGroup = bias_eq[0][3:8:2]
        node_colors = [RED, ManimColor("#83C167"), ORANGE]
        node_scale_factor = 1.1
        self.play(FadeIn(adj_coeff_expl, adj_coeff_rect))
        self.play(
            Succession(*[
                AnimationGroup(
                    ms.dense_layer.output_nodes[i].animate.set_color(node_colors[i]).scale(node_scale_factor),
                    *[
                        weight.animate.set_color(node_colors[i]).scale(node_scale_factor)
                        for weight in wees[3*i:3*(i+1)]
                    ],
                    bees[i].animate.set_color(node_colors[i]).scale(node_scale_factor)
                )
                for i in range(3)]
            )
        )

        # SLIDE 20:  ===========================================================
        # DENSE LAYER GOES BACK TO CENTER, SOFTMAX REAPPEARS
        # SOFTMAX IS EXPANDED TO FORMULA
        self.next_slide(
            notes=
            '''The raw output of the dense layer is then processed by the
            softmax function. What is it?
            '''
        )
        self.play(
            AnimationGroup(
                FadeOut(adj_coeff_expl, adj_coeff_rect),
                dense_layer_vg.animate.restore(),
                FadeIn(ms.softmax),
                lag_ratio=0.5
            )
        )

        softmax_title = SlideTitle('Softmax') 
        softmax_eq = MathTex(r'S(y_i)=\frac{e^{y_i}}{\sum_{j=1}^K e^{y_j}}', color=BLACK)
        softmax_eq.add(SurroundingRectangle(softmax_eq, stroke_color=DARK_BLUE, fill_color=WHITE, stroke_width=4, fill_opacity=0, buff=0.5, corner_radius=0.25))
        softmax_eq.move_to(TITLED_CENTER)
        ms.softmax.background_rectangle.set_fill(opacity=0)

        self.play(
            AnimationGroup(
                FadeOut(ms.dense_layer, ms.dense_layer_title),
                AnimationGroup(
                    ReplacementTransform(ms.softmax.background_rectangle, softmax_eq[1]),
                    ReplacementTransform(ms.softmax.text, softmax_eq[0]),
                ),
                Write(softmax_title),
                lag_ratio=0.5
            )
        )

        # SLIDE 21:  ===========================================================
        # SOFTMAX EXPLANATION
        self.next_slide(
            notes=
            '''The SoftMax function is used for multi-class classification. It
            converts the raw output scores of the network into a probability
            distribution over the possible classes.
            '''
        )
        output_layer = DigitRecognitionOutputLayer()
        # Create the two array wit raw values and probabilities
        def softmax(array):
            exps = np.exp(array)
            return exps/np.sum(exps)
        np.random.seed(42)
        output_values = 5*(2*np.random.random(10)-1)
        output_values[8] = 4.9
        softmaxed = softmax(output_values)
        array_config = {'color':BLUE, 'include_dots':False, 'v_buff':0.4, 'h_buff': 0.3,
                        'elem_to_mob_class': CustomDecimalNumber,
                        'elem_to_mob_config': {'num_decimal_places':2, 'font': SANS_SERIF_FONT, 'color': BLACK, 'font_size':28},
                        'line_config':{'stroke_width':4}}
        
        example_probabilities = VectorArray(softmaxed, **array_config).match_height(output_layer)
        array_config['elem_to_mob_config'].update({'include_sign':True})
        example_raw_output = VectorArray(output_values, **array_config).match_height(output_layer)
        raw_output_label = LayerTitle('Raw Output')
        raw_output_label_tex = MathTex('y_i', color=BLACK).next_to(raw_output_label, DOWN)
        raw_output_label.add(raw_output_label_tex)
        probabilities_label = LayerTitle('Probabilities')
        probabilities_label_tex = MathTex('S(y_i)', color=BLACK).next_to(probabilities_label, DOWN)
        probabilities_label.add(probabilities_label_tex)

        # probabilites histogram on the side of the output layer
        PROBABILITY_SCALE=2
        probablities_rects = VGroup(
            Rectangle(height=output_layer[i].height*0.6, width = PROBABILITY_SCALE*softmaxed[i]/np.max(softmaxed),
                      color=BLUE, fill_opacity=1
                      ).next_to(output_layer[i], buff=0.25)
            for i in range(10)
        ) 
        rect_line = Line(UP, DOWN, stroke_width=2, color=DARK_BLUE).match_height(output_layer).next_to(probablities_rects, LEFT, buff=0)

        # Arrange + arrows
        phony_rect = softmax_eq[1].copy()
        softmax_exp_scheme = VGroup(example_raw_output, phony_rect, example_probabilities, VGroup(output_layer, probablities_rects, rect_line)).arrange(buff=0.5).move_to(TITLED_CENTER)
        arrow_config = {'color': DARK_BLUE, 'stroke_width':4, 'max_stroke_width_to_length_ratio': 20, 'buff':0}
        raw_output_label.next_to(example_raw_output, UP)
        probabilities_label.next_to(example_probabilities, UP)
        arrow_1 = Arrow(example_raw_output.get_right(), phony_rect.get_left(), **arrow_config)
        arrow_2 = Arrow(phony_rect.get_right(), example_probabilities.get_left(), **arrow_config)
        softmax_exp_scheme.add(raw_output_label, probabilities_label, arrow_1, arrow_2)
        softmax_exp_scheme.move_to(TITLED_CENTER)

        self.play(
            Succession(
                AnimationGroup(
                    Create(example_raw_output),
                    FadeIn(raw_output_label),
                    softmax_eq.animate.move_to(phony_rect)
                ),
                Create(arrow_1, run_time=0.5),
                Create(arrow_2, run_time=0.5),
                AnimationGroup(
                    Create(example_probabilities),
                    FadeIn(probabilities_label),
                ),
                FadeIn(output_layer),
                AnimationGroup(
                    *[GrowFromEdge(rect, LEFT)
                    for rect in probablities_rects],
                )
            )
        )
        self.play(output_layer.Activate(8))

        # SLIDE 22:  ===========================================================
        # DENSE LAYER + SOFTMAX + OUTPUT LAYER APPEAR
        # FLATTENED LAYER APPEARS AND HIS HIGHLIGHTED
        self.next_slide(
            notes=
            '''Now let's see how the dense layer input vector is constructed.
            '''
        )
        ms.restore()
        ms.output_layer.activate(8)
        flat_to_end_g = Group(ms.flattened_vector, ms.dense_layer, ms.softmax, ms.dense_layer, ms.output_arrow, ms.dense_layer_title, ms.output_layer)
        flat_to_end_g.center()
        self.play(
            FadeOut(*[mob for mob in self.mobjects if mob not in [output_layer]]),
            ReplacementTransform(output_layer, ms.output_layer))
        self.play(FadeIn(flat_to_end_g[1:-1]))

        # flat_to_end_g.center()
        flattened_vector_highlight = SurroundingRectangle(ms.flattened_vector, buff=0.25, color=BLUE, stroke_width=4, corner_radius = 0.075)
        self.play(
            Succession(
                FadeIn(ms.flattened_vector),
                Create(flattened_vector_highlight)
            )
        )

        # SLIDE 23:  ===========================================================
        # FULL DIGIT RECOGNITION SCHEME REAPPEARS, CONVOLUTIONAL LAYER HIGHLIGHT
        self.next_slide(
            notes=
            '''Actually, we need to go back to the first layer, called
            convolutional layer. Given an image, our goal is to extract its most
            important features. To that end, we can resort to the convolution
            with a filter.
            '''
        )
        for mob in ms.submobjects:
            if (mob not in flat_to_end_g): mob.restore();
        ms.output_layer.saved_state.activate(8)  # instead of saving the full scheme again, hack the current saved copy
        conv_highlight = ms.get_layer_highlight('conv', **highlight_config)

        self.play(
            AnimationGroup(
                FadeOut(flattened_vector_highlight),
                AnimationGroup(mob.animate.restore() for mob in flat_to_end_g),
                FadeIn(*[mob for mob in ms.submobjects if mob not in flat_to_end_g]),
                lag_ratio=0.5
            )
        )
        self.play(Create(conv_highlight))
    
        # SLIDE 24:  ===========================================================
        # FOCUS ON INPUT AND CONVOLUTION LAYER WHILE ADDING A 3X3 KERNEL
        self.next_slide(
            notes=
            '''We recall, from week 4, that Filters are small matrices, for
            instance 3x3, with specific entries.
            '''
        )
        # Create new target for focusing on the convolutioon
        new_highlight_config = {'color':GOLD, 'stroke_width':6}
        new_input = ms.input.copy()
        new_conv = ms.conv_layer.top().copy()
        new_input_highlight = ms.input_highlight.copy().set(**new_highlight_config)
        new_conv_highlight = ms.conv_highlight_1.copy().set(**new_highlight_config)
        Group(new_input, new_input_highlight).scale(3).move_to(HALF_SCREEN_LEFT + 2*UP)
        Group(new_conv, new_conv_highlight).scale(3).move_to(HALF_SCREEN_RIGHT + 2*UP)

        side=1
        filter_kernel = VGroup(
            Square(side, fill_opacity=0, **new_highlight_config).add(
                MathTex(f'l_{i}', color=GOLD).scale_to_fit_height(side*0.5)
            ) for i in range(9)
        ).arrange_in_grid(3,3, buff=0).move_to(3*DOWN)

        phony_filter_square = Square(filter_kernel.width, **new_highlight_config).set_opacity(0).move_to(filter_kernel)
        new_giz1 = create_gizmo(new_input_highlight, phony_filter_square)
        new_giz2 = create_gizmo(phony_filter_square, new_conv_highlight)

        self.play(
            FadeOut(*[mob for mob in ms.submobjects if mob not in (ms.input, ms.input_highlight, ms.conv_layer, ms.conv_highlight_1)]),
            FadeOut(ms.conv_layer[1:], conv_highlight)
        )
        self.play(
            ReplacementTransform(ms.input, new_input),
            ReplacementTransform(ms.input_highlight, new_input_highlight),
            ReplacementTransform(ms.conv_layer.top(), new_conv),
            ReplacementTransform(ms.conv_highlight_1, new_conv_highlight),
        )
        self.play(
            Create(filter_kernel, run_time=1),
            Succession(Wait(0.3),  Create(new_giz1,lag_ratio=0,run_time=0.5)),
            Succession(Wait(1),  Create(new_giz2,lag_ratio=0,run_time=0.5)),
        )

        # SLIDE 25:  ===========================================================
        # GRID WITH: INPUT * KERNEL = RESULT IS CREATED, WITH THE EXAMPLE RESULT
        # BECOMING THE FIRST ROW
        self.next_slide(
            notes=
            '''When we apply each filter we produce a new output matrix with a
            specific effect. This effect sheds a light on a specific feature.
            '''
        )
        filter_colors = [GOLD, RED, GREEN, ORANGE, PURPLE]
        star_symbol = MathTex('*', color=BLACK, font_size=64)
        equal_symbol = MathTex('=', color=BLACK, font_size=64)
        grid_of_convolutions = Group( 
            *[m for i in range(5) for m in [
                ms.input.copy().restore(),
                star_symbol.copy(),
                filter_kernel.copy().scale(0.5).set_color(filter_colors[i]),
                equal_symbol.copy(),
                ms.conv_layer[i].copy().restore()
            ]]
        ).arrange_in_grid(5,5, (1, 0.25)).center()
        self.play(
            AnimationGroup(
                FadeOut(VGroup(new_input_highlight, new_conv_highlight, new_giz1, new_giz2)),
                AnimationGroup(
                    ReplacementTransform(new_input, grid_of_convolutions[0]),
                    ReplacementTransform(filter_kernel, grid_of_convolutions[2]),
                    ReplacementTransform(new_conv, grid_of_convolutions[4]),
                ),
                FadeIn(grid_of_convolutions[1], grid_of_convolutions[3]),
                Succession(
                    *[FadeIn(grid_of_convolutions[5*i:5*(i+1)], shift=DOWN, run_time=0.3) for i in range(1,5)],
                ),
                lag_ratio=0.5
            )
        )

        # SLIDE 26:  ===========================================================
        # LIST ADJUSTABLE FILTER COEFFICIENTS
        self.next_slide(
            notes=
            '''But since these features are not known in advance, the entries of
            the filters must be adjusted during training, just as the weights
            and biases of the dense neural networks. This way, the filters are
            automatically adapted to the specific types of images the model
            needs to process.
            '''
        )
        kernels = VGroup(grid_of_convolutions[2+i*5] for i in range(5))
        self.play(
            AnimationGroup(
                FadeOut(*[mob for mob in grid_of_convolutions if mob not in kernels]),
                kernels.animate.arrange(RIGHT, buff=0.5)
            )
        )
        kernels_highlight = SurroundingRectangle(
            kernels,
            buff=0.5, corner_radius = 0.25,
            color=GOLD, stroke_width=4, fill_opacity=0
        )
        adj = LayerTitle('Learnable Coefficients').next_to(kernels_highlight, UP)
        self.play(Create(kernels_highlight), FadeIn(adj))

        # SLIDE 27:  ===========================================================
        # FULL DIGIT RECOGNITION SCHEME REAPPEARS, POOLING LAYER HIGHLIGHT
        self.next_slide(
            notes=
            '''Pooling is the second building block: it reduces the
            dimensionality of the input by processing groups of pixels.
            '''
        )
        ms.restore()
        pool_highlight = ms.get_layer_highlight('pool', **highlight_config)

        self.play(
            Succession(
                FadeOut(kernels_highlight, adj, kernels),
                FadeIn(ms),
                Create(pool_highlight)
            )
        )

        # SLIDE 28:  ===========================================================
        # MAX POOLING TITLE APPEARS
        self.next_slide(
            notes=
            '''There are different options for the pooling operation, but in our
            example, we present the max pooling.
            '''
        )
        max_pooling_title = SlideTitle('Max Pooling')
        pooling_example_g = Group(ms.conv_layer[0], ms.pooling_layer[0], ms.conv_highlight_2, ms.pooling_highlight_1, ms.conv_pool_gizmo)

        self.play(
            AnimationGroup(
                FadeOut(
                    *[mob for mob in ms.submobjects if mob not in (
                        ms.conv_layer, ms.pooling_layer, ms.conv_highlight_2, ms.pooling_highlight_1, ms.conv_pool_gizmo)],
                    ms.conv_layer[1:], ms.pooling_layer[1:], pool_highlight  
                ),
                pooling_example_g.animate.scale_to_fit_height(3.75).center(),
                lag_ratio=0.5
            )
        )
        self.play(Write(max_pooling_title))

        # SLIDE 29:  ===========================================================
        # 3X3 OF THE CONVOLUTION OUTPUT SI EXTRACTED, GRAYSCALE VALUES APPEAR
        # MAX()=ENCLOSES THE 3X3
        # THE MAXIMUM  IS HIGHLIGHTED AND "COPIED" TO THE SIDE AND TO THE RESULT
        self.next_slide(
            notes=
            '''This latter compress information by selecting the largest value
            among a group of pixels in the feature image. In this case, the
            input is the output of the convolutional layer.
            '''
        )
        pool_f, pool_x, pool_y = ms.pooling_factor, ms.POOLING_HIGHLIGHT_POS[0], ms.POOLING_HIGHLIGHT_POS[1]
        to_pool = PixelArray(ms.filtered_[0][pool_f*pool_x:pool_f*(pool_x+1), pool_f*pool_y:pool_f*(pool_y+1)], stroke_width=2, stroke_color=WHITE)
        to_pool.match_height(ms.conv_highlight_2).move_to(ms.conv_highlight_2)
        to_pool.add_pixel_values(color=SATURATED_RED)
        to_pool.set_z_index(ms.conv_highlight_2.z_index + 1)
        grid_factor_highlight = VGroup(
            ms.conv_highlight_2.copy() for _ in range(16)
        ).set_stroke(WHITE, 3).arrange_in_grid(4,4, buff=0).move_to(pooling_example_g[0]).set_z_index(pooling_example_g[0].z_index + 0.5)
        pooling_example_g.add(grid_factor_highlight)
        
        self.play(Create(grid_factor_highlight, run_time=0.5))
        self.play(Create(to_pool.pixel_array), Create(to_pool.pixel_values))
        self.play(
            pooling_example_g.animate.next_to(max_pooling_title, DOWN, buff=0.5),
            to_pool.animate.scale_to_fit_height(3.75).set_x(0).to_edge(DOWN, buff=0.5),
        )

        to_pool.add_brackets(left=r"(", right=r")", color=BLACK)
        max_label = MathTex(r'\max', color=BLACK, font_size=64).next_to(to_pool.brackets, LEFT)
        equal_label = MathTex(r'=', color=BLACK, font_size=64).next_to(to_pool.brackets, RIGHT)
        max_id = np.argmax(to_pool.array)  # index into the flattened array
        max_result = to_pool.pixel_array[max_id].copy().set_stroke(GOLD,3).set_fill(opacity=0)
        
        self.play(
            Succession(
                FadeIn(max_label, to_pool.brackets, equal_label),
                Wait(0.5),
                Create(max_result)
            )
        )
        max_result.set_fill(opacity=1).add(to_pool.pixel_values[max_id].copy())

        def put_result_into_position(mob):
            mob.match_height(ms.pooling_highlight_1).move_to(ms.pooling_highlight_1)
            return mob
        
        self.play(
            Succession(
                ApplyMethod(max_result.next_to, equal_label, RIGHT),
                Wait(0.5),
                ApplyFunction(put_result_into_position, max_result)
            )
        )
        
        # SLIDE 30:  ===========================================================
        # MAX POOLING A SECOND TIME
        self.next_slide(
            notes=
            '''[...]
            '''
        )
        # Do it a second time
        ms.conv_pool_gizmo[0].add_updater(
            lambda mob: mob.put_start_and_end_on(
                ms.conv_highlight_2.get_corner(UL), ms.pooling_highlight_1.get_corner(UL)
            )
        )
        ms.conv_pool_gizmo[1].add_updater(
            lambda mob: mob.put_start_and_end_on(
                ms.conv_highlight_2.get_corner(DR), ms.pooling_highlight_1.get_corner(DR)
            )
        )

        self.play(
            FadeOut(to_pool.pixel_array, to_pool.pixel_values, max_result),
            ms.conv_highlight_2.animate.shift(LEFT*ms.conv_highlight_2.width),
            ms.pooling_highlight_1.animate.shift(LEFT*ms.pooling_highlight_1.width)
        )

        pool_f, pool_x, pool_y = ms.pooling_factor, ms.POOLING_HIGHLIGHT_POS[0], ms.POOLING_HIGHLIGHT_POS[1]-1
        to_pool_2 = PixelArray(ms.filtered_[0][pool_f*pool_x:pool_f*(pool_x+1), pool_f*pool_y:pool_f*(pool_y+1)], stroke_width=2, stroke_color=WHITE)
        to_pool_2.match_height(ms.conv_highlight_2).move_to(ms.conv_highlight_2)
        to_pool_2.add_pixel_values(color=SATURATED_RED)
        to_pool_2.set_z_index(ms.conv_highlight_2.z_index + 1)
        
        self.play(Create(to_pool_2.pixel_array), Create(to_pool_2.pixel_values))
        self.play(to_pool_2.animate.match_height(to_pool.pixel_array).move_to(to_pool.pixel_array))

        max_id = np.argmax(to_pool_2.array)  # index into the flattened array
        max_result = to_pool_2.pixel_array[max_id].copy().set_stroke(GOLD,3).set_fill(opacity=0)
        
        self.play(Create(max_result))
        max_result.set_fill(opacity=1).add(to_pool_2.pixel_values[max_id].copy())

        self.play(
            Succession(
                ApplyMethod(max_result.next_to, equal_label, RIGHT),
                Wait(0.5),
                ApplyFunction(put_result_into_position, max_result)
            )
        )

        # SLIDE 31:  ===========================================================
        # HIGHLIGHT POOLING OUTPUT MATRICES IN FULL SCHEME
        self.next_slide(
            notes=
            '''This layer produces smaller matrices, one for each filter.
            '''
        )
        ms.conv_pool_gizmo.clear_updaters()
        pooling_example_g.remove(grid_factor_highlight)
        self.play(
           FadeOut(max_pooling_title, to_pool.brackets, max_label, equal_label, to_pool_2, max_result, grid_factor_highlight),
           AnimationGroup(Wait(0.5), AnimationGroup(mob.animate.restore() for mob in pooling_example_g), lag_ratio=1),
           AnimationGroup(Wait(1), FadeIn(*[mob for mob in ms.submobjects if mob not in (*pooling_example_g, ms.conv_layer, ms.pooling_layer)], ms.conv_layer[1:], ms.pooling_layer[1:]), lag_ratio=1),
        )

        pooled_matrices_highlight = SurroundingRectangle(ms.pooling_layer, color=GOLD, stroke_width=4, buff=0.5, corner_radius = 0.25)
        self.play(Create(pooled_matrices_highlight))

        # SLIDE 32:  ===========================================================
        # POOLED MATRICES ARE EXTRACTED, FLATTENING TITLE APPEARS
        self.next_slide(
            notes=
            '''Finally we need to map the matrices resulting from max pooling
            into a vector that a dense neural network can process. This
            transformation is made by flattening.
            '''
        )
        self.play(FadeOut(*[mob for mob in ms.submobjects if mob not in [ms.pooling_layer]], pooled_matrices_highlight))

        flattening_title = SlideTitle('Flattening')
        self.play(ms.pooling_layer.animate.scale(3).arrange(UL, buff=0.1).move_to(TITLED_CENTER), Write(flattening_title))
        pooled_PA = VGroup(
            PixelArray(ms.pooled_[i], stroke_color=None, stroke_width=0).match_height(ms.pooling_layer[i]).move_to(ms.pooling_layer[i]).set_z_index(ms.pooling_layer[i].z_index)
            for i in range(5)
        )
        self.add(pooled_PA)
        
        # SLIDE 33:  ===========================================================
        # CREATE BRACES WITH MATRIX DIMENSIONS
        self.next_slide(
            notes=
            ''' In the example, flattening simply takes all the entries from the
            5 matrices, each of size 4x4, ...
            '''
        )
        brace_config = {'color': DARK_BLUE}
        brace_width = Brace(pooled_PA[2], UP, **brace_config)
        w = Text("4", color=BLACK, font_size=48, font=SANS_SERIF_FONT)
        brace_width.put_at_tip(w)
        brace_height = Brace(pooled_PA[2], RIGHT, **brace_config)
        h = w.copy()
        brace_height.put_at_tip(h)
        brace_length = BraceBetweenPoints(pooled_PA[-1].get_corner(DL), pooled_PA[0].get_corner(DL), DL, **brace_config)
        l = Text("5", color=BLACK, font_size=48, font=SANS_SERIF_FONT)
        brace_length.put_at_tip(l)

        self.play(
            Succession(
                FadeIn(brace_length, l),
                Wait(0.5),
                FadeIn(brace_width, w),
                FadeIn(brace_height, h),
            )
        )

        # SLIDE 34:  ===========================================================
        # REMOVE BRACES WITH MATRIX DIMENSIONS
        # MATRICES ARE FLATTENED AND CONCATENATED
        self.next_slide(
            notes=
            '''..., resulting from max pooling and lines them in a vector of
            length 5x16=80, which is the dense layer input.
            '''
        )
        self.play(
            AnimationGroup(
                FadeOut(ms.pooling_layer, brace_width, w, brace_height, h, brace_length, l),
                AnimationGroup(
                    p.pixel_array.animate.arrange(RIGHT, buff=0).move_to(p)
                    for p in pooled_PA
                ),
                Wait(0.5),
                lag_ratio=1
            )
        )
        self.play(pooled_PA.animate.arrange(LEFT, buff=0).scale_to_fit_width(0.9*FRAME_WIDTH).move_to(TITLED_CENTER))
        
        pooled_PA.set_z_index(1)
        pooled_PA_outline = SurroundingRectangle(pooled_PA, color=DARK_BLUE, buff=0, stroke_width=6).set_z_index(pooled_PA.z_index-1)
        brace_tot = Brace(pooled_PA, UP, **brace_config)
        tot = Text("80", color=BLACK, font_size=48, font=SANS_SERIF_FONT)
        brace_length.put_at_tip(l)

        self.play(Create(pooled_PA_outline), FadeIn(brace_tot, tot))
        pooled_PA.add(pooled_PA_outline)

        # SLIDE 35:  ===========================================================
        # RETURN TO CNN SCHEME, PUTTING FLATTENED VECTOR IN CORRECT POSITION
        self.next_slide(
            notes=
            '''Now that we have examined each layer of the convolutional neural
            network, the next step is to select the optimal adjustable
            coefficients.
            '''
        )
        ms.restore()
        ms.move_to(TITLED_CENTER)

        self.play(
            AnimationGroup(
                FadeOut(brace_tot, tot, flattening_title),
                pooled_PA.animate.rotate(-PI/2).match_height(ms.flattened_vector).stretch_to_fit_width(ms.flattened_vector.width).move_to(ms.flattened_vector),
                FadeIn(*[mob for mob in ms.submobjects if mob not in (ms.flattened_vector)], CNN_title),
                lag_ratio = 0.5
            )
        )

        # SLIDE 36:  ===========================================================
        # SCHEME OF ALL CNN ADJUSTABLE COEFFICIENTS SHOWN
        self.next_slide(
            notes=
            '''As we have seen, the convolutional layer and the dense neural
            network depend on a certain number of them, such as the entries of
            each filter, the weights and biases of the dense layer.
            '''
        )
        self.play(FadeOut(*[mob for mob in self.mobjects]))
        self.clear()

        # Create the scheme with the adjustable coefficients
        # CNN_t =  LayerTitle('Convolutional Neural Network')
        adjustable_coeff_t =  Text('Learnable coefficients:', font=SANS_SERIF_FONT, weight=BOLD, font_size=32, color=BLACK)
        filter_entries_t = Text('Filter entries:', font=SANS_SERIF_FONT, font_size=28, color=BLACK)
        weights_and_biases_t =  Text('Weights and biases of the dense layer:', font=SANS_SERIF_FONT, font_size=28, color=BLACK)

        # Filter entries
        node_colors = [RED, GREEN, ORANGE]
        matrix_config = {'v_buff':0.6, 'h_buff':0.6, 'bracket_h_buff':0.1, 'bracket_v_buff':0.1}
        filter_matrix = Matrix([[f'l_{3*i +j}' for j in range(3)] for i in range(3)], color=BLACK, **matrix_config)
        filters: VGroup[Matrix] = VGroup(filter_matrix.copy().set_color(c) for c in node_colors)
        filters.add(MathTex(r'\dots', color=BLACK, font_size=48))
        filters.arrange(RIGHT, buff=0.5)

        # NN parameters
        weights_eq = MathTex(r'\mathbf{W}=[w_0,w_1,w_2,w_0,w_1,w_2,w_0,w_1,w_2,\dots]', color=BLACK, font_size=48,
                             substrings_to_isolate=[f'w_{i}' for i in range(3)])
        substr_bias_eq = [r'\mathbf{b}','=', '[', ']', r'\dots', ',']  # Needed for later
        bias_eq = MathTex(r'\mathbf{b}=[b,b,b,\dots]', color=BLACK, font_size=48,
                          substrings_to_isolate=substr_bias_eq)  # isolating 'b' breaks here
        
        # Arrange vertically
        # VGroup(CNN_t, adjustable_coeff_t, filter_entries_t).arrange(DOWN, buff=0.5)
        filters.next_to(filter_entries_t, DOWN, buff=0.3)
        weights_and_biases_t.next_to(filters, DOWN, buff=0.5)
        VGroup(weights_eq, bias_eq).arrange(DOWN, buff=0.5).next_to(weights_and_biases_t, DOWN, buff=0.25)

        # Add surrounding rectangle
        CNN_coeff = VGroup(adjustable_coeff_t, filter_entries_t, filters, weights_and_biases_t, weights_eq, bias_eq).center()
        CNN_coeff_rect = SurroundingRectangle(CNN_coeff, stroke_color=BLUE, stroke_width=4, fill_opacity=0, buff=0.5, corner_radius=0.25)
        adjustable_coeff_t.next_to(CNN_coeff_rect, UP, buff=0.5)
        CNN_coeff.add(CNN_coeff_rect)

        self.play(FadeIn(CNN_coeff))

        # SLIDE 37:  ===========================================================
        # TRAINING TITLE APPEARS
        self.next_slide(
            notes=
            '''To find the optimal values, we perform the so-called training
            procedure. This involves showing the network many examples of
            already labeled images and adjusting the network coefficients based
            on the difference between the network's digit predictions and the
            ground truth. Now, let's take a look at how the training procedure
            is carried out.
            '''
        )
        training_title = SlideTitle('Training')
        
        self.play(
            Succession(
                CNN_coeff.animate.shift(DOWN*0.5),
                Write(training_title)
            )
        )

        # SLIDE 38:  ===========================================================
        # EXAMPLES OF LABELED IMAGES SHOWN
        self.next_slide(
            notes=
            ''' Suppose we have a collection of images and their actual values
            stored in a dataset. The Training process is made of 5 steps.
            '''
        )
        self.play(FadeOut(CNN_coeff))

        ground_truth_config = {'stroke_color':GREEN_D,'fill_color':WHITE, 'text_kwargs':{'fill_color': GREEN_D, 'stroke_color':GREEN_D}}

        training_sample = Group(
            *[PixelImage(rf'Assets\W5\mnist{i}.png').scale_to_fit_height(0.25*FRAME_HEIGHT) for i in (3,5,8)]
        ).arrange(RIGHT, buff=1)
        training_labels = VGroup(
            DigitRecognitionOutputCircle(i, stroke_width=12, **ground_truth_config).scale_to_fit_height(im.height).next_to(im, DOWN, buff=0.5)
            for i, im in zip((3,5,8), training_sample)
        )
        training_data = Group(training_sample, training_labels).move_to(TITLED_CENTER)

        self.play(FadeIn(training_data))
        
        # SLIDE 39:  ===========================================================
        # 1. INITIALIZATION TITLES APPEARS
        # FILTERS, WEIGHTS AND BIASES ARE SUBSTITUTED BY RANDOM VALUES
        self.next_slide(
            notes=
            '''Step 1 is initialization. We assign random values to the
            parameters of the neural network.
            '''
        )
        self.play(FadeOut(training_title, training_data))
        initialization_title = SlideTitle('1. Initialization')

        # Remake the adjustable coefficients scheme removing some parts for space
        CNN_coeff_remix = VGroup(filter_entries_t, filters, weights_and_biases_t, weights_eq, bias_eq)
        CNN_coeff_remix.add(SurroundingRectangle(CNN_coeff_remix, stroke_color=BLUE, stroke_width=4, fill_opacity=0, buff=0.3, corner_radius=0.25))
        CNN_coeff_remix.move_to(TITLED_CENTER)

        # ARROWS CONFIG
        line_config = {'color': DARK_BLUE, 'stroke_width':4}
        broken_arrow_config = {'color': DARK_BLUE, 'stroke_width':4, 'add_tip':False}

        # Create random initial values
        np.random.seed(0)
        def init_function(shape=None):
            return np.round(7.5*(2*np.random.random(shape)-1),1)
        value_display_config = {'include_sign':True, 'font_size':28, 'num_decimal_places':1}
        initialized_filters = VGroup(
            Matrix(
                init_function((3,3)),  **matrix_config,
                element_to_mobject=ValueDisplay, element_to_mobject_config={'include_sign':True, 'font_size':24, 'num_decimal_places':1},
            ).set_color(node_colors[i]).move_to(filters[i]) for i in range(3)
        )
        for filter in initialized_filters:
            filter.remove(filter.brackets)

        initialized_weights = VGroup(
            ValueDisplay(init_function(), **{'include_sign':True, 'font_size':28, 'num_decimal_places':1}).set_color(BLACK).move_to(w)
            for w in weights_eq[1:19:2]
        ).align_to(weights_eq[-1][1], DOWN)  # align with the \dots (weights_eq[-1] is ',\dots]')

        # For the biases we need more space; this requires much more work by creating a target "empty" expression
        bees = bias_eq[3:8:2]
        not_bees = VGroup(*bias_eq[:3], *bias_eq[4:9:2], *bias_eq[-2:])
        blank_space = r'\hspace{0.5cm}'
        beq2 = MathTex(r'\mathbf{b}=['+ ','.join([blank_space for _ in range(3)]) + r',\dots]', color=BLACK, font_size=48,
                       substrings_to_isolate=substr_bias_eq).move_to(bias_eq)
        initialized_biases = VGroup(
            ValueDisplay(
                init_function(), **{'include_sign':True, 'font_size':28, 'num_decimal_places':1}
            ).set_color(BLACK).match_x(beq2[2+2*i:3+2*(i+1):2])
            for i in range(3)
        ).align_to(beq2[-2], DOWN)

        learnable_coeffs = VGroup(*[elem for filter in initialized_filters for elem in filter.elements], *initialized_weights, *initialized_biases)
        learnable_coeffs.suspend_updating()  # otherwise the transform does not work

        self.play(
            Succession(
                FadeIn(CNN_coeff_remix), Write(initialization_title),
                Wait(0.2),
                AnimationGroup(
                    ReplacementTransform(filters[i].elements, initialized_filters[i].elements) for i in range(3)
                ),
                Wait(0.3),
                AnimationGroup(
                    ReplacementTransform(w, wi) for w, wi in zip(weights_eq[1:19:2], initialized_weights)
                ),
                Wait(0.3),
            )
        )
        self.play(TransformMatchingShapes(not_bees, beq2), ReplacementTransform(bees, initialized_biases))

        # SLIDE 40:  ===========================================================
        # 2. FORWARD PROPAGATION TITLE
        # INPUT DATA APPEARS, THEN ARROW FROM IT TO CENTRAL SCHEME
        # OUTPUT LAYER APPEARS, THEN ARROW TO THE PREDICTION
        self.next_slide(
            notes=
            '''Step 2 is forward propagation. Here, input data (the image) is
            passed through the network, generating a prediction.
            '''
        )
        self.play(FadeOut(initialization_title))
        forward_prop_title = SlideTitle('2. Forward Propagation')
        self.play(Write(forward_prop_title))

        forward_input = ms.input.copy().scale_to_fit_height(0.1*FRAME_HEIGHT).next_to(CNN_coeff_remix, LEFT)
        forward_input.set_x((-FRAME_WIDTH/2 + CNN_coeff_remix.get_left()[0])/2)  # middle point
        input_label =  LayerTitle('Input').next_to(forward_input, UP, buff=0.3)

        output_layer = DigitRecognitionOutputLayer().set_y(TITLED_CENTER[1]).shift(DOWN*0.7)
        output_layer.set_x((+FRAME_WIDTH/2 + CNN_coeff_remix.get_right()[0])/2)  # middle point
        output_layer.save_state()
        output_label = LayerTitle('Output').next_to(output_layer, UP, buff=0.3)

        # Start Creating the arrows
        input_arrow = Arrow(forward_input.get_right(), CNN_coeff_remix.get_left(), buff=0, max_stroke_width_to_length_ratio=20, **line_config)
        
        def make_output_arrow(digit: int, start=None):
            start = CNN_coeff_remix.get_right() if start is None else start
            arr = BrokenArrow(start, output_layer[digit].circle.point_at_angle(3/4*PI), 
                              n_turns=2, first_direction='h', **broken_arrow_config)
            arr.keypoints[2] += UP*(arr.keypoints[3][0]-arr.keypoints[2][0])
            arr.set_points_as_corners(arr.keypoints)
            arr.add_tip(input_arrow.tip.copy())
            return arr

        output_arrow = make_output_arrow(6)

        self.play(
            Succession(
                FadeIn(forward_input, input_label),
                Create(input_arrow),
                FadeIn(output_layer, output_label),
                Create(output_arrow),
            )
        )
        self.play(output_layer.Activate(6))

        # SLIDE 41:  ===========================================================
        # 2. LOSS COMPUTATION TITLE
        # GROUND TRUTH AND LOSS SYMBOL APPEAR
        # ARROWS FROM PREDICTION AND GROUND TRUTH TO LOSS APPEAR
        self.next_slide(
            notes=
            '''Step 3 is the calculation of the loss, which is a measure of the
            difference between the prediction (6) and the actual value (8).
            '''
        )
        self.play(FadeOut(forward_prop_title))

        loss_title = SlideTitle('3. Loss Computation')
        shift_into_place =  (loss_title.get_bottom() - CNN_coeff_remix.get_top() - 0.25)[1]*UP
        target_output_arrow = make_output_arrow(6, start=CNN_coeff_remix.get_right() +shift_into_place)
        CNN_coeff_remix.remove(bias_eq).add(beq2, initialized_biases)  # update the scheme to have the new equation
        self.play(
            Write(loss_title),
            Group(forward_input, input_label, input_arrow, CNN_coeff_remix).animate.shift(shift_into_place),
            output_arrow.animate.become(target_output_arrow)
            )
        
        # loss function and its gradient
        loss_symbol = MathTex(r'\mathcal{L}', color=BLACK, font_size=64)
        square_ = Square(side_length=loss_symbol.height + 0.6, stroke_color=PURPLE, stroke_width=4, fill_opacity=0)
        loss_symbol.add_to_back(square_.move_to(loss_symbol))
        gradient_loss_symbol = MathTex(r'\nabla \mathcal{L}', color=BLACK, font_size=64)
        gradient_loss_symbol.add_to_back(square_.copy().move_to(gradient_loss_symbol))
        VGroup(gradient_loss_symbol, loss_symbol).arrange(DOWN, buff=0.3).next_to(CNN_coeff_remix, DOWN, buff=0.3)

        def make_output_loss_arrow(digit: int):
            arr = BrokenArrow(output_layer[digit].circle.point_at_angle(5/4*PI), loss_symbol.get_right(),
                              n_turns=2, first_direction='h', **broken_arrow_config)
            target_x = output_arrow.keypoints[1][0]
            for i in range(1,3):
                arr.keypoints[i][0] = target_x
            arr.keypoints[1] += DOWN*(arr.keypoints[0][0]-arr.keypoints[1][0])
            arr.set_points_as_corners(arr.keypoints)
            arr.add_tip(input_arrow.tip.copy())
            return arr
        
        output_loss_arrow = make_output_loss_arrow(6)

        loss_to_gradient_arrow = Line(loss_symbol.get_top(), gradient_loss_symbol.get_bottom(), **line_config)
        loss_to_gradient_arrow.add_tip(input_arrow.tip.copy())
        
        # Ground truth of the input
        ground_truth = DigitRecognitionOutputCircle(8, **ground_truth_config).next_to(forward_input, DOWN, buff=3.5)
        ground_truth_label = Paragraph('Ground\ntruth', color=BLACK, font=SANS_SERIF_FONT, font_size=32, weight=BOLD, alignment='center'
                                       ).scale(0.8).next_to(ground_truth, UP, buff=0.3)
        ground_truth_loss_arrow = BrokenArrow(ground_truth.get_bottom(), loss_symbol.get_left(), **broken_arrow_config,
                                              n_turns=1, first_direction='v')
        ground_truth_loss_arrow.add_tip(input_arrow.tip.copy())

        self.play(
            Succession(
                FadeIn(ground_truth, ground_truth_label),
                FadeIn(loss_symbol),
                AnimationGroup(Create(output_loss_arrow), Create(ground_truth_loss_arrow))
            )
        )

        # SLIDE 42:  ===========================================================
        # 4. BACKPROPAGATION TITLE
        # LOSS GRADIENT SYMBOL APPEARS AND ARROW TO IT FROM LOSS
        self.next_slide(
            notes=
            '''Step 4 is backpropagation that computes the gradient of the loss
            with respect to each weight and bias, moving backwards through the
            network.
            '''
        )
        self.play(FadeOut(loss_title))
        backpropagation_title = SlideTitle('4. Backpropagation')
        self.play(Write(backpropagation_title))

        self.play(FadeIn(gradient_loss_symbol))
        self.play(Create(loss_to_gradient_arrow))
        
        # SLIDE 43:  ===========================================================
        # 5. UPDATE TITLE
        # ARROW FROM LOSS GRADIENT TO MAIN SCHEME
        # ADJUSTABLE COEFFICIENTS ARE MODIFIED
        self.next_slide(
            notes=
            '''Finally, step 5 consists in updating the weights and biases in
            the direction that minimizes the loss, that is the discrepancy
            between the digits prediction and the actual values of the digits.
            '''
        )
        self.play(FadeOut(backpropagation_title))
        update_title = SlideTitle('5. Update')
        self.play(Write(update_title))

        update_arrow = Line(gradient_loss_symbol.get_top(), CNN_coeff_remix.get_bottom(), **line_config)
        update_arrow.add_tip(input_arrow.tip.copy())

        self.play(Create(update_arrow))

        learnable_coeffs.resume_updating()  # otherwise the transform does not work
        self.play(
            UpdateLearnableCoefficients(learnable_coeffs, u_range=1.5)
        )

        # SLIDE 44:  ===========================================================
        # TITLE DISAPPEARS
        # REPLACE INPUT AND GROUND TRUTH AND REPEAT THE PROCEDURE TWICE
        self.next_slide(
            notes=
            '''Steps 2-5 are repeated many times with different input samples,
            gradually improving the network's prediction performance.
            Summarizing, the training of a neural network is nothing else than
            the minimization of the loss with respect to the values of the
            parameters.
            '''
        )
        CREATE_ARROW_RUNTIME = 0.5
        # Restore to previous state to repeat
        self.play(FadeOut(update_title))
        self.play(FadeIn(training_title))
        self.play(
            FadeOut(
                forward_input, ground_truth, input_arrow, output_arrow, output_loss_arrow,
                ground_truth_loss_arrow, loss_to_gradient_arrow, update_arrow
            ),
            output_layer.animate.restore()
        )

        # Create new arrows for different output
        output_arrow_2 = make_output_arrow(3)
        output_loss_arrow_2 = make_output_loss_arrow(3)
        output_arrow_3 = make_output_arrow(3)
        output_loss_arrow_3 = make_output_loss_arrow(3)

        # new input
        forward_input_2 = PixelImage(r'Assets\W5\mnist5.png', add_outline=True, outline_kwargs={'color':DARK_BLUE, 'stroke_width':6}).match_height(forward_input).move_to(forward_input)
        ground_truth_2 = DigitRecognitionOutputCircle(5, **ground_truth_config).move_to(ground_truth)
        forward_input_3 = PixelImage(r'Assets\W5\mnist3.png', add_outline=True, outline_kwargs={'color':DARK_BLUE, 'stroke_width':6}).match_height(forward_input).move_to(forward_input)
        ground_truth_3 = DigitRecognitionOutputCircle(3, **ground_truth_config).move_to(ground_truth)
        
        self.play(
            Succession(
                FadeIn(forward_input_2, ground_truth_2), 
                Create(input_arrow, run_time=CREATE_ARROW_RUNTIME),
                Create(output_arrow_2, run_time=CREATE_ARROW_RUNTIME),
            )
        )
        self.play(output_layer.Activate(3))  # doe snot work in the middle of a succession
        self.play(
            Succession(
                AnimationGroup(Create(output_loss_arrow_2), Create(ground_truth_loss_arrow), run_time=CREATE_ARROW_RUNTIME),
                Create(loss_to_gradient_arrow, run_time=CREATE_ARROW_RUNTIME),
                Create(update_arrow, run_time=CREATE_ARROW_RUNTIME)
            )
        )
        self.play(
            UpdateLearnableCoefficients(learnable_coeffs, u_range=1)
        )

        # Reset and repeat
        self.play(
            FadeOut(forward_input_2, ground_truth_2, input_arrow, output_arrow_2, ground_truth_loss_arrow, loss_to_gradient_arrow, update_arrow, output_loss_arrow_2),
            output_layer.animate.restore()
        )
        self.play(
            Succession(
                FadeIn(forward_input_3, ground_truth_3),
                Create(input_arrow, run_time=CREATE_ARROW_RUNTIME),
                Create(output_arrow_3, run_time=CREATE_ARROW_RUNTIME),
            )
        )
        self.play(output_layer.Activate(3))  # does not work in the middle of a succession
        self.play(
            Succession(
                AnimationGroup(Create(output_loss_arrow_3), Create(ground_truth_loss_arrow), run_time=CREATE_ARROW_RUNTIME),
                Create(loss_to_gradient_arrow, run_time=CREATE_ARROW_RUNTIME),
                Create(update_arrow, run_time=CREATE_ARROW_RUNTIME)
            )
        )
        self.play(
            UpdateLearnableCoefficients(learnable_coeffs, u_range=0.75)
        )

        # SLIDE 45:  ===========================================================
        # FULL CNN SCHEME REAPPEARS
        self.next_slide(
            notes=
            '''In the following section, we are going to see this in action by
            implementing in matlab and python a digit classifier based on image
            processing and convolutional neural networks.
            '''
        )
        self.play(FadeOut(*[mob for mob in self.mobjects]))
        self.clear()
        ms.restore()

        self.play(FadeIn(ms))
        self.wait(0.05)
 