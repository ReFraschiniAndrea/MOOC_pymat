import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from mooc_utils import *
from mooc_utils.colab import *
from mooc_utils.matlab import *
from W5Anim import *
from PIL import Image

config.update(RELEASE_CONFIG)


class W5Wrapup_slides(MOOCSlide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # DIGIT RECOGNITION SCHEME APPEARS
        self.next_slide(
            notes=
            '''In this project, we have learned how to recognize digits in
            images.
            '''
        )
        digit = np.array(Image.open(r'Assets\W5\mnist\mnist80.png'))
        ms = CNNDigitRecognitionScheme(digit, input_label=8, n_filters=5, pooling_factor=7,
                                       pixel_size=(1.5/28, 1.5/28, 0.125, 0.125), horizontal_spacing=0.8,
                                       highlights_kwargs={'color': GOLD, 'stroke_width': 3},
                                       outline_kwargs={'color': DARK_BLUE, 'stroke_width': 6})
        ms.move_to(TITLED_CENTER)
        ms.save_state()

        digit_recognition_algo = Paragraph('Digit Recognition\nAlgorithm', font=SANS_SERIF_FONT, weight=BOLD, font_size=48, color=DARK_BLUE, alignment='center')
        digit_recognition_algo.background_rectangle = SurroundingRectangle(
            digit_recognition_algo,
            fill_color=WHITE, fill_opacity=1, stroke_color=DARK_BLUE, stroke_width=4,
            corner_radius=0.25, buff=0.5
        )
        digit_recognition_algo.add_to_back(digit_recognition_algo.background_rectangle)

        intro_scheme = Group(Group(ms.input, ms.input_title), digit_recognition_algo, Group(ms.output_layer, ms.output_layer_title)).arrange(RIGHT, buff=0.75)
        digit_recognition_algo.match_y(ms.input)

        left_arr = Arrow(ms.input.get_right(), digit_recognition_algo.get_left(), color=DARK_BLUE, stroke_width=4, buff=0)
        right_arr = BrokenArrow(digit_recognition_algo.get_right(), ms.output_layer[8].get_left(), n_turns=2, color=DARK_BLUE, add_tip=False)
        right_arr.add_tip(left_arr.tip.copy())
        intro_scheme.add(left_arr, right_arr) 
        # save arrangement for first exercise explanation
        intro_scheme_for_later = Group(ms.input, ms.input_title, left_arr, digit_recognition_algo).copy()
        
        self.play(FadeIn(intro_scheme[:-1]))  # Everything but the output arrow
        self.play(Create(right_arr, run_time=0.5))
        self.play(ms.output_layer.Activate(8), run_time=0.5)

        # SLIDE 02:  ===========================================================
        # FULL CNN SCHEME APPEARS
        # CNN TITLE APPEARS
        self.next_slide(
            notes=
            '''We did it by constructing and training a so called convolutional
            neural network, that receives the image as input and returns its
            classification, using well-known libraries and toolkits in both
            Python and MATLAB.
            '''
        )
        CNN_title = SlideTitle("Convolutional Neural Network")
        _mobs_to_restore = [ms.input, ms.input_title, ms.output_layer, ms.output_layer_title]
        self.play(
            AnimationGroup(
                FadeOut(left_arr, right_arr, digit_recognition_algo),
                AnimationGroup(
                    *[mob.animate.restore() for mob in _mobs_to_restore],
                ),
                AnimationGroup(
                    Write(CNN_title),
                    FadeIn(
                        *[mob for mob in ms.submobjects if mob not in [*_mobs_to_restore, ms.output_arrow]]
                    ),
                ),
                lag_ratio=0.5
            )
        )
        self.play(Create(ms.output_arrow, run_time=0.5))
        self.play(ms.output_layer.Activate(8), run_time=0.5)

        # SLIDE 03:  ===========================================================
        # ARCHITECTURE PYTHON CODE APPEARS
        self.next_slide(
            notes=
            '''In particular, in Python, we used Tensorflow - Keras to define
            and train the convolutional neural network.
            '''
        )
        self.play(FadeOut(ms, CNN_title))

        cnn_python_code = ColabCodeWithLogo(
            r'''
            # Define the model
            CNN = keras.Sequential([
                keras.Input(shape=(img_height, img_width, 1)),
                layersModule.Conv2D(32, (3, 3),
                    activation='relu', padding='same'),
                layersModule.MaxPooling2D(pool_size=(7, 7)),
                layersModule.Flatten(),
                layersModule.Dense(10, activation='softmax')
            ])
            '''
        )
        cnn_matlab_code = MatlabCodeWithLogo(
            r'''
            % Define the model
            layers = [
                imageInputLayer([img_height img_width 1])
                convolution2dLayer(3, 32, 'Padding', 'same')
                reluLayer
                maxPooling2dLayer(7, 'Stride', 7)
                flattenLayer
                fullyConnectedLayer(10)
                softmaxLayer
                classificationLayer
            ];
            '''
        )
        cnn_python_code_focus = cnn_python_code.copy().center()
        _left_align = cnn_matlab_code.get_left()
        cnn_matlab_code.codeMobject.window.stretch_to_fit_width(cnn_python_code.width).align_to(_left_align, LEFT)
        Group(cnn_python_code, cnn_matlab_code).arrange(buff=0.5).scale_to_fit_width(0.95*FRAME_WIDTH)
        cnn_matlab_code.shift(UP*(cnn_python_code.codeMobject.get_top()[1] - cnn_matlab_code.codeMobject.get_top()[1]))
        Group(cnn_python_code, cnn_matlab_code).center()

        self.play(FadeIn(cnn_python_code_focus))

        # SLIDE 04:  ===========================================================
        # ARCHITECTURE MATLAB CODE APPEARS
        self.next_slide(
            notes=
            '''While, in Matlab, we constructed the same architecture by means
            of the Deep Learning toolbox functionalities.
            '''
        )
        self.play(ReplacementTransform(cnn_python_code_focus, cnn_python_code))
        self.play(FadeIn(cnn_matlab_code))

        # SLIDE 05:  ===========================================================
        # CNN SCHEME BROUGHT IN FROM TOP
        self.next_slide(
            notes=
            '''The main steps and ingredients are similar in the two libraries:
            '''
        )
        DSS = DynamicSplitScreen(WHITE, WHITE)
        ms.save_state(); ms.scale(0.75)
        DSS.add_side_obj(ms)
        DSS.add_main_obj(Group(cnn_python_code, cnn_matlab_code))

        self.play(DSS.bringIn())

        # SLIDE 06:  ===========================================================
        # INPUT LINE HIGHLIGHT
        # INPUT BRACES DIMENSION APPEAR
        self.next_slide(
            notes=
            '''First, we defined the size of our input images, height and width
            in pixels
            '''
        )
        input_braces = get_labeled_braces(
            ms.input, LEFT, '28', DOWN, '28', label_config={'font_size': 24}
        )
        input_code_highlight = VGroup(
            HighlightRectangle(cnn_python_code[2]),
            HighlightRectangle(cnn_matlab_code[2]),
        )

        self.play(FadeIn(input_braces), Create(input_code_highlight))

        # SLIDE 07:  ===========================================================
        # CONV LAYER AND CORRESPONDING CODE HIGHLIGHT
        self.next_slide(
            notes=
            '''Then, we defined the convolutional layers: in our example they
            are 32, and operate with kernels of size 3 by 3, using padding.
            '''
        )
        layer_highlight_config = {'height':5.5, 'v_buff':0.25, 'h_buff':0.15}
        conv_layer_highlight = ms.get_layer_highlight('conv', **layer_highlight_config)
        conv_layer_code_highlight = VGroup(
            HighlightRectangle(cnn_python_code[3:5]),
            HighlightRectangle(cnn_matlab_code[3:5]),
        )
        self.play(
            Succession(
                FadeOut(input_braces),
                AnimationGroup(
                    Create(conv_layer_highlight),
                    ReplacementTransform(input_code_highlight, conv_layer_code_highlight)
                )
            )
        )
        # SLIDE 08:  ===========================================================
        # POOLING LAYER AND CORRESPONDING CODE HIGHLIGHT
        self.next_slide(
            notes=
            '''The max pooling layer is defined by the size of the "pools":
            here, they are 7 by 7, producing 16 compressed values.
            '''
        )
        pool_layer_highlight = ms.get_layer_highlight('pool', **layer_highlight_config)
        pool_layer_code_highlight = VGroup(
            HighlightRectangle(cnn_python_code[5]),
            HighlightRectangle(cnn_matlab_code[5]),
        )

        self.play(
            ReplacementTransform(conv_layer_highlight, pool_layer_highlight),
            ReplacementTransform(conv_layer_code_highlight, pool_layer_code_highlight)
        )
        # SLIDE 09:  ===========================================================
        # FLATTEN LAYER AND CORRESPONDING CODE HIGHLIGHT
        self.next_slide(
            notes=
            '''We then have the flatten layer,
            '''
        )
        flatten_layer_highlight = ms.get_layer_highlight('flat', **layer_highlight_config)
        flatten_layer_code_highlight = VGroup(
            HighlightRectangle(cnn_python_code[6]),
            HighlightRectangle(cnn_matlab_code[6]),
        )
        self.play(
            ReplacementTransform(pool_layer_highlight, flatten_layer_highlight),
            ReplacementTransform(pool_layer_code_highlight, flatten_layer_code_highlight)
        )

        # SLIDE 10:  ===========================================================
        # DENSE LAYER AND CORRESPONDING CODE HIGHLIGHT
        self.next_slide(
            notes=
            '''And finally the fully connected layer: we specify the number of
            outputs, 10 in our case, and the choice of the softmax activation
            function.
            '''
        )
        dense_layer_highlight = ms.get_layer_highlight('dense', **layer_highlight_config)
        dense_layer_code_highlight = VGroup(
            HighlightRectangle(cnn_python_code[7]),
            HighlightRectangle(cnn_matlab_code[7:10]),
        )
        self.play(
            ReplacementTransform(flatten_layer_highlight, dense_layer_highlight),
            ReplacementTransform(flatten_layer_code_highlight, dense_layer_code_highlight)
        )

        # SLIDE 11:  ===========================================================
        # LEARNABLE COEFFICIENTS SCHEME APPEARS
        self.next_slide(
            notes=
            '''The network cannot be used as it is: we need to optimize its
            coefficients.
            '''
        )
        self.play(FadeOut(dense_layer_highlight, ms, cnn_python_code, cnn_matlab_code, dense_layer_code_highlight))
        DSS.reset()
        self.remove(DSS)

        LCNNscheme = LearnableCoefficientsScheme()
        learn_coeff_t = LayerTitle('Learnable coefficients:').next_to(LCNNscheme, UP, buff=0.25)
        LCNNscheme.add(learn_coeff_t).center()
        
        self.play(FadeIn(LCNNscheme))

        # SLIDE 12:  ===========================================================
        # TRAINING OPTION CODE APPEARS
        # LEARNING RATE HIGHLIGHTED
        self.next_slide(
            notes=
            '''First, we define some options for the optimizer, in particular
            the learning rate.
            '''
        )
        self.play(FadeOut(LCNNscheme))

        cnn_options_python_code = ColabCodeWithLogo(
            r'''
            # Select the optmizer
            optimizer = keras.optimizers.SGD(learning_rate=0.01)

            CNN.compile(optimizer=optimizer,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
            ''',
            logo_pos=LEFT
        )
        cnn_options_matlab_code = MatlabCodeWithLogo(
            r'''
            options = trainingOptions(...
                'sgdm', 'InitialLearnRate', 0.01, ...
                'MaxEpochs', 20, 'MiniBatchSize', 16, ...
                'Shuffle', 'every-epoch', ...
                'ValidationData', {X_validation, y_validation}, ...
                'Plots', 'training-progress', 'Verbose', false ...
            );
            ''',
            logo_pos=LEFT
        ).next_to(cnn_options_python_code, DOWN, buff=0.5)
        cnn_options_matlab_code.shift(RIGHT*(cnn_options_python_code.codeMobject.get_left()[0] - cnn_options_matlab_code.codeMobject.get_left()[0]))
        Group(cnn_options_python_code, cnn_options_matlab_code).center()

        learning_rate_highligths = VGroup(
            HighlightRectangle(cnn_options_python_code[1][-19:-1]),
            HighlightRectangle(cnn_options_matlab_code[1][7:-4]),
        )

        self.play(
            Succession(
                FadeIn(cnn_options_python_code, cnn_options_matlab_code),
                Wait(1),
                Create(learning_rate_highligths, lag_ratio=0)
            )
        )

        # SLIDE 13:  ===========================================================
        # MANY LABELED DATASET EXAMPLES APPEAR 
        self.next_slide(
            notes=
            '''Then, we perform training, using a large set of "examples":
            '''
        )
        self.play(FadeOut(cnn_options_python_code, cnn_options_matlab_code, learning_rate_highligths))

        N_SHOWN_SAMPLES = 40
        N_PER_ROW = 10; N_ROWS = N_SHOWN_SAMPLES//N_PER_ROW
        phony_rects = VGroup(Square(0.08*FRAME_HEIGHT) for _ in range(N_SHOWN_SAMPLES*2)).arrange_in_grid(N_ROWS*2, N_PER_ROW, buff=(0.5, 0.2))
        for i in range(1, N_ROWS):
            phony_rects[2*N_PER_ROW*i:].shift(DOWN*0.5)
        phony_rects.center()
        image_phony_rects = [p for i in range(N_ROWS) for p in phony_rects[N_PER_ROW*2*i:N_PER_ROW*(2*i+1)]]
        label_phony_rects = [p for i in range(N_ROWS) for p in phony_rects[N_PER_ROW*(2*i+1):N_PER_ROW*(2*i+2)]]

        # generate random order
        np.random.seed(2)
        sample_files = [f for f in os.listdir(r'Assets\W5\mnist') if f.startswith('mnist')]
        sample_labels = [int(f[-6]) for f in sample_files]
        _shuffled = np.arange(N_SHOWN_SAMPLES)
        np.random.shuffle(_shuffled)
        training_sample = Group(
            *[PixelImage(os.path.join(r'Assets\W5\mnist', sample_files[i])).match_height(rect).move_to(rect)
              for i, rect in zip(_shuffled, image_phony_rects)]
        )

        ground_truth_config = {'stroke_color':GREEN_D,'fill_color':WHITE, 'text_kwargs':{'fill_color': GREEN_D, 'stroke_color':GREEN_D}}
        training_labels = VGroup(
            DigitRecognitionOutputCircle(sample_labels[i], stroke_width=6, **ground_truth_config).match_height(rect).move_to(rect)
            for i, rect in zip(_shuffled, label_phony_rects)
        )
        sample_rects = VGroup(
            SurroundingRectangle(im, lab, color=BLUE, stroke_width=4, buff=0.15, corner_radius=0.25)
            for im, lab in zip(image_phony_rects, label_phony_rects)
        )

        self.play(
            Succession(
                Succession(
                    FadeIn(sample, label, rect, run_time=0),
                    Wait(1.5/40)
                )
                for sample, label, rect in zip(training_sample, training_labels, sample_rects)
            )
        )

        # SLIDE 14:  ===========================================================
        # PYTON AN DMATLAB TRAINING CODE APPEAR
        self.next_slide(
            notes=
            '''in Python, with tensor flow, we use the method "fit", and in
            matlab the corresponding function "trainNetwork".
            '''
        )
        self.play(FadeOut(*[mob for mob in self.mobjects]))

        train_python_code = ColabCodeWithLogo(
            r'''
            # Train the network
            history = CNN.fit(
                X_train, y_train,
                epochs=num_epochs,
                batch_size=batch_size,
                validation_data=(X_validation, y_validation),
                callbacks=[loss_results],
                verbose=False,
            )
            ''',
            logo_pos=LEFT
        )
        train_matlab_code = MatlabCodeWithLogo(
            r'''
            % Train the network
            CNN = trainNetwork(...
                X_train, categorical(y_train), ...
                layers, options ...
            );
            ''',
            logo_pos=LEFT
        ).next_to(train_python_code, DOWN, buff=0.5)
        # _left_align = train_matlab_code.get_left()
        # train_matlab_code.codeMobject.window.match_width(train_python_code).align_to(_left_align, LEFT)
        train_matlab_code.shift(RIGHT*(train_python_code.codeMobject.get_left()[0] - train_matlab_code.codeMobject.get_left()[0]))
        Group(train_python_code, train_matlab_code).center()

        self.play(FadeIn(train_python_code, train_matlab_code))

        # SLIDE 15:  ===========================================================
        # CODE FADES OUT
        self.next_slide(
            notes=
            '''Now it's your turn! Play with the learning rate and observe how
            the model's ability to classify digits changes.
            '''
        )
        self.play(FadeOut(train_python_code, train_matlab_code))

        rigth_arr = left_arr.copy().next_to(intro_scheme_for_later, RIGHT, buff=0)
        question_mark = Text("?", color=DARK_BLUE, font=SANS_SERIF_FONT, weight=BOLD, font_size=128).next_to(rigth_arr, RIGHT)

        self.play(
            Succession(
                FadeIn(intro_scheme_for_later),
                AnimationGroup(
                    GrowArrow(rigth_arr),
                    FadeIn(question_mark)   
                )
            )
        )

        # SLIDE 16:  ===========================================================
        # PYTHON AND MATLAB TRAINING CODES REAPPEAR
        # LEARNING RATE HIGHLIGHT IN BOTH
        self.next_slide(
            notes=
            '''For example, what happens to digit recognition accuracy when you
            select a learning rate of 1.0? And how does the model performance
            change when you lower it to 0.0005?
            '''
        )
        self.play(FadeOut(intro_scheme_for_later, rigth_arr, question_mark))

        self.play(
            Succession(
                FadeIn(cnn_options_python_code, cnn_options_matlab_code),
                Wait(1),
                Create(learning_rate_highligths, lag_ratio=0)
            )
        )

        # SLIDE 17:  ===========================================================
        # LAYERS ADDED TO THE SCHEME
        self.next_slide(
            notes=
            '''You can also construct a deeper convolutional neural network,
            meaning, a network with more layers.
            '''
        )
        self.play(FadeOut(cnn_options_python_code, cnn_options_matlab_code, learning_rate_highligths))

        ms.restore()
        deepms = DeeperCNNDigitRecognitionScheme(
            digit, input_label=8, horizontal_spacing=0.6,
            conv_layers_config = [
            {'n_filters':5, 'pooling_factor':2, 'pixel_size': 0.05, 'offset_scale':1.5},
            {'n_filters':10, 'pooling_factor':2, 'pixel_size': 0.05, 'offset_scale':1.5}
            ],
            highlights_kwargs={'color': GOLD, 'stroke_width': 3},
            outline_kwargs={'color': DARK_BLUE, 'stroke_width': 6}
        )
        deepms.output_layer.activate(8)

        LAYER_T_SCALE = 0.75
        self.play(FadeIn(ms))
        self.play(
                FadeOut(*[mob for mob in ms.submobjects if mob not in [ms.input, ms.dense_layer, ms.output_arrow, ms.output_layer, ms.softmax, ms.input_title, ms.output_layer_title]]),
                ReplacementTransform(ms.input, deepms.input),
                ReplacementTransform(ms.dense_layer, deepms.dense_layer),
                ReplacementTransform(ms.output_arrow, deepms.output_arrow),
                ReplacementTransform(ms.output_layer, deepms.output_layer),
                ReplacementTransform(ms.softmax, deepms.softmax),
                ms.input_title.animate.scale(LAYER_T_SCALE).next_to(deepms.input, UP, buff=0.15),
                ms.output_layer_title.animate.scale(LAYER_T_SCALE).next_to(deepms.output_layer, UP, buff=0.15), 
        )
        self.play(
            FadeIn(*[mob for mob in deepms.submobjects if mob not in [deepms.input, deepms.dense_layer, deepms.output_arrow, deepms.output_layer, deepms.softmax]])
        )
        
        # SLIDE 18:  ===========================================================
        # HIGHLIGHT LAYER
        self.next_slide(
            notes=
            '''Let's consider an architecture made of: a convolutional layer
            with 8 filters ;
            '''
        )
        sub_config = {'color':BLACK, 'font':SANS_SERIF_FONT, 'font_size':32}
        SUB_LT_SCALE=0.6
        conv1_t = LayerTitle('Convolutional','Layer').scale(LAYER_T_SCALE)
        conv1_t.add(Text('8 filters',**sub_config).scale(SUB_LT_SCALE*LAYER_T_SCALE).next_to(conv1_t, DOWN, buff=0.05))
        pool1_t = LayerTitle('Pooling','Layer',).scale(LAYER_T_SCALE)
        pool1_t.add(Text('Pool size (2,2)' ,**sub_config).scale(SUB_LT_SCALE*LAYER_T_SCALE).next_to(pool1_t, DOWN, buff=0.05))
        conv2_t = LayerTitle('Convolutional','Layer').scale(LAYER_T_SCALE)
        conv2_t.add(Text('16 filters', **sub_config).scale(SUB_LT_SCALE*LAYER_T_SCALE).next_to(conv2_t, DOWN, buff=0.05))
        pool2_t = pool1_t.copy()
        flatten_t = LayerTitle('Flatten','layer').scale(LAYER_T_SCALE)
        dense_t = LayerTitle('Dense','layer').scale(LAYER_T_SCALE)

        conv1_t.next_to(deepms.conv_layer_1, UP, buff=0.8)
        conv2_t.next_to(deepms.conv_layer_2, UP).match_y(conv1_t)
        flatten_t.next_to(deepms.flattened_vector, UP).align_to(ms.output_layer_title, UP)
        dense_t.next_to(Group(deepms.dense_layer, deepms.softmax), UP).align_to(flatten_t, UP)
        pool1_t.next_to(deepms.pooling_layer_1, DOWN, buff=1)
        pool2_t.next_to(deepms.pooling_layer_2, DOWN).match_y(pool1_t)

        def make_layer_highlight(mobject):
            r = RoundedRectangle(
                width=mobject.width + 2*0.1, height=7,
                color=GOLD, stroke_width=4, fill_opacity=0,
                corner_radius=0.25
            ).set_z_index(50)
            r.match_x(mobject).set_y(0)
            return r
            
        layer_highlights = [
            make_layer_highlight(Group(layer, t)) for layer, t in zip(
                [deepms.conv_layer_1, deepms.pooling_layer_1, deepms.conv_layer_2, deepms.pooling_layer_2,
                 deepms.flattened_vector, VGroup(deepms.dense_layer, deepms.softmax)],
                [conv1_t, pool1_t, conv2_t, pool2_t, flatten_t, dense_t]
            )
        ]
        self.play(FadeIn(conv1_t), Create(layer_highlights[0]))

        # SLIDE 19:  ===========================================================
        # HIGHLIGHT LAYER
        self.next_slide(
            notes=
            '''a pooling layer with pool size (2,2) ;
            '''
        )
        self.play(
            AnimationGroup(
                ReplacementTransform(layer_highlights[0], layer_highlights[1]),
                FadeIn(pool1_t),
                lag_ratio=0.5
            )
        )

        # SLIDE 20:  ===========================================================
        # HIGHLIGHT LAYER
        self.next_slide(
            notes=
            '''a convolutional layer with 16 filters ;
            '''
        )
        self.play(
            AnimationGroup(
                ReplacementTransform(layer_highlights[1], layer_highlights[2]),
                FadeIn(conv2_t),
                lag_ratio=0.5
            )
        )

        # SLIDE 21:  ===========================================================
        # HIGHLIGHT LAYER
        self.next_slide(
            notes=
            '''a pooling layer with pool size (2,2) ;
            '''
        )
        self.play(
            AnimationGroup(
                ReplacementTransform(layer_highlights[2], layer_highlights[3]),
                FadeIn(pool2_t),
                lag_ratio=0.5
            )
        )

        # SLIDE 22:  ===========================================================
        # HIGHLIGHT LAYER
        self.next_slide(
            notes=
            '''a flatten layer;
            '''
        )
        self.play(
            AnimationGroup(
                ReplacementTransform(layer_highlights[3], layer_highlights[4]),
                FadeIn(flatten_t),
                lag_ratio=0.5
            )
        )

        # SLIDE 23:  ===========================================================
        # HIGHLIGHT LAYER
        self.next_slide(
            notes=
            '''a dense layer of 10 neurons.
            '''
        )
        self.play(
            AnimationGroup(
                ReplacementTransform(layer_highlights[4], layer_highlights[5]),
                FadeIn(dense_t),
                lag_ratio=0.5
            )
        )

        # SLIDE 24:  ===========================================================
        # LAYER HIGHLIGHT DISAPPEARS
        self.next_slide(
            notes=
            '''How would the code change? Does this deeper architecture classify
            digits in a more accurate way? If so, is the training time
            comparable or longer?
            '''
        )
        self.play(FadeOut(layer_highlights[5]))
        self.wait(0.05)
