function compare_images(matrix1, matrix2)
    figure;
    
    % --- Layout parameters ---
    ratio = size(matrix2, 2) / size(matrix1, 2);
    margin = 0.02;         % outer margin (left/right)
    spacing = 0.03;        % space between the two images
    total_width = 1 - 2*margin - spacing;
    % width of the two axes
    w1 = total_width * (1/(1+ratio));
    w2 = total_width * (ratio/(1+ratio));
    % Vertical centering
    h = 0.9;               % height of the axes (90% of figure)
    y = (1 - h) / 2;       % centered vertically

    % Axes 1
    ax1 = axes('Position', [margin, y, w1, h]);
    imshow(matrix1, [0 255]);
    title(ax1, "image 1");
    axis(ax1, 'off');

    % Axes 2
    ax2 = axes('Position', [margin + w1 + spacing, y, w2, h]);
    imshow(matrix2, [0 255]);
    title(ax2, "Image 2");
    axis(ax2, 'off');

end