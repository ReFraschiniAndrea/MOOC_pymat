function plot_robot_arm(theta, target, arm_lengths)
    l1 = arm_lengths(1);
    l2 = arm_lengths(2);

    x0 = 0;
    y0 = 0;
    x1 = l1 * cos(theta(1));
    y1 = l1 * sin(theta(1));
    x2 = x1 + l2 * cos(theta(2));
    y2 = y1 + l2 * sin(theta(2));

    % Plot the arm segments
    figure;
    hold on;
    plot([x0, x1], [y0, y1], 'b-', 'DisplayName', 'Segment 1', 'LineWidth',2);
    plot([x1, x2], [y1, y2], 'b-', 'DisplayName', 'Segment 2', 'LineWidth',2);

    % Plot the target point
    plot(target(1), target(2), 'Marker', 'pentagram', 'DisplayName', 'Target', 'MarkerSize', 20,'Color', 'magenta', 'MarkerFaceColor', 'magenta');
    plot(x0, y0, 'Marker', 'square', 'MarkerSize', 12, 'Color', 'black', 'MarkerFaceColor', 'black');
    plot(x1, y1, 'Marker', 'square', 'MarkerSize', 12, 'Color', 'black', 'MarkerFaceColor', 'black');
    plot(x2, y2, 'Marker', 'o', 'MarkerSize', 12, 'Color', 'green', 'MarkerFaceColor', 'green');

    % Setting up the plot
    plot_max = max([target(1), l1 + l2, target(2)]);
    xlim([- plot_max - 0.5, plot_max + 0.5]);
    ylim([- plot_max - 0.5, plot_max + 0.5]);
    axis equal;
    grid on;
    title('Robotic Arm Configuration');
    xlabel('X');
    ylabel('Y');

    % Show the plot
    hold off;
end