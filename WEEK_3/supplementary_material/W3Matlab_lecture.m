% Input parameters 
xp = [0.75 , -1];     % target point 
L1 = 1;               % length of arm1 
L2 = 1.5;             % length of arm2 
theta = [2.5 2.7];    % initial angles 
tol = 0.01;           % tolerance 
alpha = 0.1;          % learning rate 
Niter = 1000;         % max iteration number

% Compute tip robot position
function [x,y] = robot_position(theta, L1, L2)
    x = L1 * cos(theta(1)) + L2 * cos(theta(2));
    y = L1 * sin(theta(1)) + L2 * sin(theta(2));
end

% Plot robot arms
[x,y] = robot_position(theta, L1, L2);
fprintf('%f4.2 %f4.2 \n',[x,y]);
plot_robot_arm(theta, xp, [L1, L2]);

% Evaluation of J
function Jval = J(theta, xp, L1, L2) 
    [x, y] = robot_position(theta, L1, L2);
    Jval = (x-xp(1))^2 + (y-xp(2))^2;
end

Jval = J(theta, xp, L1, L2);
fprintf('%f4.2 \n',Jval);

% Evaluation of grad(J)
function grad = grad_J(theta, xp, L1, L2)
    [x, y] = robot_position(theta,L1,L2);

    dx_dt1 = - L1 * sin(theta(1));
    dx_dt2 = - L2 * sin(theta(2));
    dy_dt1 =   L1 * cos(theta(1));
    dy_dt2 =   L2 * cos(theta(2));

    dJ_dx = 2*(x - xp(1));
    dJ_dy = 2*(y - xp(2));
    
    DJ_dt1 = dJ_dx*dx_dt1 + dJ_dy*dy_dt1;
    DJ_dt2 = dJ_dx*dx_dt2 + dJ_dy*dy_dt2; 
    grad = [DJ_dt1, DJ_dt2];
end

% Gradient Descent Method 
i = 1;
while i <= Niter
    % Gradient of the objective function
    grad = grad_J(theta, xp, L1, L2);
                
    % Update theta with gradient 
    theta = theta - alpha * grad;

    % Compute the current distance
    Jval = J(theta, xp, L1, L2);

    % Check for convergence
    if Jval < tol
        fprintf('Converged after %d iterations.\n', i)
        break
    end
    i = i + 1;
end

fprintf('Final angle combination: %f4.2 %f4.2 \n', theta) 
fprintf('Final distance: %f4.2 \n', Jval) 
plot_robot_arm(theta, xp, [L1, L2]);