function main()
clc;clear;close all;
    % Parameters
    q = 4/5;       % Fractional order
    mu = 2;        % Kernel width parameter
    lambda = 1e-6; % Regularization parameter

    % Create kernel object
    kernel = FractionalOccupationKernel(q, mu, lambda);

    % Initial conditions as specified
    initial_conditions = [
        0.0, 0.0;
        0.0, 0.3;
        0.0, 0.6;
        0.0, 0.9;
        0.3, 0.3;
        0.3, 0.6;
        0.3, 0.9;
        0.6, 0.6;
        0.6, 0.9;
        0.9, 0.9
    ]';

    % Time settings
    T = 1;
    N = 100;          % Number of time steps
    dt = T / N;
    t = linspace(0, T, N+1);

    % Generate trajectories using the fractional Volterra integral equation
    fprintf('Generating trajectories using fractional Volterra integral equation...\n');
    for i = 1:size(initial_conditions, 2)
        x0 = initial_conditions(:,i);
        X = fractionalVolterraSolver(@systemDynamics, x0, q, t);
        % Do not transpose X here
        % Store trajectory
        kernel.addTrajectory(X, t);
        fprintf('Trajectory %d: final position [%f, %f]\n', i, X(1, end), X(2, end));
    end

    % Proceed as before with computing Gram matrix, weights, approximations, and plotting
    fprintf('\nNumber of stored trajectories: %d\n', length(kernel.trajectories));

    % Compute Gram matrix
    fprintf('\nComputing Gram matrix...\n');
    G = kernel.computeGramMatrix();

    % Compute weights
    fprintf('\nComputing weights...\n');
    w1 = kernel.computeWeights(1);
    w2 = kernel.computeWeights(2);

    % Test points evaluation
    test_points = [0.5, 0.5; 0.2, 0.2; 0.8, 0.8]';
    fprintf('\nTest points evaluation:\n');
    for i = 1:size(test_points, 2)
        x_test = test_points(:,i);
        f_true = systemDynamics(x_test);
        f_approx = [kernel.approximate(x_test, w1); kernel.approximate(x_test, w2)];

        fprintf('\nTest point [%f, %f]:\n', x_test(1), x_test(2));
        fprintf('True value: [%f, %f]\n', f_true(1), f_true(2));
        fprintf('Approximation: [%f, %f]\n', f_approx(1), f_approx(2));
        fprintf('Absolute error: %e\n', norm(f_true - f_approx));
    end

    % Visualizations
    figure(1);
    plotTrajectoryErrors(kernel.trajectories{5}, t, kernel, w1, w2);
    sgtitle('Errors along trajectory \gamma_5');

    figure(2);
    plotTrajectoryErrors(kernel.trajectories{9}, t, kernel, w1, w2);
    sgtitle('Errors along trajectory \gamma_9');

    figure(3);
    plotDomainErrors(kernel, w1, w2);
end

function X = fractionalVolterraSolver(f, x0, q, t)
    % Solves x(t) = x0 + (1/Gamma(q)) * int_0^t (t - tau)^(q - 1) * f(x(tau)) dtau
    % using numerical quadrature and iterative evaluation

    N = length(t) - 1;
    dt = t(2) - t(1);
    n_states = length(x0);
    X = zeros(n_states, N+1);
    X(:, 1) = x0;

    gamma_q = gamma(q);
    for n = 1:N
        % Compute the integral using the trapezoidal rule
        integral = zeros(n_states, 1);
        for k = 1:n
            tk = t(k);
            delta_t = t(n+1) - tk;
            if delta_t > 0
                weight = delta_t^(q - 1);
            else
                weight = 0;
            end
            fxk = f(X(:, k));
            if k == 1 || k == n
                integral = integral + 0.5 * weight * fxk;
            else
                integral = integral + weight * fxk;
            end
        end
        integral = (dt / gamma_q) * integral;

        % Update X(:, n+1)
        X(:, n+1) = x0 + integral;
    end
end

function dx = systemDynamics(x)
    dx = [1 / (1 + x(2)^2); 1 / (1 + x(1)^2)];
end

function plotTrajectoryErrors(traj, t, kernel, w1, w2)
    errors1 = zeros(size(t));
    errors2 = zeros(size(t));

    for i = 1:length(t)
        x = traj(:,i);
        f_true = systemDynamics(x);
        f_approx = [kernel.approximate(x, w1); kernel.approximate(x, w2)];
        errors1(i) = abs(f_true(1) - f_approx(1));
        errors2(i) = abs(f_true(2) - f_approx(2));
    end

    subplot(2,1,1);
    plot(t, errors1, 'LineWidth', 2);
    ylabel('Error in f_1');
    xlabel('t');
    grid on;

    subplot(2,1,2);
    plot(t, errors2, 'LineWidth', 2);
    ylabel('Error in f_2');
    xlabel('t');
    grid on;
end

function plotDomainErrors(kernel, w1, w2)
    [X_grid, Y_grid] = meshgrid(linspace(0, 1, 50));
    E1 = zeros(size(X_grid));
    E2 = zeros(size(X_grid));

    for i = 1:numel(X_grid)
        x = [X_grid(i); Y_grid(i)];
        f_true = systemDynamics(x);
        f_approx = [kernel.approximate(x, w1); kernel.approximate(x, w2)];
        E1(i) = abs(f_true(1) - f_approx(1));
        E2(i) = abs(f_true(2) - f_approx(2));
    end

    subplot(1,2,1);
    surf(X_grid, Y_grid, E1, 'EdgeColor', 'none');
    title('Error in f_1');
    xlabel('x_1');
    ylabel('x_2');
    colorbar;
    view(2);

    subplot(1,2,2);
    surf(X_grid, Y_grid, E2, 'EdgeColor', 'none');
    title('Error in f_2');
    xlabel('x_1');
    ylabel('x_2');
    colorbar;
    view(2);
end