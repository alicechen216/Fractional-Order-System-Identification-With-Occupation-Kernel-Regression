function system_1_experiment_2()
    % Parameters
    q = 4/5;       % Fractional order
    mu_values = [0.5, 1.0, 2.0, 3.0, 5.0];  % Kernel widths to test
    lambda_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4];  % Regularization parameters to test

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
    trajectories = cell(size(initial_conditions, 2), 1);
    times = cell(size(initial_conditions, 2), 1);
    for i = 1:size(initial_conditions, 2)
        x0 = initial_conditions(:,i);
        X = fractionalVolterraSolver(@systemDynamics, x0, q, t);
        trajectories{i} = X;
        times{i} = t;
        fprintf('Trajectory %d: final position [%f, %f]\n', i, X(1, end), X(2, end));
    end

    % Prepare test points for error evaluation
    test_points = [
        0.2, 0.2;
        0.4, 0.4;
        0.6, 0.6;
        0.8, 0.8;
        0.5, 0.5
    ]';

    % Part 1: Vary mu with fixed lambda
    fixed_lambda = 1e-6;
    fprintf('\nExperiment Part 1: Varying mu with fixed lambda = %e\n', fixed_lambda);
    errors_mu = zeros(length(mu_values), 1);
    for idx = 1:length(mu_values)
        mu = mu_values(idx);
        kernel = createKernel(q, mu, fixed_lambda, trajectories, times);

        % Compute weights
        w1 = kernel.computeWeights(1);
        w2 = kernel.computeWeights(2);

        % Evaluate errors at test points
        errors = evaluateErrors(kernel, w1, w2, test_points);
        avg_error = mean(errors);
        errors_mu(idx) = avg_error;

        fprintf('mu = %.2f, Average Error = %e\n', mu, avg_error);
    end

    % Display Table I
    fprintf('\nTable I: Errors for different mu values with lambda = %e\n', fixed_lambda);
    table_mu = table(mu_values', errors_mu, 'VariableNames', {'mu', 'Average_Error'});
    disp(table_mu);

    % Part 2: Vary lambda with fixed mu
    fixed_mu = 2.0;
    fprintf('\nExperiment Part 2: Varying lambda with fixed mu = %.2f\n', fixed_mu);
    errors_lambda = zeros(length(lambda_values), 1);
    for idx = 1:length(lambda_values)
        lambda = lambda_values(idx);
        kernel = createKernel(q, fixed_mu, lambda, trajectories, times);

        % Compute weights
        w1 = kernel.computeWeights(1);
        w2 = kernel.computeWeights(2);

        % Evaluate errors at test points
        errors = evaluateErrors(kernel, w1, w2, test_points);
        avg_error = mean(errors);
        errors_lambda(idx) = avg_error;

        fprintf('lambda = %e, Average Error = %e\n', lambda, avg_error);
    end

    % Display Table II
    fprintf('\nTable II: Errors for different lambda values with mu = %.2f\n', fixed_mu);
    table_lambda = table(lambda_values', errors_lambda, 'VariableNames', {'lambda', 'Average_Error'});
    disp(table_lambda);
end

function kernel = createKernel(q, mu, lambda, trajectories, times)
    % Initialize the kernel object with given parameters
    kernel = FractionalOccupationKernel(q, mu, lambda);
    % Add trajectories to the kernel object
    for i = 1:length(trajectories)
        kernel.addTrajectory(trajectories{i}, times{i});
    end
end

function errors = evaluateErrors(kernel, w1, w2, test_points)
    % Compute the approximation errors at given test points
    num_points = size(test_points, 2);
    errors = zeros(num_points, 1);
    for i = 1:num_points
        x_test = test_points(:, i);
        f_true = systemDynamics(x_test);
        f_approx = [
            kernel.approximate(x_test, w1);
            kernel.approximate(x_test, w2)
        ];
        errors(i) = norm(f_true - f_approx);
    end
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

% Include the full FractionalOccupationKernel class as defined previously
