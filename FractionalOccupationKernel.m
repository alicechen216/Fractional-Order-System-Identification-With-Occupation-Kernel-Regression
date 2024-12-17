classdef FractionalOccupationKernel < handle
    properties
        q        
        mu       
        lambda   
        trajectories 
        times        
    end

    methods
        function obj = FractionalOccupationKernel(q, mu, lambda)
            obj.q = q;
            obj.mu = mu;
            obj.lambda = lambda;
            obj.trajectories = {};
            obj.times = {};
        end

        function addTrajectory(obj, traj, t)
            obj.trajectories{end+1} = traj;
            obj.times{end+1} = t;
        end

        function K = rbfKernel(obj, x, y)
            diff = x - y;
            K = exp(-norm(diff)^2 / obj.mu);
        end

        % Compute the Gram matrix with kernel evaluations
        function G = computeGramMatrix(obj)
            M = length(obj.trajectories);
            G = zeros(M, M);
            
            Cq = 1 / gamma(obj.q);
            
            for i = 1:M
                traj_i = obj.trajectories{i};
                t_i = obj.times{i};
                T_i = t_i(end);
                dt_i = t_i(2) - t_i(1);
                N_i = length(t_i);
                
                for j = i:M  % Exploit symmetry
                    traj_j = obj.trajectories{j};
                    t_j = obj.times{j};
                    T_j = t_j(end);
                    dt_j = t_j(2) - t_j(1);
                    N_j = length(t_j);
                    
                    sum_ij = 0;
                    for k = 1:N_i
                        delta_t_i = T_i - t_i(k);
                        if delta_t_i > 0
                            phi_i = delta_t_i^(obj.q - 1);
                        else
                            phi_i = 0;
                        end
                        
                        for l = 1:N_j
                            delta_t_j = T_j - t_j(l);
                            if delta_t_j > 0
                                phi_j = delta_t_j^(obj.q - 1);
                            else
                                phi_j = 0;
                            end
                            
                            K_val = obj.rbfKernel(traj_i(:,k), traj_j(:,l));
                            sum_ij = sum_ij + phi_i * phi_j * K_val * dt_i * dt_j;
                        end
                    end
                    G_ij = (Cq^2) * sum_ij;
                    G(i,j) = G_ij;
                    G(j,i) = G_ij;  % Symmetric Gram matrix
                end
            end
        end

        % Compute weights for each component
        function w = computeWeights(obj, component_idx)
            M = length(obj.trajectories);
            G = obj.computeGramMatrix();

            % Regularization for numerical stability
            G = G + obj.lambda * eye(M);

            % Compute observation vector c
            c = zeros(M, 1);
            for i = 1:M
                traj = obj.trajectories{i};
                c(i) = traj(component_idx, end) - traj(component_idx, 1);
            end

            % Solve linear system
            w = G \ c;
        end

        % Approximate function at a point
        function f = approximate(obj, x, w)
            M = length(obj.trajectories);
            Cq = 1 / gamma(obj.q);
            f = 0;

            for i = 1:M
                traj = obj.trajectories{i};
                t = obj.times{i};
                T_i = t(end);
                dt = t(2) - t(1);
                N = length(t);
                
                sum_i = 0;
                for k = 1:N
                    delta_t = T_i - t(k);
                    if delta_t > 0
                        phi_k = delta_t^(obj.q - 1);
                    else
                        phi_k = 0;
                    end
                    K_val = obj.rbfKernel(x, traj(:,k));
                    sum_i = sum_i + phi_k * K_val * dt;
                end
                phi = Cq * sum_i;
                f = f + w(i) * phi;
            end
        end
    end
end
