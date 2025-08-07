% =========================================================================
% CGUFS-IMC (Incomplete Multi-view Clustering based on Confidence Graph
% and Unsupervised Feature Selection) - Revised Version
% =========================================================================

% 1. Initialization of Data and Parameters
% -------------------------------------------------------------------------
clear; clc;

% Placeholder for input data and masks
% Replace these with your actual data and masks before running.
% Y_incomplete_normalized should be a cell array of size 1 x num_views,
% where each cell contains a num_samples x num_features_v matrix with NaNs for missing data.
% Mask should be a cell array of the same size, with 1s for observed
% entries and 0s for missing entries.
% num_clusters should be the number of desired clusters.
%
% Example placeholder definitions:
num_views = 3;
num_samples = 500;
num_features = [300, 250, 400];
num_clusters = 6;
Y_incomplete_normalized = cell(num_views, 1);
Mask = cell(num_views, 1);
% [Fill Y_incomplete_normalized and Mask here]

% 2. CGUFS-IMC Algorithm Parameter Settings (Default values)
% -------------------------------------------------------------------------
params.alpha = 10;     % Weight for the self-representation term
params.beta = 1;       % Weight for the L2,1-norm regularization on C^v
params.tau = 10;       % Weight for the graph regularization term
params.gamma = 0.001;  % Weight for the data imputation constraint
params.epsilon = 0.01;
params.k_neighbors = 8; % Number of kNN neighbors for graph construction
params.sigma_s_sq = 10;
params.rho1 = 1e-3;
params.rho2 = 1e-3;
params.rho3 = 1e-3;
params.max_iter = 20;  % Maximum number of ADMM iterations
params.tol = 1e-3;

% 3. CGUFS-IMC Algorithm Initialization
% -------------------------------------------------------------------------
A_views = cell(num_views, 1);
S_views = cell(num_views, 1);
C_views = cell(num_views, 1);
E_views = cell(num_views, 1);
D_mat = orth(rand(num_clusters, num_clusters));
K_mat = rand(num_samples, num_clusters);
K_mat = K_mat ./ sum(K_mat, 2);
B_mat = zeros(num_samples, max(num_features));
H_tensor = rand(num_samples, num_samples, num_views);
H_star = eye(num_samples);
lambda_weights = ones(num_views, 1) / num_views;
E1_tensor = zeros(num_samples, num_samples, num_views);
Y_bar_views = cell(num_views, 1);
Z_tensor = rand(num_samples, num_samples, num_views);
J1_tensor = zeros(num_samples, num_samples, num_views);
J2_views = cell(num_views, 1);
J3_views = cell(num_views, 1);
G_views = cell(num_views, 1);
W_views = cell(num_views, 1);

Y_completed_views = Y_incomplete_normalized;
for v = 1:num_views
    nan_indices = isnan(Y_completed_views{v});
    Y_completed_views{v}(nan_indices) = 0;
end

for v = 1:num_views
    current_Y_v = Y_incomplete_normalized{v};
    current_Mask_v = Mask{v};
    [n_v_samples, d_v_features] = size(current_Y_v);
    
    S_v_initial = zeros(n_v_samples, n_v_samples);
    available_rows_idx = find(any(current_Mask_v, 2));
    if ~isempty(available_rows_idx)
        Xv_available = current_Y_v(available_rows_idx, :);
        dist_available = pdist2(Xv_available, Xv_available);
        sigma_dynamic = mean(dist_available(:));
        
        for i_avail = 1:length(available_rows_idx)
            for j_avail = 1:length(available_rows_idx)
                idx_i = available_rows_idx(i_avail);
                idx_j = available_rows_idx(j_avail);
                common_observed_features_idx = find(current_Mask_v(idx_i, :) & current_Mask_v(idx_j, :));
                if ~isempty(common_observed_features_idx)
                    dist = norm(current_Y_v(idx_i, common_observed_features_idx) - current_Y_v(idx_j, common_observed_features_idx))^2;
                    S_v_initial(idx_i, idx_j) = exp(-dist / (2 * sigma_dynamic^2));
                else
                    S_v_initial(idx_i, idx_j) = 0;
                end
            end
        end
    end
    S_views{v} = S_v_initial;
    A_views{v} = S_v_initial;
    
    C_views{v} = rand(d_v_features, num_clusters);
    
    fully_missing_rows_idx = find(all(Mask{v} == 0, 2));
    if ~isempty(fully_missing_rows_idx)
        E_views{v} = zeros(d_v_features, length(fully_missing_rows_idx));
    else
        E_views{v} = [];
    end
    
    Y_bar_views{v} = zeros(n_samples, d_v_features);
    J2_views{v} = zeros(n_samples, d_v_features);
    J3_views{v} = zeros(n_samples, d_v_features);
    
    G_v = zeros(n_samples, n_samples);
    W_v = zeros(n_samples, n_samples);
    for i = 1:n_samples
        if any(current_Mask_v(i, :))
            G_v(i, i) = 1;
        else
            W_v(i, i) = 1;
        end
    end
    G_views{v} = G_v;
    W_views{v} = W_v;
end

% 4. Main ADMM Iteration Loop
% -------------------------------------------------------------------------
fprintf('Starting ADMM iterations...\n');
for iter = 1:params.max_iter
    
    % --- 1. Update Y_completed_v (completed data matrix) ---
    for v = 1:num_views
        current_Y_v_orig = Y_incomplete_normalized{v};
        current_Mask_v = Mask{v};
        current_Y_bar_v = Y_bar_views{v};
        current_J2_v = J2_views{v};
        current_A_v = A_views{v};
        current_B_v = B_mat(:, 1:num_features(v));
        current_J3_v = J3_views{v};
        
        fully_missing_rows_idx = find(all(current_Mask_v == 0, 2));
        observed_rows_idx = find(any(current_Mask_v, 2));
        
        RHS_admm = params.rho2 * (current_Y_bar_v - current_J2_v/params.rho2) + ...
                   params.rho3 * (current_A_v * Y_completed_views{v} + current_B_v - current_J3_v / params.rho3);
        
        Y_hat_new = zeros(num_samples, num_features(v));
        Y_hat_new(observed_rows_idx, :) = current_Y_v_orig(observed_rows_idx, :);
        
        if ~isempty(fully_missing_rows_idx)
            denom_missing = params.rho2 + params.rho3 + 2 * params.gamma;
            numerator_missing = RHS_admm(fully_missing_rows_idx, :);
            Y_hat_new(fully_missing_rows_idx, :) = numerator_missing / denom_missing;
        end
        Y_completed_views{v} = Y_hat_new;
        
        if ~isempty(fully_missing_rows_idx)
            E_views{v} = Y_completed_views{v}(fully_missing_rows_idx, :)';
        else
            E_views{v} = [];
        end
    end
    
    % --- 2. Update C^v (projection matrix) ---
    for v = 1:num_views
        current_Y_bar_v = Y_bar_views{v};
        current_Y_hat_v = Y_completed_views{v};
        
        degree_matrix = diag(sum(H_star, 2));
        L_H_star = degree_matrix - H_star;
        
        C_views{v} = update_C_v(C_views{v}, current_Y_bar_v, current_Y_hat_v, D_mat, K_mat, L_H_star, params.beta, params.tau, num_features(v), num_clusters);
    end
    
    % --- 3. Update S^v (similarity matrix for available samples) ---
    for v = 1:num_views
        current_Y_v = Y_incomplete_normalized{v};
        current_mask_v = Mask{v};
        F_v = calculate_confidence_graph(current_Y_v, current_mask_v, params.k_neighbors);
        S_views{v} = update_S_v(S_views{v}, A_views{v}, F_v, current_Y_v, current_mask_v, params.alpha, params.sigma_s_sq);
    end
    
    % --- 4. Update A^v (complete similarity matrix) ---
    for v = 1:num_views
        current_Y_completed_v = Y_completed_views{v};
        current_S_v = S_views{v};
        current_B_v = B_mat(:, 1:num_features(v));
        current_J3_v = J3_views{v};
        
        Term1 = params.rho3 * (current_Y_completed_v * current_Y_completed_v');
        Term2 = params.alpha * eye(num_samples);
        LHS = Term1 + Term2;
        
        RHS = params.rho3 * (current_Y_completed_v * (current_Y_completed_v - (A_views{v} * current_Y_completed_v) - current_B_v - current_J3_v/params.rho3)') + params.alpha * current_S_v';
        
        A_views{v} = (LHS \ RHS')';
        
        A_views{v}(A_views{v} < 0) = 0;
        row_sums = sum(A_views{v}, 2);
        zero_rows_idx = (row_sums == 0);
        A_views{v}(~zero_rows_idx, :) = A_views{v}(~zero_rows_idx, :) ./ row_sums(~zero_rows_idx);
        A_views{v}(zero_rows_idx, :) = 0;
    end
    
    % --- 5. Update D (orthogonal basis) and K (pseudo cluster labels) ---
    W_K = zeros(num_samples, num_clusters);
    for v = 1:num_views
        W_K = W_K + (Y_bar_views{v} * C_views{v});
    end
    Temp_K = W_K * D_mat';
    Temp_K(Temp_K < 0) = 0;
    row_sums = sum(Temp_K, 2);
    zero_rows_idx = (row_sums == 0);
    K_mat(~zero_rows_idx, :) = Temp_K(~zero_rows_idx, :) ./ row_sums(~zero_rows_idx);
    K_mat(zero_rows_idx, :) = 0;
    [U_d, ~, V_d] = svd(W_K' * K_mat);
    D_mat = U_d * V_d';
    
    % --- 6. Update B (self-representation error) and E1 (noise tensor) ---
    Residual_B_sum = zeros(num_samples, max(num_features));
    for v = 1:num_views
        current_res_v = Y_completed_views{v} - (A_views{v} * Y_completed_views{v});
        Residual_B_sum(:, 1:num_features(v)) = Residual_B_sum(:, 1:num_features(v)) + current_res_v + J3_views{v}/params.rho3;
    end
    B_mat = Solve_L21_norm(Residual_B_sum, 1/params.rho3);
    
    Residual_E1_tensor = H_tensor - Z_tensor + J1_tensor/params.rho1;
    for v = 1:num_views
        E1_tensor(:, :, v) = Solve_L21_norm(Residual_E1_tensor(:, :, v), 1/params.rho1);
    end
    
    % --- 7. Update H (intrinsic tensor) and H_star (consensus similarity matrix) ---
    H_tensor = update_H_tensor_WTNN(Z_tensor - J1_tensor/params.rho1, params.rho1, lambda_weights);
    
    H_star_temp = zeros(num_samples, num_samples);
    for v = 1:num_views
        H_star_temp = H_star_temp + lambda_weights(v) * H_tensor(:, :, v);
    end
    H_star = (H_star_temp + H_star_temp') / 2;
    H_star(H_star < 0) = 0;
    row_sums_H_star = sum(H_star, 2);
    zero_rows_idx = (row_sums_H_star == 0);
    H_star(~zero_rows_idx, :) = H_star(~zero_rows_idx, :) ./ row_sums_H_star(~zero_rows_idx);
    H_star(zero_rows_idx, :) = 0;
    
    % --- 8. Update lambda^v (view weights) ---
    dist_H_star_H_v = zeros(num_views, 1);
    for v = 1:num_views
        dist_H_star_H_v(v) = norm(H_star - H_tensor(:, :, v), 'fro')^2;
    end
    inv_dist = 1 ./ (dist_H_star_H_v + eps);
    lambda_weights = inv_dist / sum(inv_dist);
    
    % --- 9. Update auxiliary variables Y_bar and Z ---
    for v = 1:num_views
        term_OCLSP = K_mat * D_mat * C_views{v}';
        
        term1_rho2 = params.rho2 * (Y_completed_views{v} - J2_views{v}/params.rho2);
        term2_rho3 = params.rho3 * (Y_completed_views{v} - (A_views{v} * Y_completed_views{v}) - B_mat(:, 1:num_features(v)) - J3_views{v}/params.rho3);
        Y_bar_views{v} = (term1_rho2 + term2_rho3 + term_OCLSP) / (params.rho2 + params.rho3 + 1);
    end
    
    Z_tensor = H_tensor - E1_tensor + J1_tensor / params.rho1;
    
    % --- 10. Update Lagrange Multipliers J1, J2, J3 ---
    for v = 1:num_views
        J1_tensor(:, :, v) = J1_tensor(:, :, v) + params.rho1 * (H_tensor(:, :, v) - Z_tensor(:, :, v));
        J2_views{v} = J2_views{v} + params.rho2 * (Y_completed_views{v} - Y_bar_views{v});
        J3_views{v} = J3_views{v} + params.rho3 * (Y_completed_views{v} - (A_views{v} * Y_completed_views{v}) - B_mat(:, 1:num_features(v)));
    end
    
    if mod(iter, 10) == 0 || iter == 1 || iter == params.max_iter
        fprintf('Iteration %d/%d\n', iter, params.max_iter);
    end
end

% -------------------------------------------------------------------------
% Helper Function Definitions
% -------------------------------------------------------------------------

function F_v = calculate_confidence_graph(X_v_data, mask_v, k_neighbors)
    [n_samples, ~] = size(X_v_data);
    available_samples_idx = find(any(mask_v, 2));
    if isempty(available_samples_idx)
        F_v = zeros(n_samples, n_samples);
        return;
    end
    dist_matrix = zeros(length(available_samples_idx), length(available_samples_idx));
    for i_avail = 1:length(available_samples_idx)
        for j_avail = 1:length(available_samples_idx)
            idx_i = available_samples_idx(i_avail);
            idx_j = available_samples_idx(j_avail);
            common_observed_features_idx = find(mask_v(idx_i, :) & mask_v(idx_j, :));
            if ~isempty(common_observed_features_idx)
                vec_i = X_v_data(idx_i, common_observed_features_idx);
                vec_j = X_v_data(idx_j, common_observed_features_idx);
                dist_matrix(i_avail, j_avail) = norm(vec_i - vec_j)^2;
            else
                dist_matrix(i_avail, j_avail) = inf;
            end
        end
    end
    Z_temp = zeros(length(available_samples_idx), length(available_samples_idx));
    for i = 1:length(available_samples_idx)
        [~, sorted_idx] = sort(dist_matrix(i, :));
        knn_idx = sorted_idx(2:min(k_neighbors+1, end));
        Z_temp(i, knn_idx) = 1;
        Z_temp(knn_idx, i) = 1;
        Z_temp(i, i) = 1;
    end
    F_v_tilde = Z_temp * Z_temp';
    F_v = zeros(n_samples, n_samples);
    for i = 1:length(available_samples_idx)
        for j = 1:length(available_samples_idx)
            F_v(available_samples_idx(i), available_samples_idx(j)) = F_v_tilde(i,j);
        end
    end
    max_F_val = max(F_v(:));
    if max_F_val > 0
        F_v = F_v / max_F_val;
    end
end

function C_updated = update_C_v(C_current, Y_bar_v, Y_hat_v, D_mat, K_mat, L_H_star, beta_param, tau_param, num_features_v, num_clusters)
    max_irls_iter = 5;
    tol_irls = 1e-4;
    C_updated = C_current;
    for iter_irls = 1:max_irls_iter
        row_norms = sqrt(sum(C_updated.^2, 2));
        W_l21_diag = 1 ./ (2 * row_norms + eps);
        W_l21 = diag(W_l21_diag);
        LHS_C = Y_bar_v' * Y_bar_v + tau_param * Y_hat_v' * L_H_star * Y_hat_v + beta_param * W_l21;
        RHS_C = Y_bar_v' * K_mat * D_mat;
        C_new = LHS_C \ RHS_C;
        if norm(C_new - C_updated, 'fro') / norm(C_updated, 'fro') < tol_irls
            C_updated = C_new;
            break;
        end
        C_updated = C_new;
    end
end

function S_updated = update_S_v(S_current, A_v, F_v, Y_v_data, mask_v, alpha_param, sigma_s_sq)
    n_samples = size(S_current, 1);
    dist_matrix = zeros(n_samples, n_samples);
    available_samples_idx = find(any(mask_v, 2));
    if isempty(available_samples_idx)
        S_updated = S_current;
        return;
    end
    for i_avail = 1:length(available_samples_idx)
        for j_avail = 1:length(available_samples_idx)
            idx_i = available_samples_idx(i_avail);
            idx_j = available_samples_idx(j_avail);
            common_observed_features_idx = find(mask_v(idx_i, :) & mask_v(idx_j, :));
            if ~isempty(common_observed_features_idx)
                vec_i = Y_v_data(idx_i, common_observed_features_idx);
                vec_j = Y_v_data(idx_j, common_observed_features_idx);
                dist_matrix(i_avail, j_avail) = norm(vec_i - vec_j)^2;
            else
                dist_matrix(i_avail, j_avail) = inf;
            end
        end
    end
    S_tilde = zeros(n_samples, n_samples);
    for i_avail = 1:length(available_samples_idx)
        idx_i = available_samples_idx(i_avail);
        for j_avail = 1:length(available_samples_idx)
            idx_j = available_samples_idx(j_avail);
            if dist_matrix(i_avail,j_avail) ~= inf
                w_ij_v = exp(-dist_matrix(i_avail,j_avail) / (2 * sigma_s_sq));
                S_tilde(idx_i, idx_j) = (w_ij_v + 2 * alpha_param * A_v(idx_i, idx_j)) * F_v(idx_i, idx_j);
            else
                S_tilde(idx_i, idx_j) = 0;
            end
        end
    end
    S_updated = zeros(n_samples, n_samples);
    for i = 1:n_samples
        row_sum = sum(S_tilde(i, :));
        if row_sum > 0
            S_updated(i, :) = S_tilde(i, :) / row_sum;
        end
    end
    S_updated(isnan(S_updated)) = 0;
    S_updated = S_updated - diag(diag(S_updated));
end

function X_l21 = Solve_L21_norm(X, lambda_param)
    X_l21 = zeros(size(X));
    for j = 1:size(X, 2)
        col_norm = norm(X(:, j), 2);
        if col_norm > lambda_param
            X_l21(:, j) = (col_norm - lambda_param) * X(:, j) / col_norm;
        else
            X_l21(:, j) = 0;
        end
    end
end

function H_updated = update_H_tensor_WTNN(X_tensor, rho_param, omega_weights)
    [n1, n2, n3] = size(X_tensor);
    H_updated = zeros(n1, n2, n3);
    X_f = fft(X_tensor, [], 3);
    for k = 1:n3
        X_slice = X_f(:,:,k);
        weight_threshold = omega_weights(min(k, length(omega_weights))) / rho_param;
        [U, S, V] = svd(X_slice, 'econ');
        S_diag = diag(S);
        S_shrink = max(0, S_diag - weight_threshold);
        H_updated(:,:,k) = U * diag(S_shrink) * V';
    end
    H_updated = ifft(H_updated, [], 3);
    H_updated = real(H_updated);
end