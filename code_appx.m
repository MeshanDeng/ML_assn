
clear; clc;
load('data_appendix.mat');
save_folder = fileparts(mfilename('fullpath'));

%% Step 1. Initialization
T_total = size(data,1);
N       = size(data,2);
H_max   = 8;
T0      = 30;

target_list = {'GDP','CONS','EMP'};
model_list  = {'VAR','LBVAR','SBVAR','DFM', 'FAVAR'};

rng(123);

fprintf('\n=== Step 1: Initialization ===\n');
fprintf('Total observations (T): %d | Variables (N): %d\n', T_total, N);
fprintf('Initial training window: T0 = %d | Forecast horizon H_max = %d\n', T0, H_max);
fprintf('Targets: %s\n', strjoin(target_list, ', '));


%% Step 2a. VAR lag selection via BIC (small system on targets)
fprintf('\n=== Step 2a: VAR lag selection (BIC on targets) ===\n');
[p_opt, ~] = select_VAR_lag(data, target_list, var_names, 6);
fprintf('→ Selected optimal VAR lag length (p_opt): %d\n', p_opt);


%% Step 2b. LBVAR hyperparameter calibration via EB
fprintf('\n=== Step 2b: LBVAR (full-system BVAR) calibration (EB over lambda, delta) ===\n');
[lambda_LBVAR, delta_LBVAR] = calibrate_BVAR_EB(data, p_opt);
fprintf('→ LBVAR hyperparameters: lambda=%.3f, delta=%.3f\n', lambda_LBVAR, delta_LBVAR);


%% Step 2c. SBVAR variable selection + hyperparameter calibration via EB
fprintf('\n=== Step 2c: SBVAR variable selection (K=6:10) + EB calibration ===\n');

[best_sbvar_list, lambda_SBVAR, delta_SBVAR, best_rmse_val] = ...
    select_and_calibrate_SBVAR(data, var_names, target_list, p_opt, T0, H_max);

fprintf('\n=== Final SBVAR variable set (%d vars total: %d targets + %d selected) ===\n', ...
        numel(best_sbvar_list), numel(target_list), numel(best_sbvar_list)-numel(target_list));
disp(best_sbvar_list');

fprintf('→ SBVAR EB hyperparameters: lambda=%.3f, delta=%.3f (validation RMSE=%.4f)\n', ...
    lambda_SBVAR, delta_SBVAR, best_rmse_val);


%% Step 2d. DFM setup (Bai & Ng for r, q = p_opt)
fprintf('\n=== Step 2d: Dynamic Factor Model (DFM) setup ===\n');
[r_opt, q_opt] = setup_DFM_BaiNg(data, p_opt);


%% Step 3. Initialize forecast containers
fprintf('\n=== Step 4: Set up containers ===\n');
results = init_forecast_containers(model_list, target_list, T_total, T0, H_max);


%% Step 4. Recursive forecasting (expanding window)
fprintf('\n=== Step 5: Recursive forecasting (expanding window) ===\n');
results = recursive_forecasting_hybrid(data, var_names, target_list, best_sbvar_list, ...
    results, p_opt, ...
    lambda_LBVAR, delta_LBVAR, ...
    lambda_SBVAR, delta_SBVAR, ...
    r_opt, q_opt, H_max, T0);

save_folder = fileparts(mfilename('fullpath'));


%% Step 5. Evaluation
fprintf('\n=== Step 6: Evaluation and Output ===\n');
evaluate_forecasts(results, save_folder);


%% Step 6. Plot RMSE curves
plot_RMSE_curves(results, save_folder);

fprintf('\n=== All steps completed successfully ===\n');


%======================================================================
%  Local functions
%======================================================================

%% ---------- VAR lag selection (on targets) ----------
function [p_opt, bic_vals] = select_VAR_lag(data, target_list, var_names, p_max)

    [~, idx_VAR] = ismember(target_list, var_names);
    Y_VAR = data(:, idx_VAR);
    K_VAR = size(Y_VAR,2);
    bic_vals = nan(p_max,1);

    for p = 1:p_max
        Y_lag_p = lagmatrix(Y_VAR, 1:p);
        valid   = (p+1):size(Y_VAR,1);

        Y_reg = Y_VAR(valid,:);
        X_reg = [ones(length(valid),1), Y_lag_p(valid,:)];

        nan_rows = any(~isfinite(X_reg),2) | any(~isfinite(Y_reg),2);
        X_reg(nan_rows,:) = [];
        Y_reg(nan_rows,:) = [];

        if isempty(Y_reg) || rank(X_reg) < size(X_reg,2)
            bic_vals(p) = Inf; continue;
        end

        B         = X_reg \ Y_reg;
        E         = Y_reg - X_reg*B;
        Sigma_hat = (E'*E)/size(E,1);
        detS = max(real(det(Sigma_hat)), eps);

        logL = -0.5 * size(E,1) * (K_VAR*log(2*pi) + log(detS) + K_VAR);
        k          = K_VAR*(1+K_VAR*p);
        bic_vals(p)= -2*logL + k*log(size(E,1));
    end

    [~, p_opt] = min(bic_vals);
    if isempty(p_opt) || ~isfinite(p_opt), p_opt = 1; end
end


%% ---------- EB calibration for BVAR (LBVAR or SBVAR, depending on data) ----------
function [lambda_star, delta_star] = calibrate_BVAR_EB(data, p_opt)
    fprintf('Empirical Bayes search for BVAR hyperparameters (lambda, delta)...\n');


     Z = data; %original data
    % Base regressors
    [Y, X] = build_var_regressors(Z, p_opt);
    if isempty(Y)
        error('Insufficient observations for EB search.');
    end

    % Grids: lambda in [0.3, 100], delta in {0.5, 1, 2, 5}
    lambda_grid = logspace(log10(0.3), 2, 13);
    delta_grid  = [0.5, 1.0, 2.0, 5.0];

    best_ml = -Inf;
    lambda_star = 1;
    delta_star  = 1.0;

    for L = 1:numel(lambda_grid)
        for D = 1:numel(delta_grid)
            hyp_lambda = lambda_grid(L);
            hyp_delta  = delta_grid(D);

            [Yd, Xd] = make_bvar_dummies(Z, p_opt, hyp_lambda, hyp_delta);
            Y_aug = [Y; Yd];
            X_aug = [X; Xd];

            ml = approx_marginal_likelihood(Y_aug, X_aug);

            if isscalar(ml) && isfinite(ml) && ml > best_ml
                best_ml     = ml;
                lambda_star = hyp_lambda;
                delta_star  = hyp_delta;
            end
        end
    end

    % Stability safeguard: lambda cannot be smaller than 0.3
    if lambda_star < 0.3
        lambda_star = 0.3;
    end

    fprintf('EB selected: lambda=%.3f, delta=%.3f (ML=%.2f)\n', ...
        lambda_star, delta_star, best_ml);
end



%% ---------- SBVAR variable selection (LASSO) + EB calibration ----------
function [best_sbvar_list, lambda_SBVAR, delta_SBVAR, best_rmse_val] = ...
    select_and_calibrate_SBVAR(data, var_names, target_list, p_opt, T0, H_max)

    fprintf('\n=== SBVAR variable selection using LASSO (K_max = 20) ===\n');

    [T, N]   = size(data);
    all_vars = var_names(:);

    % Identify target and non-target variables
    [tf_targets, idx_targets] = ismember(target_list, var_names);
    if any(~tf_targets)
        error('Some target variables not found in var_names.');
    end

    Y_all = data(:, idx_targets);          % T × (#targets)
    X_all = data(:, ~ismember(all_vars, target_list));   % T × (#non-targets)
    X_names = all_vars(~ismember(all_vars, target_list));
    N_non = size(X_all,2);

    % ----- step 1. LASSO selection: join all target regressions -----
    fprintf('Running LASSO for variable selection...\n');

    % Standardize predictors
    Xs = zscore(X_all);
    Xs(isnan(Xs)) = 0;

    % Stack targets into one vector
    Y_stack = [];
    X_stack = [];
    for j = 1:size(Y_all,2)
        yj = Y_all(:, j);
        idx_valid = isfinite(yj);
        Y_stack = [Y_stack; yj(idx_valid)];
        X_stack = [X_stack; Xs(idx_valid, :)];
    end

    % Fit LASSO
    [B_lasso, FitInfo] = lasso(X_stack, Y_stack, ...
        'CV', 5, 'Standardize', false);

    idx_min = FitInfo.IndexMinMSE;
    coef = B_lasso(:, idx_min);

    % Pick variables with nonzero coefficients
    selected_idx = find(abs(coef) > 1e-6);

    if isempty(selected_idx)
        warning('LASSO selected zero variables! Using top 5 by absolute coef.');
        [~, ii] = sort(abs(coef), 'descend');
        selected_idx = ii(1:5);
    end

    % Apply K_max = 10
    K_max = 10;
    if numel(selected_idx) > K_max
        [~, ii] = sort(abs(coef(selected_idx)), 'descend');
        selected_idx = selected_idx(ii(1:K_max));
    end

    selected_non_targets = X_names(selected_idx);
    fprintf('LASSO selected %d non-target variables:\n', numel(selected_non_targets));
    disp(selected_non_targets');

    % Construct SBVAR variable list
    sbvar_list_tmp = [target_list(:); selected_non_targets(:)];
    fprintf('SBVAR candidate list (%d vars):\n', numel(sbvar_list_tmp));
    disp(sbvar_list_tmp');

    % ---------- step 2. EB calibration ----------
    fprintf('\nCalibrating SBVAR (EB) on selected subset...\n');

    [lambda_SBVAR, delta_SBVAR] = calibrate_BVAR_EB( ...
        data(:, ismember(var_names, sbvar_list_tmp)), p_opt);

    % For safety ensure lambda >= 0.3
    if lambda_SBVAR < 0.3
        lambda_SBVAR = 0.3;
    end

    % ---------- step 3. Validation using recursive sample ----------
    fprintf('Validating SBVAR subset...\n');
    [~, ~, rmse_val] = validate_SBVAR_set_EB( ...
        data, var_names, target_list, sbvar_list_tmp, ...
        p_opt, T0, H_max);

    best_sbvar_list = sbvar_list_tmp;
    best_rmse_val   = rmse_val;

    fprintf('\n=== LASSO-SBVAR final set: %d vars (%d targets + %d selected) ===\n', ...
        numel(best_sbvar_list), numel(target_list), numel(best_sbvar_list)-numel(target_list));
    disp(best_sbvar_list');

    fprintf('SBVAR EB: lambda=%.3f, delta=%.3f, validation RMSE=%.4f\n', ...
        lambda_SBVAR, delta_SBVAR, best_rmse_val);
end


%% ---------- Validation of one SBVAR candidate (EB BVAR) ----------
function [lambda_sbvar, delta_sbvar, rmse_val] = ...
    validate_SBVAR_set_EB(data, var_names, target_list, sbvar_list, p_opt, T0, H_max)

    T_total = size(data,1);

    [~, idx_SBVAR] = ismember(sbvar_list, var_names);
    if any(idx_SBVAR == 0)
        error('In validate_SBVAR_set_EB: some sbvar_list vars not found in var_names.');
    end
    data_SBVAR = data(:, idx_SBVAR);

    [~, idx_VAR_full]  = ismember(target_list, var_names);
    [~, idx_VAR_SBVAR] = ismember(target_list, sbvar_list);

    valid_targets = find(idx_VAR_SBVAR > 0 & idx_VAR_full > 0);
    if isempty(valid_targets)
        error('In validate_SBVAR_set_EB: no valid targets found in sbvar_list.');
    end

    target_list_SB = target_list(valid_targets);

    % EB calibration on subset
    [lambda_sbvar, delta_sbvar] = calibrate_BVAR_EB(data_SBVAR, p_opt);

    % Recursive forecasting on validation sample
    N_pred_val = T_total - T0 - H_max;
    V          = numel(valid_targets);

    err_SBVAR_val = NaN(N_pred_val, H_max, V);

    for t0 = 1:N_pred_val
        t_train_end  = T0 + t0 - 1;
        t_test_start = t_train_end + 1;

        data_train_SBVAR = data_SBVAR(1:t_train_end,:);

        for v = 1:V
            v_global = valid_targets(v);

            idx_full  = idx_VAR_full(v_global);
            idx_sbvar = idx_VAR_SBVAR(v_global);

            for h = 1:H_max
                try
                    y_pred = run_once_BVAR_EB(data_train_SBVAR, idx_sbvar, h, ...
                                              p_opt, lambda_sbvar, delta_sbvar);
                    y_true = data(t_test_start + h - 1, idx_full);
                    err_SBVAR_val(t0,h,v) = y_pred - y_true;
                catch
                    err_SBVAR_val(t0,h,v) = NaN;
                end
            end
        end
    end

    e_vec = err_SBVAR_val(:);
    e_vec = e_vec(~isnan(e_vec));
    if isempty(e_vec)
        rmse_val = Inf;
    else
        rmse_val = sqrt(mean(e_vec.^2));
    end
end


%% ---------- DFM setup (Bai & Ng for r, q = p_opt) ----------
function [r_opt, q_opt] = setup_DFM_BaiNg(data, p_opt)
    [r_opt, table_ic] = select_factor_number_BaiNg(data);
    q_opt = p_opt;
    fprintf('→ DFM: r_opt = %d (ICp2), q_opt = %d\n', r_opt, q_opt);
    disp(table_ic);
end


%% ---------- Init forecast containers ----------
function results = init_forecast_containers(model_list, target_list, T_total, T0, H_max)

    N_pred    = T_total - T0 - H_max;
    N_targets = length(target_list);

    for m = 1:length(model_list)
        model_name = model_list{m};
        results.(model_name).pred = NaN(N_pred, H_max, N_targets);
        results.(model_name).true = NaN(N_pred, H_max, N_targets);
        results.(model_name).err  = NaN(N_pred, H_max, N_targets);
    end

    results.meta = struct('T_total',   T_total, ...
                          'T0',        T0, ...
                          'H_max',     H_max, ...
                          'target_list', {target_list}, ...
                          'model_list',  {model_list});
end


%% ---------- Recursive forecasting for VAR / LBVAR / SBVAR / DFM ----------
function results = recursive_forecasting_hybrid(data, var_names, target_list, sbvar_list, ...
    results, p, lambda_LBVAR, delta_LBVAR, lambda_SBVAR, delta_SBVAR, ...
    r_opt, q_opt, H_max, T0)

    warning('off', 'MATLAB:nearlySingularMatrix');

    N_pred = results.meta.T_total - T0 - H_max;
    [~, idx_VAR]          = ismember(target_list, var_names);
    [~, idx_SBVAR]        = ismember(sbvar_list,  var_names);
    [~, target_idx_SBVAR] = ismember(target_list, sbvar_list);

    for t0 = 1:N_pred
        t_train_end  = T0 + t0 - 1;
        t_test_start = t_train_end + 1;

        fprintf('\n[Window %d/%d] Training end = %d\n', t0, N_pred, t_train_end);

        data_train       = data(1:t_train_end,:);
        Y_VAR_train      = data_train(:, idx_VAR);
        data_train_SBVAR = data_train(:, idx_SBVAR);

        for v = 1:length(target_list)
            target_name      = target_list{v};
            target_idx_VAR   = v;
            target_idx_FULL  = idx_VAR(v);
            target_idx_SB    = target_idx_SBVAR(v);

            for h = 1:H_max

                % --- VAR (classical) ---
                try
                    y_pred_VAR = run_once_VAR(Y_VAR_train, target_idx_VAR, h, p);
                    y_true_VAR = data(t_test_start + h - 1, target_idx_FULL);
                    results.VAR.pred(t0,h,v) = y_pred_VAR;
                    results.VAR.true(t0,h,v) = y_true_VAR;
                    results.VAR.err(t0,h,v)  = y_pred_VAR - y_true_VAR;
                catch ME
                    warning('VAR failed (t0=%d, h=%d, %s): %s', ...
                            t0, h, target_name, ME.message);
                end

                % --- LBVAR (full system) ---
                try
                    y_pred_LBVAR = run_once_BVAR_EB(data_train, target_idx_FULL, h, ...
                                                    p, lambda_LBVAR, delta_LBVAR);
                    y_true_LBVAR = data(t_test_start + h - 1, target_idx_FULL);
                    results.LBVAR.pred(t0,h,v) = y_pred_LBVAR;
                    results.LBVAR.true(t0,h,v) = y_true_LBVAR;
                    results.LBVAR.err(t0,h,v)  = y_pred_LBVAR - y_true_LBVAR;
                catch ME
                    warning('LBVAR failed (t0=%d, h=%d, %s): %s', ...
                            t0, h, target_name, ME.message);
                end

                % --- SBVAR (subset system) ---
                if target_idx_SB > 0
                    try
                        y_pred_SBVAR = run_once_BVAR_EB(data_train_SBVAR, target_idx_SB, h, ...
                                                        p, lambda_SBVAR, delta_SBVAR);
                        y_true_SBVAR = data(t_test_start + h - 1, target_idx_FULL);
                        results.SBVAR.pred(t0,h,v) = y_pred_SBVAR;
                        results.SBVAR.true(t0,h,v) = y_true_SBVAR;
                        results.SBVAR.err(t0,h,v)  = y_pred_SBVAR - y_true_SBVAR;
                    catch ME
                        warning('SBVAR failed (t0=%d, h=%d, %s): %s', ...
                                t0, h, target_name, ME.message);
                    end
                else
                    results.SBVAR.pred(t0,h,v) = NaN;
                    results.SBVAR.true(t0,h,v) = NaN;
                    results.SBVAR.err(t0,h,v)  = NaN;
                end

                % --- DFM (EM + Kalman)---
                try
                    y_pred_DFM = run_once_DFM(data_train, target_idx_FULL, h, r_opt, q_opt);
                    y_true_DFM = data(t_test_start + h - 1, target_idx_FULL);
                    results.DFM.pred(t0,h,v) = y_pred_DFM;
                    results.DFM.true(t0,h,v) = y_true_DFM;
                    results.DFM.err(t0,h,v)  = y_pred_DFM - y_true_DFM;
                catch ME
                    warning('DFM failed (t0=%d, h=%d, %s): %s', ...
                            t0, h, target_name, ME.message);
                end

                % --- FAVAR (VAR + factors) ---
                try
                    y_pred_FAVAR = run_once_FAVAR(data_train, target_idx_FULL, h, p, r_opt, q_opt);
                    y_true_FAVAR = data(t_test_start + h - 1, target_idx_FULL);
                    results.FAVAR.pred(t0,h,v) = y_pred_FAVAR;
                    results.FAVAR.true(t0,h,v) = y_true_FAVAR;
                    results.FAVAR.err(t0,h,v)  = y_pred_FAVAR - y_true_FAVAR;
                catch ME
                    warning('FAVAR failed (t0=%d, h=%d, %s): %s', ...
                            t0, h, target_name, ME.message);
                end

            end
        end
    end

    fprintf('\nRecursive forecasting complete.\n');
end


%% ---------- Step 6: Evaluation ----------
function evaluate_forecasts(results, save_folder)

    models = fieldnames(results);
    models(strcmp(models,'meta')) = [];

    target_list = results.meta.target_list;
    H_max       = results.meta.H_max;
    M           = length(models);

    scale_factor = 100;

    for v = 1:length(target_list)

        target_name = target_list{v};
        fprintf('\n--- Evaluating forecasts for target: %s ---\n', target_name);

        MAE  = nan(M,H_max);
        RMSE = nan(M,H_max);

        for m = 1:M

            model = models{m};
            e_all = results.(model).err(:,:,v);

            for h = 1:H_max
                e = e_all(:,h);
                e = e(~isnan(e));

                if isempty(e)
                    MAE(m,h)  = NaN;
                    RMSE(m,h) = NaN;
                else
                    MAE(m,h)  = mean(abs(e));
                    RMSE(m,h) = sqrt(mean(e.^2));
                end
            end
        end

        MAE  = MAE  * scale_factor;
        RMSE = RMSE * scale_factor;

        all_metrics = [MAE; RMSE];
        row_names   = [strcat(models,'_MAE'); strcat(models,'_RMSE')];

        table_output = array2table(all_metrics, ...
            'VariableNames', strcat('h', string(1:H_max)), ...
            'RowNames', row_names);

        disp(table_output);

        csv_name = sprintf('Forecast_Eval_%s.csv', target_name);
        writetable(table_output, fullfile(save_folder, csv_name), 'WriteRowNames', true);
    end
end


%% ---------- VAR one-run (small system) ----------
function [y_pred_VAR, err_VAR] = run_once_VAR(data_train, target_idx, h, p_opt)

    [T, ~] = size(data_train);
    if T <= p_opt+1
        y_pred_VAR = NaN; err_VAR = NaN; return;
    end

    Y_lag = lagmatrix(data_train, 1:p_opt);
    valid = (p_opt+1):T;

    Y_reg = data_train(valid,:);
    X_reg = [ones(length(valid),1), Y_lag(valid,:)];

    nan_rows = any(~isfinite(X_reg),2) | any(~isfinite(Y_reg),2);
    X_reg(nan_rows,:) = [];
    Y_reg(nan_rows,:) = [];

    if isempty(Y_reg) || rank(X_reg) < size(X_reg,2)
        y_pred_VAR = NaN; err_VAR = NaN; return;
    end

    B = X_reg \ Y_reg;

    Y_last = data_train(end-p_opt+1:end,:);

    for t=1:h
        X_new = [1, reshape(flipud(Y_last)',1,[])];
        Y_hat = X_new*B;
        Y_last = [Y_last(2:end,:); Y_hat];
    end

    y_pred_VAR = Y_hat(1,target_idx);
    err_VAR    = NaN;
end


function [y_pred_BVAR, err_BVAR] = run_once_BVAR_EB(data_train, target_idx, h, p_opt, lambda, delta)

    if size(data_train,1) <= p_opt+1
        y_pred_BVAR = NaN; err_BVAR = NaN; return;
    end

    
    Z = data_train;

    [Y, X] = build_var_regressors(Z, p_opt);
    if isempty(Y)
        y_pred_BVAR = NaN; err_BVAR = NaN; return;
    end

    % Minnesota + DIO dummies
    [Yd, Xd] = make_bvar_dummies(Z, p_opt, lambda, delta);

    % Augmented system
    Y_aug = [Y; Yd];
    X_aug = [X; Xd];

    % Posterior mean of B
    B_post = (X_aug' * X_aug + 1e-6*eye(size(X_aug,2))) \ (X_aug' * Y_aug);

    % Rolling forward h steps
    Y_last = Z(end-p_opt+1:end,:);

    for t = 1:h
        X_new = [1, reshape(flipud(Y_last)',1,[])];
        Y_hat = X_new * B_post;
        Y_last = [Y_last(2:end,:); Y_hat];
    end

    y_pred_BVAR = Y_hat(1, target_idx);
    err_BVAR = NaN;
end


%% ---------- Helpers for BVAR: build_regressors ----------
function [Y, X] = build_var_regressors(Z, p)
    [T, K] = size(Z);
    if T <= p+1
        Y=[]; X=[]; return;
    end
    Y = Z(p+1:end, :);
    Xlags = [];
    for L=1:p
        Xlags = [Xlags, lagmatrix(Z, L)]; %#ok<AGROW>
    end
    X = Xlags(p+1:end, :);
    Y = Y(1:size(X,1), :);
    X = [ones(size(X,1),1), X];

    nan_rows = any(~isfinite(X),2)|any(~isfinite(Y),2);
    Y(nan_rows,:) = [];
    X(nan_rows,:) = [];
    X(~isfinite(X)) = 0; Y(~isfinite(Y)) = 0;
end


%% ---------- Helpers for BVAR: dummies (Minnesota mean=0 + DIO) ----------
function [Yd, Xd] = make_bvar_dummies(Z, p, lambda, delta)
    [T, K] = size(Z);
    P = 1 + K*p;

    % (1) Minnesota: mean=0, shrinkage λ / l^2
    Xd_minn = zeros(K*K*p, P);
    Yd_minn = zeros(K*K*p, K);
    row = 0;
    for i = 1:K
        for l = 1:p
            for j = 1:K
                row = row + 1;
                col = 1 + (l-1)*K + j;
                Xd_minn(row, col) = lambda / (l^2);
                Yd_minn(row, i)   = 0;
            end
        end
    end

    % (2) DIO: dummy initial observation
    if T <= p
        Xd_dio = zeros(K, P);
        Yd_dio = zeros(K, K);
    else
        t0   = p + 1;
        y0   = Z(t0, :);
        lags = zeros(1, K*p);
        for l = 1:p
            lags(1, (l-1)*K + (1:K)) = Z(t0 - l, :);
        end
        y0(~isfinite(y0))     = 0;
        lags(~isfinite(lags)) = 0;

        Xd_dio = zeros(K, P);
        Yd_dio = zeros(K, K);
        Xd_dio(:,1)     = delta;
        Xd_dio(:,2:end) = repmat(delta * lags, K, 1);
        Yd_dio          = delta * diag(y0);
    end

    Xd = [Xd_minn; Xd_dio];
    Yd = [Yd_minn; Yd_dio];
end


%% ---------- Marginal likelihood proxy ----------
function ml = approx_marginal_likelihood(Y, X)
    X(~isfinite(X)) = 0;
    Y(~isfinite(Y)) = 0;

    [T, K] = size(Y);
    nX = size(X,2);

    XtX = X' * X + 1e-10 * eye(nX);
    XtY = X' * Y;

    if rank(XtX) < nX
        ml = -Inf; return;
    end

    Bhat = XtX \ XtY;
    E = Y - X * Bhat;
    S = (E' * E) + 1e-10 * eye(K);

    [Rx, px] = chol(XtX); if px>0, ml = -Inf; return; end
    [Rs, ps] = chol(S);   if ps>0, ml = -Inf; return; end

    logdet_XtX = 2*sum(log(diag(Rx)));
    logdet_S   = 2*sum(log(diag(Rs)));

    ml = -0.5 * K * logdet_XtX  -  0.5 * T * logdet_S;
end


%% ---------- DFM: one-run using EM + KF (Method 2 style) ----------
function [y_pred_DFM, err_DFM] = run_once_DFM(X_train, target_idx, h, r_opt, q_opt)

    y_pred_DFM = NaN;
    err_DFM    = NaN;

    mX = nanmean(X_train, 1);
    sX = nanstd(X_train, 0, 1);
    sX(sX == 0 | isnan(sX)) = 1;

    X_std = (X_train - mX) ./ sX;
    X_std(isnan(X_std)) = 0;

    try
        res = DFM_EM_KF(X_std, r_opt, q_opt);
    catch ME
        warning('DFM EM/KF failed: %s', ME.message);
        return;
    end

    if ~isfield(res, 'S') || isempty(res.S)
        fprintf('[DFM] Empty res.S, skip.\n');
        return;
    end

    F_all = res.S;
    if size(F_all, 1) < size(F_all, 2)
        F_all = F_all';
    end
    if isempty(F_all)
        fprintf('[DFM] Empty F_all.\n');
        return;
    end
    if size(F_all, 2) < r_opt
        r_opt = size(F_all, 2);
    end

    F_hat = F_all(:, 1:r_opt);

    q_opt = min([q_opt, r_opt, size(F_hat, 1) - 2]);
    if q_opt < 1 || size(F_hat, 1) <= q_opt + 2
        fprintf('[DFM] Sample too short (T=%d, q=%d).\n', size(F_hat,1), q_opt);
        return;
    end

    F_lag       = lagmatrix(F_hat, 1:q_opt);
    F_lag       = F_lag(q_opt + 1:end, :);
    F_hat_valid = F_hat(q_opt + 1:end, :);

    y_target = X_std(:, target_idx);
    y_target = fillmissing(y_target, 'previous');
    y_valid  = y_target(q_opt + 1:end);

    T_min        = min([size(F_hat_valid, 1), size(F_lag, 1), length(y_valid)]);
    F_hat_valid  = F_hat_valid(1:T_min, :);
    F_lag        = F_lag(1:T_min, :);
    y_valid      = y_valid(1:T_min);

    X_reg = [ones(T_min, 1), F_hat_valid, F_lag];
    Y_reg = y_valid;

    X_reg = fillmissing(X_reg, 'previous');
    Y_reg = fillmissing(Y_reg, 'previous');

    if isempty(X_reg) || size(X_reg, 1) <= size(X_reg, 2)
        fprintf('[DFM] Not enough obs. Xrows=%d, Xcols=%d\n', size(X_reg,1), size(X_reg,2));
        beta = NaN(size(X_reg, 2), 1);
    else
        beta = X_reg \ Y_reg;
    end

    if any(isnan(beta))
        beta = fillmissing(beta, 'previous');
    end

    F_future = F_hat(end, :)';
    if any(isnan(F_future))
        F_future = fillmissing(F_future, 'previous');
    end

    x_input = [1, F_future', zeros(1, max(0, length(beta) - 1 - length(F_future)))];
    x_input = x_input(1:length(beta));

    y_pred_std  = x_input * beta;
    y_pred_DFM  = y_pred_std * sX(target_idx) + mX(target_idx);
end


%% ---------- DFM EM + Kalman ----------
function res = DFM_EM_KF(X, r, p)

    [T, N] = size(X);

    [U, S, V] = svd(X, 'econ');
    F_init      = U(:, 1:r) * S(1:r, 1:r) / sqrt(T);
    Lambda_init = V(:, 1:r) * S(1:r, 1:r) / sqrt(T);

    Psi_init = diag(var(X - F_init * Lambda_init', 0, 1, 'omitnan'));
    Psi_init(~isfinite(Psi_init)) = 1;

    A = zeros(r * p);
    A(1:r, 1:r) = eye(r);
    if p > 1
        A(r + 1:end, 1:r * (p - 1)) = eye(r * (p - 1));
    end

    Q = eye(r * p);
    C = [Lambda_init, zeros(N, r * (p - 1))];
    R = diag(Psi_init);

    S0 = zeros(r * p, 1);
    P0 = eye(r * p);

    [S_filter, P_filter, S_fore, P_fore, KL] = kalman_filter(X, S0, P0, C, R, A, Q);
    [S_smooth, ~, ~] = kalman_smoother(S_filter, P_filter, S_fore, P_fore, KL, C, R, A, Q);

    res.S = S_smooth(1:r, 2:end)';
end


%% ---------- Kalman filter ----------
function [S_filter, P_filter, S_fore, P_fore, KL] = kalman_filter(x, S0, P0, C, R, A, Q)

    T  = size(x, 1);
    nS = size(A, 1);

    S_filter      = zeros(nS, T + 1);
    S_filter(:,1) = S0;

    P_filter      = cell(T + 1, 1);
    P_filter{1}   = P0;

    S_fore        = zeros(nS, T);
    P_fore        = cell(T, 1);

    KL = cell(T, 2);

    for t = 1:T
        S_fore(:, t) = A * S_filter(:, t);
        P_fore{t}    = A * P_filter{t} * A' + Q;

        H  = C * P_fore{t} * C' + R + 1e-6 * eye(size(R));
        Kt = P_fore{t} * C' / H;

        S_filter(:, t + 1) = S_fore(:, t) + Kt * (x(t, :)' - C * S_fore(:, t));
        P_filter{t + 1}    = (eye(nS) - Kt * C) * P_fore{t};

        KL{t, 1} = Kt;
        KL{t, 2} = P_filter{t} * A' / P_fore{t};
    end
end


%% ---------- Kalman smoother ----------
function [S_smooth, P_smooth, PP_smooth] = kalman_smoother(S_filter, P_filter, S_fore, P_fore, KL, C, R, A, Q)

    [nS, T1] = size(S_filter);
    T        = T1 - 1;

    S_smooth = zeros(nS, T1);
    P_smooth = cell(T1, 1);
    PP_smooth= cell(T1, 1);

    S_smooth(:, T1) = S_filter(:, T1);
    P_smooth{T1}    = P_filter{T1};

    for t = T:-1:1
        Lt            = KL{t, 2};
        S_smooth(:,t) = S_filter(:,t) + Lt * (S_smooth(:,t+1) - S_fore(:,t));
        P_smooth{t}   = P_filter{t} - Lt * (P_fore{t} - P_smooth{t+1}) * Lt';
    end
end


%% ---------- Bai & Ng factor number selection ----------
function [r_opt, table_ic] = select_factor_number_BaiNg(X_raw)

    [T,N] = size(X_raw);

    mX = nanmean(X_raw,1);
    sX = nanstd(X_raw,0,1);
    sX(sX==0|isnan(sX))=1;
    X  = (X_raw - mX)./sX;
    X(isnan(X)) = 0;

    r_max = min(8,min(T,N));

    [U,S,V] = svd(X,'econ');
    SSR = zeros(r_max+1,1);
    SSR(1) = sum(X(:).^2);

    for r=1:r_max
        X_r     = U(:,1:r)*S(1:r,1:r)*V(:,1:r)';
        SSR(r+1)= sum((X-X_r).^2,'all');
    end

    ICp1 = zeros(r_max+1,1);
    ICp2 = ICp1;
    ICp3 = ICp1;

    for r=0:r_max
        V_r = SSR(r+1)/(N*T);
        if V_r<=0 || isnan(V_r)
            V_r=1e-12;
        end

        penalty1 = r*(N+T)/(N*T)*log((N*T)/(N+T));
        penalty2 = r*(N+T)/(N*T)*log(min(N,T));
        penalty3 = r*log(min(N,T))/min(N,T);

        ICp1(r+1)=log(V_r)+penalty1;
        ICp2(r+1)=log(V_r)+penalty2;
        ICp3(r+1)=log(V_r)+penalty3;
    end

    [~,idx_min] = min(ICp2);
    r_opt       = idx_min-1;

    table_ic = table((0:r_max)',SSR,ICp1,ICp2,ICp3,...
        'VariableNames',{'r','SSR','ICp1','ICp2','ICp3'});
    fprintf('Optimal number of factors (ICp2): r = %d\n', r_opt);
end


%% ---------- FAVAR one-run: VAR on [y, F] ----------
function [y_pred_FAVAR, err_FAVAR] = run_once_FAVAR(X_train, target_idx, h, p_opt, r_opt, q_opt)

    y_pred_FAVAR = NaN;
    err_FAVAR    = NaN;

    [T, ~] = size(X_train);
    if T <= p_opt + 5
        return;  
    end

   
    mX = nanmean(X_train, 1);
    sX = nanstd(X_train, 0, 1);
    sX(sX == 0 | isnan(sX)) = 1;
    X_std = (X_train - mX) ./ sX;
    X_std(isnan(X_std)) = 0;

    % factors from DFM_EM_KF
    try
        res = DFM_EM_KF(X_std, r_opt, q_opt);
    catch ME
        warning('FAVAR: DFM_EM_KF failed: %s', ME.message);
        return;
    end

    if ~isfield(res, 'S') || isempty(res.S)
        fprintf('[FAVAR] Empty res.S, skip.\n');
        return;
    end

    F_all = res.S;
    if size(F_all, 1) < size(F_all, 2)
        F_all = F_all';
    end
    if isempty(F_all)
        fprintf('[FAVAR] Empty F_all.\n');
        return;
    end

    
    r_eff = min(r_opt, size(F_all, 2));
    F_hat = F_all(:, 1:r_eff);   % T × r_eff

    
    y_std = X_std(:, target_idx);   % T × 1
    y_std = fillmissing(y_std, 'previous');

    % FAVAR variables
    Z = [y_std, F_hat];   % T × (1 + r_eff)

    
    try
        [y_pred_std, ~] = run_once_VAR(Z, 1, h, p_opt);
    catch ME
        warning('FAVAR: VAR on [y,F] failed: %s', ME.message);
        return;
    end

    y_pred_FAVAR = y_pred_std * sX(target_idx) + mX(target_idx);
end



%% ---------- Plot RMSE across horizons ----------
function plot_RMSE_curves(results, save_folder)

    models = fieldnames(results);
    models(strcmp(models,'meta')) = [];

    target_list = results.meta.target_list;
    H_max       = results.meta.H_max;
    M           = length(models);

    cols = lines(M);

    for v = 1:length(target_list)

        target_name = target_list{v};

        figure('Name', ['RMSE - ' target_name], 'Color', 'w'); hold on;

        for m = 1:M
            model = models{m};
            e_all = results.(model).err(:,:,v);

            RMSE_h = nan(H_max,1);
            for h = 1:H_max
                e = e_all(:,h);
                e = e(~isnan(e));
                if isempty(e)
                    RMSE_h(h) = NaN;
                else
                    RMSE_h(h) = sqrt(mean(e.^2));
                end
            end

            plot(1:H_max, RMSE_h, '-o', ...
                'LineWidth', 1.8, ...
                'MarkerSize', 6, ...
                'Color', cols(m,:));
        end

        xlabel('Forecast horizon h');
        ylabel('RMSE');
        title(['RMSE across horizons - ' target_name]);
        legend(models, 'Location', 'northwest');
        grid on;

        saveas(gcf, fullfile(save_folder, sprintf('RMSE_%s.png', target_name)));
    end
end
