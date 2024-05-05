t_values = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.0010];
N_values = [500, 1000, 1500, 2000, 2500, 3000];

prompt1(t_values);
prompt2(N_values);

function prompt1(t_values)
    N = 1000;
    [b, v, lambda, A] = setup(N); %ensure that b, v, lambda are columns
    
    for idx = 1:length(t_values)
        t = t_values(idx);
        f = @(x) exp(-1i*t*x);

        % Compute using Fourier Matvec
        w_fourier_matvec = fourier_matvec(f, lambda, v');

        % Compute using Direct method
        A_input = -1i*t*A;
        w_direct_compute = expm(A_input)*v;

        %figure
        figure;
        % Plot result from my implementation
        subplot(1, 2, 1); 
        plot(real(w_fourier_matvec));
        title(sprintf('Fourier Matvec at t = %.4f', t));
        % Plot result from Direct Compute 
        subplot(1, 2, 2);
        plot(real(w_direct_compute));
        title(sprintf('Direct Compute at t = %.4f', t));
    end
end

function prompt2(N_values)
    t = 0.0010;
    f = @(x) exp(-1i*t*x);
    
    results_table = table(N_values', zeros(length(N_values), 1), zeros(length(N_values), 1), ...
                          'VariableNames', {'N', 'Fourier_Matvec_Time', 'Direct_Compute_Time'});
    
    for idx = 1:length(N_values)
        N = N_values(idx);
        [b, v, lambda, A] = setup(N); %b, v, lambda, A depends on N

        % Compute using Fourier Matvec
        tic;
        w_fourier_matvec = fourier_matvec(f, lambda, v);
        results_table.Fourier_Matvec_Time(idx) = toc;

        % Compute using Direct method
        tic;
        A_input = -1i*t*A;
        w_direct_compute = expm(A_input)*v;
        results_table.Direct_Compute_Time(idx) = toc;
    end
    
    % Display the results table
    disp(results_table);
    % Plot the asymptotic behavior of time complexity
    prompt2_2(results_table);
end

function prompt2_2(results_table) %Acknowledge of using Chatgpt
    N = results_table.N;
    fourier_times = results_table.Fourier_Matvec_Time;
    direct_compute_times = results_table.Direct_Compute_Time;

    NlogN = N .* log2(N);
    Ncubed = N.^3;

    % Scaling constants for plotting purposes
    scale_NlogN = min(fourier_times) / min(NlogN);
    scale_Ncubed = min(direct_compute_times) / min(Ncubed);

    figure;
    loglog(N, fourier_times, 'bo-', 'DisplayName', 'Fourier Matvec Times');
    hold on;
    loglog(N, NlogN * scale_NlogN, 'r--', 'DisplayName', 'N log N Scaling');
    loglog(N, direct_compute_times, 'gs-', 'DisplayName', 'Direct Compute Times');
    loglog(N, Ncubed * scale_Ncubed, 'k--', 'DisplayName', 'N^3 Scaling');
    xlabel('N');
    ylabel('Time (s)');
    legend('show', 'Location', 'northwest');
    title('Time Complexity Analysis');
    hold off;
end

function w = fourier_matvec(f, lambda, v) %suppose that lambda is a array of lambda's, not whole matrix 
    v_1 = my_ivsfft(v); %i/o is column vector
    lambda_lc = lin_comb_lambda(lambda, f); %arrays of diagonal entries of Lambda'
    v_2 = lambda_lc .* v_1'; %element-wise multiplication
    w = fft(v_2); %get column vector and output column vector
end

function [b, v, lambda, A] = setup(N)
    b = zeros(N, 1);
    b(1) = 2*N^2;
    b(2) = -1 * N^2;
    b(N) = -1 * N^2;
    
    k = (0: N-1)';
    v = exp(-1*(k/N - 1/2).^2 / (2 * 0.01^2));
    % rowVector = [b(1); fliplr(b(2:end))'];
    b_in = b';
    A = toeplitz(b, [b(1) ; fliplr(b(2:end))]);
    lambda = get_fourier_diag(b);
end

function [y_hat] = my_ivsfft(y) %To handle the case when N = 2^n
    % Based on the theoretical derivation to get the ifft
    y_hat = fft(y);
    y_hat = conj(y_hat) / length(y);
end

function lambda = get_fourier_diag(b)
    lambda_0 = fft(b);
    lambda_0 = conj(lambda_0);
    lambda = lambda_0;
end

%Apply the function entrywise
function lambda_lc = lin_comb_lambda(lambda, f) 
    N = length(lambda);
    result = zeros(N, 1);
    for k = 1:N
        result(k) = f(lambda(k));
    end
    lambda_lc = result;
end