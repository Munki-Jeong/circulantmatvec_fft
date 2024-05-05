# Efficient Matrix-Vector Multiplication for Circulant Matrices

This project provides an efficient method to compute \(f(A)v\), where \(A\) is a circulant matrix, and \(f\) is a function from \(\mathbb{R}\) to \(\mathbb{R}\). The code is designed to efficiently compute the result using the diagonalization properties of circulant matrices and FFT.

## Overview

### Problem Formulation

**Given**:
1. A circulant matrix \(A\) defined by vector \(b \in \mathbb{C}^n\).
2. A function \(f:\mathbb{R}\rightarrow\mathbb{R}\) that can be evaluated at any given \(x\).
3. A vector \(v = (v_0, \ldots, v_{N-1})^T\).

**Objective**:
- Compute \(w = f(A)v\) efficiently.

### Approach

#### Mathematical Formulation

1. **Diagonalization**: \(A\) can be diagonalized as \(A = U \Lambda U^{-1}\), where \(U\) is the DFT matrix.
2. **Compact Formulation**: Using diagonalization, we express \(A^n\) and ultimately compute \(f(A)v\) in terms of the FFT.

#### Algorithm

1. **Compute \(F^{-1}v\)**.
2. **Apply \(f\) to the diagonal entries of \(\Lambda\)**.
3. **Compute \(Fv''\)** using FFT.

### Implementation

Below are the key functions for this approach.

#### Key Functions

```matlab
function w = fourier_matvec(f, lambda, v)
    v_1 = my_ivsfft(v);
    lambda_lc = lin_comb_lambda(lambda, f);
    v_2 = lambda_lc .* v_1';
    w = fft(v_2);
end

function [y_hat] = my_ivsfft(y)
    y_hat = fft(y);
    y_hat = conj(y_hat) / length(y);
end

function lambda = get_fourier_diag(b)
    lambda_0 = fft(b);
    lambda_0 = conj(lambda_0);
    lambda = lambda_0;
end

function lambda_lc = lin_comb_lambda(lambda, f)
    N = length(lambda);
    result = zeros(N, 1);
    for k = 1:N
        result(k) = f(lambda(k));
    end
    lambda_lc = result;
end
