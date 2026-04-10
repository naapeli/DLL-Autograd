#include "ops/linalg.h"
#include "ops/binary.h"
#include "ops/unary.h"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include <tuple>

// LAPACK signatures (Fortran naming convention)
extern "C" {
    void sgetrf_(int* m, int* n, float* a, int* lda, int* ipiv, int* info);
    void sgetrs_(char* trans, int* n, int* nrhs, float* a, int* lda, int* ipiv, float* b, int* ldb, int* info);
    void sgetri_(int* n, float* a, int* lda, int* ipiv, float* work, int* lwork, int* info);
    void sgesv_(int* n, int* nrhs, float* a, int* lda, int* ipiv, float* b, int* ldb, int* info);
    void spotrf_(char* uplo, int* n, float* a, int* lda, int* info);
    void sgesvd_(char* jobu, char* jobvt, int* m, int* n, float* a, int* lda, float* s, float* u, int* ldu, float* vt, int* ldvt, float* work, int* lwork, int* info);
    void ssyev_(char* jobz, char* uplo, int* n, float* a, int* lda, float* w, float* work, int* lwork, int* info);
    void sgeqrf_(int* m, int* n, float* a, int* lda, float* tau, float* work, int* lwork, int* info);
    void sorgqr_(int* m, int* n, int* k, float* a, int* lda, float* tau, float* work, int* lwork, int* info);
    void spotrs_(char* uplo, int* n, int* nrhs, float* a, int* lda, float* b, int* ldb, int* info);
    void sgels_(char* trans, int* m, int* n, int* nrhs, float* a, int* lda, float* b, int* ldb, float* work, int* lwork, int* info);
}

void check_square(const std::shared_ptr<Tensor>& a) {
    int ndim = a->shape.size();
    if (ndim < 2 || a->shape[ndim-1] != a->shape[ndim-2]) {
        throw std::invalid_argument("Linalg operation requires a square matrix (shape [..., N, N]).");
    }
}

std::shared_ptr<Tensor> inverse(const std::shared_ptr<Tensor>& a) {
    // TODO: implement gpu acceleration for this operation
    check_square(a);
    int ndim = a->shape.size();
    int n = a->shape[ndim - 1];
    int matrix_size = n * n;
    int num_batches = a->numel() / matrix_size;
    
    a->ensure_cpu_data();
    auto out_data = std::make_shared<std::vector<float>>(*a->data);
    
    for (int b = 0; b < num_batches; ++b) {
        float* batch_ptr = out_data->data() + b * matrix_size;
        std::vector<int> ipiv(n);
        int info;
        
        // 1. LU Factorization
        sgetrf_(&n, &n, batch_ptr, &n, ipiv.data(), &info);
        if (info != 0) {
            throw std::runtime_error("LAPACK sgetrf failed: Matrix at batch index " + std::to_string(b) + " is singular.");
        }
        
        // 2. Query optimal work size for inversion
        float work_query;
        int lwork = -1;
        sgetri_(&n, batch_ptr, &n, ipiv.data(), &work_query, &lwork, &info);
        
        lwork = (int)work_query;
        std::vector<float> work(lwork);
        
        // 3. Inversion
        sgetri_(&n, batch_ptr, &n, ipiv.data(), work.data(), &lwork, &info);
        if (info != 0) {
            throw std::runtime_error("LAPACK sgetri failed at batch index " + std::to_string(b) + ".");
        }
    }
    
    auto out = std::make_shared<Tensor>(out_data, a->shape);
    out->requires_grad = a->requires_grad;
    
    if (a->requires_grad) {
        out->_prev = {a};
        out->_backward = [a, out]() {
            if (!a->grad) a->zero_grad();
            if (!out->grad) return;
            
            // Formula: grad_a = -(out^T) @ grad_out @ (out^T)
            // out is the already inverted matrix A^-1
            auto out_t = transpose(out);
            auto m1 = matmul(out_t, out->grad);
            auto grad_a_update = matmul(m1, out_t);
            
            auto new_grad = sub(a->grad, grad_a_update);
            a->grad = new_grad;
        };
    }
    
    return out;
}

std::shared_ptr<Tensor> determinant(const std::shared_ptr<Tensor>& a) {
    // TODO: implement gpu acceleration for this operation
    check_square(a);
    int ndim = a->shape.size();
    int n = a->shape[ndim-1];
    int matrix_size = n * n;
    int num_batches = a->numel() / matrix_size;
    
    a->ensure_cpu_data();
    auto a_data = *a->data;
    
    std::vector<float> det_data(num_batches);
    std::vector<int> out_shape;
    if (ndim > 2) {
        for (int i = 0; i < ndim - 2; ++i) out_shape.push_back(a->shape[i]);
    } else {
        out_shape = {1};
    }
    
    for (int b = 0; b < num_batches; ++b) {
        std::vector<float> temp_data(a_data.begin() + b * matrix_size, a_data.begin() + (b + 1) * matrix_size);
        
        std::vector<int> ipiv(n);
        int info;
        
        sgetrf_(&n, &n, temp_data.data(), &n, ipiv.data(), &info);
        
        float det = 1.0f;
        int swaps = 0;
        for (int i = 0; i < n; ++i) {
            det *= temp_data[i * n + i]; // Diagonal of U
            if (ipiv[i] != i + 1) swaps++;
        }
        
        if (swaps % 2 != 0) det = -det;
        det_data[b] = det;
    }
    
    auto out = std::make_shared<Tensor>(det_data, out_shape);
    out->requires_grad = a->requires_grad;
    
    if (a->requires_grad) {
        out->_prev = {a};
        out->_backward = [a, out]() {
            if (!a->grad) a->zero_grad();
            if (!out->grad) return;
            
            // Formula: grad_a = det(A) * grad_det * (A^-T)
            auto a_inv_t = transpose(inverse(a));
            auto det_val = out; // det(A)
            
            // We need to broadcast det_val and out->grad (which are same shape)
            // to shape of a_inv_t [..., N, N]
            auto term = mul(det_val, out->grad);
            // Reshape term [...] to [..., 1, 1] for broadcasting
            std::vector<int> term_shape = out->shape;
            term_shape.push_back(1);
            term_shape.push_back(1);
            auto term_reshaped = std::make_shared<Tensor>(term->data, term_shape);
            
            auto grad_a_update = mul(term_reshaped, a_inv_t);
            
            a->grad = add(a->grad, grad_a_update);
        };
    }
    
    return out;
}

std::shared_ptr<Tensor> solve(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    // TODO: implement gpu acceleration for this operation
    check_square(a);
    int ndim_a = a->shape.size();
    int ndim_b = b->shape.size();
    int n = a->shape[ndim_a - 1];
    
    if (b->shape[ndim_b - 2] != n) {
        throw std::invalid_argument("Incompatible dimensions for solve: A is " + std::to_string(n) + "x" + std::to_string(n) + ", but B's matrix dimension is " + std::to_string(b->shape[ndim_b-2]));
    }
    int nrhs = b->shape[ndim_b - 1];
    
    // Determine output batch shape
    int batch_ndim_a = ndim_a - 2;
    int batch_ndim_b = ndim_b - 2;
    int batch_ndim = std::max(batch_ndim_a, batch_ndim_b);
    std::vector<int> out_batch_shape(batch_ndim);
    std::vector<int> a_batch_strides(batch_ndim, 0);
    std::vector<int> b_batch_strides(batch_ndim, 0);
    std::vector<int> out_batch_strides(batch_ndim, 0);

    // Contig strides for batch part
    std::vector<int> contig_strides_a(ndim_a, 1);
    for (int i = ndim_a - 2; i >= 0; --i) contig_strides_a[i] = contig_strides_a[i+1] * a->shape[i+1];
    std::vector<int> contig_strides_b(ndim_b, 1);
    for (int i = ndim_b - 2; i >= 0; --i) contig_strides_b[i] = contig_strides_b[i+1] * b->shape[i+1];
    
    for (int i = 0; i < batch_ndim; ++i) {
        int dim_a = (i < batch_ndim - batch_ndim_a) ? 1 : a->shape[i - (batch_ndim - batch_ndim_a)];
        int dim_b = (i < batch_ndim - batch_ndim_b) ? 1 : b->shape[i - (batch_ndim - batch_ndim_b)];
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::invalid_argument("Solve batch shapes are not broadcastable.");
        }
        out_batch_shape[i] = std::max(dim_a, dim_b);
        a_batch_strides[i] = (dim_a == 1) ? 0 : contig_strides_a[i - (batch_ndim - batch_ndim_a)];
        b_batch_strides[i] = (dim_b == 1) ? 0 : contig_strides_b[i - (batch_ndim - batch_ndim_b)];
    }
    
    int num_matrices = 1;
    for (int i = batch_ndim - 1; i >= 0; --i) {
        out_batch_strides[i] = num_matrices;
        num_matrices *= out_batch_shape[i];
    }
    
    std::vector<int> out_shape = out_batch_shape;
    out_shape.push_back(n);
    out_shape.push_back(nrhs);
    
    a->ensure_cpu_data();
    b->ensure_cpu_data();
    auto out_data = std::make_shared<std::vector<float>>(num_matrices * n * nrhs);
    
    for (int b_idx = 0; b_idx < num_matrices; ++b_idx) {
        int temp = b_idx;
        int offset_a = 0;
        int offset_b = 0;
        for (int d = 0; d < batch_ndim; ++d) {
            int coord = temp / out_batch_strides[d];
            temp %= out_batch_strides[d];
            offset_a += coord * a_batch_strides[d];
            offset_b += coord * b_batch_strides[d];
        }
        
        // Transpose A and B to column-major
        std::vector<float> a_cm(n * n);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) a_cm[j * n + i] = (*a->data)[offset_a + i * n + j];
        
        std::vector<float> b_cm(n * nrhs);
        for (int i = 0; i < n; i++) for (int j = 0; j < nrhs; j++) b_cm[j * n + i] = (*b->data)[offset_b + i * nrhs + j];
        
        std::vector<int> ipiv(n);
        int info;
        sgetrf_(&n, &n, a_cm.data(), &n, ipiv.data(), &info);
        if (info != 0) throw std::runtime_error("LAPACK sgetrf failed at batch " + std::to_string(b_idx));
        
        char trans = 'N';
        sgetrs_(&trans, &n, &nrhs, a_cm.data(), &n, ipiv.data(), b_cm.data(), &n, &info);
        if (info != 0) throw std::runtime_error("LAPACK sgetrs failed at batch " + std::to_string(b_idx));
        
        // Transpose B back to row-major
        float* out_ptr = out_data->data() + b_idx * n * nrhs;
        for (int i = 0; i < n; i++) for (int j = 0; j < nrhs; j++) out_ptr[i * nrhs + j] = b_cm[j * n + i];
    }
    
    auto out = std::make_shared<Tensor>(out_data, out_shape);
    out->requires_grad = a->requires_grad || b->requires_grad;
    
    if (out->requires_grad) {
        out->_prev = {a, b};
        out->_backward = [a, b, out]() {
            if (!out->grad) return;
            
            // For AX = B, X = A^-1 B
            // grad_B = A^-T @ grad_X
            // grad_A = - (grad_B @ X^T) = - (A^-T @ grad_X @ X^T)
            
            auto a_inv_t = transpose(inverse(a));
            auto grad_b = matmul(a_inv_t, out->grad);
            
            if (b->requires_grad) {
                if (!b->grad) b->zero_grad();
                b->grad = add(b->grad, grad_b);
            }
            
            if (a->requires_grad) {
                if (!a->grad) a->zero_grad();
                auto x_t = transpose(out);
                auto grad_a_update = mul_scalar(matmul(grad_b, x_t), -1.0f);
                a->grad = add(a->grad, grad_a_update);
            }
        };
    }
    
    return out;
}

std::shared_ptr<Tensor> cholesky(const std::shared_ptr<Tensor>& a) {
    // TODO: implement gpu acceleration for this operation
    check_square(a);
    int ndim = a->shape.size();
    int n = a->shape[ndim - 1];
    int matrix_size = n * n;
    int num_batches = a->numel() / matrix_size;
    
    a->ensure_cpu_data();
    auto out_data = std::make_shared<std::vector<float>>(*a->data);
    
    for (int b = 0; b < num_batches; ++b) {
        float* batch_ptr = out_data->data() + b * matrix_size;
        char uplo = 'U'; // LAPACK col-major 'U' = row-major 'L'
        int info;
        
        spotrf_(&uplo, &n, batch_ptr, &n, &info);
        if (info != 0) {
            throw std::runtime_error("LAPACK spotrf failed at batch " + std::to_string(b) + ": Matrix is not positive-definite.");
        }
        
        // Zero out the upper triangle
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                batch_ptr[i * n + j] = 0.0f;
            }
        }
    }
    
    auto out = std::make_shared<Tensor>(out_data, a->shape);
    out->requires_grad = a->requires_grad;
    
    if (a->requires_grad) {
        out->_prev = {a};
        out->_backward = [a, out]() {
            if (!a->grad) a->zero_grad();
            if (!out->grad) return;
            
            // Formula from Reverse-Mode Differentiation of Cholesky
            // S = L^T @ grad_L
            auto S = matmul(transpose(out), out->grad);
            
            S->ensure_cpu_data();
            auto Phi_S_data = std::make_shared<std::vector<float>>(*S->data);
            int n = S->shape[S->shape.size() - 1];
            int matrix_size = n * n;
            int num_batches = S->numel() / matrix_size;
            
            for (int b = 0; b < num_batches; ++b) {
                float* batch_ptr = Phi_S_data->data() + b * matrix_size;
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        if (i < j) {
                            batch_ptr[i * n + j] = 0.0f;
                        } else if (i == j) {
                            batch_ptr[i * n + j] *= 0.5f;
                        }
                    }
                }
            }
            auto Phi_S = std::make_shared<Tensor>(Phi_S_data, S->shape);
            
            auto L_inv = inverse(out);
            auto L_inv_t = transpose(L_inv);
            
            // dA = L^{-T} @ Phi_S @ L^{-1}
            auto tmp = matmul(L_inv_t, Phi_S);
            auto grad_a_update = matmul(tmp, L_inv);
            
            // Symmetrize grad_a
            auto grad_a_sym = mul_scalar(add(grad_a_update, transpose(grad_a_update)), 0.5f);
            
            a->grad = add(a->grad, grad_a_sym);
        };
    }
    
    return out;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> svd(const std::shared_ptr<Tensor>& a, bool full_matrices) {
    int ndim = a->shape.size();
    if (ndim < 2) {
        throw std::invalid_argument("SVD operation requires at least a 2D matrix.");
    }
    int M = a->shape[ndim - 2];
    int N = a->shape[ndim - 1];
    int min_mn = std::min(M, N);
    int matrix_size = M * N;
    int num_batches = a->numel() / matrix_size;
    
    a->ensure_cpu_data();
    auto a_data_copy = *a->data;
    
    // LAPACK operates column-major, so passing row-major A is like computing SVD of A^T.
    // A^T = V S U^T, so LAPACK's "U" output is our V, and LAPACK's "Vt" output is our U^T.
    int R = N; // rows of A^T
    int C = M; // cols of A^T
    
    char jobu, jobvt;
    int u_cols, vt_rows;
    if (full_matrices) {
        jobu = 'A'; vt_rows = C;
        jobvt = 'A'; u_cols = R;
    } else {
        jobu = 'S'; vt_rows = min_mn;
        jobvt = 'S'; u_cols = min_mn;
    }
    
    int U_rows = M;
    int U_cols = full_matrices ? M : min_mn;
    int Vt_rows = full_matrices ? N : min_mn;
    int Vt_cols = N;
    
    auto s_total_data = std::make_shared<std::vector<float>>(num_batches * min_mn);
    auto u_total_data = std::make_shared<std::vector<float>>(num_batches * U_rows * U_cols);
    auto vt_total_data = std::make_shared<std::vector<float>>(num_batches * Vt_rows * Vt_cols);
    
    for (int b = 0; b < num_batches; ++b) {
        float* a_batch_ptr = a_data_copy.data() + b * matrix_size;
        float* s_batch_ptr = s_total_data->data() + b * min_mn;
        float* u_batch_ptr = u_total_data->data() + b * U_rows * U_cols;
        float* vt_batch_ptr = vt_total_data->data() + b * Vt_rows * Vt_cols;
        
        // Note: because we are treating row-major as col-major, 
        // LAPACK U (Rxu_cols) -> Vt_rows x Vt_cols
        // LAPACK Vt (vtxC) -> U_rows x U_cols
        float work_query;
        int lwork = -1;
        int info;
        sgesvd_(&jobu, &jobvt, &R, &C, a_batch_ptr, &R, s_batch_ptr, vt_batch_ptr, &R, u_batch_ptr, &vt_rows, &work_query, &lwork, &info);
        lwork = (int)work_query;
        std::vector<float> work(lwork);
        sgesvd_(&jobu, &jobvt, &R, &C, a_batch_ptr, &R, s_batch_ptr, vt_batch_ptr, &R, u_batch_ptr, &vt_rows, work.data(), &lwork, &info);
        if (info != 0) throw std::runtime_error("LAPACK sgesvd failed at batch " + std::to_string(b));
    }
    
    std::vector<int> u_shape = a->shape; u_shape[ndim-2] = U_rows; u_shape[ndim-1] = U_cols;
    std::vector<int> s_shape = a->shape; s_shape.pop_back(); s_shape[ndim-2] = min_mn;
    std::vector<int> vt_shape = a->shape; vt_shape[ndim-2] = Vt_rows; vt_shape[ndim-1] = Vt_cols;
    
    auto U = std::make_shared<Tensor>(u_total_data, u_shape);
    auto S = std::make_shared<Tensor>(s_total_data, s_shape);
    auto Vt = std::make_shared<Tensor>(vt_total_data, vt_shape);
    
    U->requires_grad = a->requires_grad; S->requires_grad = a->requires_grad; Vt->requires_grad = a->requires_grad;
    
    if (a->requires_grad) {
        U->_prev = {a}; S->_prev = {a}; Vt->_prev = {a};
        int M_cap = M, N_cap = N, K_cap = min_mn;
        auto svd_backward_helper = [a, U, S, Vt, M_cap, N_cap, K_cap](const std::shared_ptr<Tensor>& dU, const std::shared_ptr<Tensor>& dS, const std::shared_ptr<Tensor>& dVt) {
            int K = K_cap;
            int num_batches = S->numel() / K;
            
            // F_ij = 1 / (s_j^2 - s_i^2)
            std::vector<float> F_data(num_batches * K * K, 0.0f);
            S->ensure_cpu_data();
            for (int b = 0; b < num_batches; ++b) {
                for (int i = 0; i < K; ++i) {
                    float s_i = (*S->data)[b * K + i];
                    for (int j = 0; j < K; ++j) {
                        if (i != j) {
                            float s_j = (*S->data)[b * K + j];
                            float diff = s_j * s_j - s_i * s_i;
                            if (std::abs(diff) < 1e-12f) diff = (diff >= 0) ? 1e-12f : -1e-12f;
                            F_data[b * K * K + i * K + j] = 1.0f / diff;
                        }
                    }
                }
            }
            auto F = std::make_shared<Tensor>(F_data, std::vector<int>{num_batches, K, K});
            // Manual diagonal embedding for batches
            std::vector<int> s_mat_shape = S->shape; s_mat_shape.push_back(K);
            std::vector<float> s_mat_data(num_batches * K * K, 0.0f);
            std::vector<float> ds_mat_data(num_batches * K * K, 0.0f);
            
            S->ensure_cpu_data();
            dS->ensure_cpu_data();
            for (int b = 0; b < num_batches; ++b) {
                for (int i = 0; i < K; ++i) {
                    s_mat_data[b * K * K + i * K + i] = (*S->data)[b * K + i];
                    ds_mat_data[b * K * K + i * K + i] = (*dS->data)[b * K + i];
                }
            }
            auto S_matrix = std::make_shared<Tensor>(s_mat_data, s_mat_shape);
            auto dS_matrix = std::make_shared<Tensor>(ds_mat_data, s_mat_shape);
            
            auto Ut = transpose(U);
            auto V = transpose(Vt);
            auto dV = transpose(dVt);
            
            // M1 = F * (U^T dU - dU^T U)
            auto Ut_dU = matmul(Ut, dU);
            auto M1 = mul(F, sub(Ut_dU, transpose(Ut_dU)));
            
            // M2 = F * (V^T dV - dV^T V)
            auto Vt_dV = matmul(Vt, dV);
            auto M2 = mul(F, sub(Vt_dV, transpose(Vt_dV)));
            
            // dA_mid = U @ (dS_matrix + M1 @ S + S @ M2) @ V^T
            auto term1 = matmul(M1, S_matrix);
            auto term2 = matmul(S_matrix, M2);
            auto inner = add(dS_matrix, add(term1, term2));
            auto dA = matmul(U, matmul(inner, Vt));
            
            if (M_cap > K) {
                // Projection for tall matrices: (I - U U^T) dU S^{-1} V^T
                auto UtU = matmul(U, Ut);
                std::vector<float> I_data(num_batches * M_cap * M_cap, 0.0f);
                for (int b = 0; b < num_batches; ++b) for (int i = 0; i < M_cap; ++i) I_data[b * M_cap * M_cap + i * M_cap + i] = 1.0f;
                auto I_mat = std::make_shared<Tensor>(I_data, std::vector<int>{num_batches, M_cap, M_cap});
                auto proj = sub(I_mat, UtU);
                
                std::vector<float> invS_diag_data(num_batches * K * K, 0.0f);
                for (int i = 0; i < S->numel(); ++i) {
                    int b = i / K;
                    int k = i % K;
                    invS_diag_data[b * K * K + k * K + k] = 1.0f / ((*S->data)[i] + 1e-12f);
                }
                auto invS = std::make_shared<Tensor>(invS_diag_data, s_mat_shape);
                dA = add(dA, matmul(proj, matmul(dU, matmul(invS, Vt))));
            }
            if (N_cap > K) {
                // Projection for wide matrices: U S^{-1} dV^T (I - V V^T)
                auto VVt = matmul(V, transpose(V));
                std::vector<float> I_data(num_batches * N_cap * N_cap, 0.0f);
                for (int b = 0; b < num_batches; ++b) for (int i = 0; i < N_cap; ++i) I_data[b * N_cap * N_cap + i * N_cap + i] = 1.0f;
                auto I_mat = std::make_shared<Tensor>(I_data, std::vector<int>{num_batches, N_cap, N_cap});
                auto proj = sub(I_mat, VVt);
                
                std::vector<float> invS_diag_data(num_batches * K * K, 0.0f);
                for (int i = 0; i < S->numel(); ++i) {
                    int b = i / K;
                    int k = i % K;
                    invS_diag_data[b * K * K + k * K + k] = 1.0f / ((*S->data)[i] + 1e-12f);
                }
                auto invS = std::make_shared<Tensor>(invS_diag_data, s_mat_shape);
                dA = add(dA, matmul(U, matmul(invS, matmul(transpose(dV), proj))));
            }
            
            if (!a->grad) a->zero_grad();
            a->grad = add(a->grad, dA);
        };
        
        U->_backward = [a, U, S, Vt, svd_backward_helper]() {
            if (!U->grad) return;
            svd_backward_helper(U->grad, std::make_shared<Tensor>(std::vector<float>(S->numel(), 0.0f), S->shape), 
                                std::make_shared<Tensor>(std::vector<float>(Vt->numel(), 0.0f), Vt->shape));
        };
        
        S->_backward = [a, U, S, Vt, svd_backward_helper]() {
            if (!S->grad) return;
            svd_backward_helper(std::make_shared<Tensor>(std::vector<float>(U->numel(), 0.0f), U->shape), S->grad, 
                                std::make_shared<Tensor>(std::vector<float>(Vt->numel(), 0.0f), Vt->shape));
        };
        
        Vt->_backward = [a, U, S, Vt, svd_backward_helper]() {
            if (!Vt->grad) return;
            svd_backward_helper(std::make_shared<Tensor>(std::vector<float>(U->numel(), 0.0f), U->shape), 
                                std::make_shared<Tensor>(std::vector<float>(S->numel(), 0.0f), S->shape), Vt->grad);
        };
    }

    
    return std::make_tuple(U, S, Vt);
}

std::shared_ptr<Tensor> diag(const std::shared_ptr<Tensor>& a, int diagonal) {
    a->ensure_cpu_data();
    int ndim = a->shape.size();
    
    if (ndim >= 1 && (ndim == 1 || a->shape[ndim-1] != a->shape[ndim-2] || ndim > 2)) {
        // We assume ndim=1 or ndim > 1 where we want to embed the last dim as a diagonal
        // PyTorch logic: if 1D, embed. If >1D, usually it extracts. 
        // But for our linalg internal needs, we often need to embed a batch of diagonals.
        // Let's check the logic: if 1D -> 2D. If >1D -> extract? 
        // PyTorch: torch.diag(1D) -> 2D. torch.diag(2D) -> 1D. torch.diag(>2D) -> Error.
        // torch.diag_embed(batch of 1D) -> batch of 2D.
        
        if (ndim == 1) {
            int n = a->shape[0];
            int abs_d = std::abs(diagonal);
            int sz = n + abs_d;
            std::vector<float> out_data(sz * sz, 0.0f);
            for (int i = 0; i < n; ++i) {
                int row = (diagonal >= 0) ? i : i + abs_d;
                int col = (diagonal >= 0) ? i + abs_d : i;
                out_data[row * sz + col] = (*a->data)[i];
            }
            auto out = std::make_shared<Tensor>(out_data, std::vector<int>{sz, sz});
            out->requires_grad = a->requires_grad;
            if (a->requires_grad) {
                out->_prev = {a};
                out->_backward = [a, out, diagonal, n, sz]() {
                    if (!a->grad) a->zero_grad();
                    if (!out->grad) return;
                    out->grad->ensure_cpu_data();
                    int abs_d = std::abs(diagonal);
                    auto new_grad_data = std::vector<float>(n, 0.0f);
                    for (int i = 0; i < n; ++i) {
                        int row = (diagonal >= 0) ? i : i + abs_d;
                        int col = (diagonal >= 0) ? i + abs_d : i;
                        new_grad_data[i] = (*out->grad->data)[row * sz + col];
                    }
                    a->grad = add(a->grad, std::make_shared<Tensor>(new_grad_data, a->shape));
                };
            }
            return out;
        } else if (ndim == 2) {
            int M = a->shape[0], N = a->shape[1], abs_d = std::abs(diagonal);
            int diag_len = (diagonal >= 0) ? std::max(0, std::min(M, N - diagonal)) : std::max(0, std::min(M + diagonal, N));
            std::vector<float> out_data(diag_len);
            for (int i = 0; i < diag_len; ++i) {
                int row = (diagonal >= 0) ? i : i + abs_d;
                int col = (diagonal >= 0) ? i + abs_d : i;
                out_data[i] = (*a->data)[row * N + col];
            }
            auto out = std::make_shared<Tensor>(out_data, std::vector<int>{diag_len});
            out->requires_grad = a->requires_grad;
            if (a->requires_grad) {
                out->_prev = {a};
                out->_backward = [a, out, diagonal, M, N, diag_len]() {
                    if (!a->grad) a->zero_grad();
                    if (!out->grad) return;
                    out->grad->ensure_cpu_data();
                    int abs_d = std::abs(diagonal);
                    std::vector<float> update(M * N, 0.0f);
                    for (int i = 0; i < diag_len; ++i) {
                        int row = (diagonal >= 0) ? i : i + abs_d;
                        int col = (diagonal >= 0) ? i + abs_d : i;
                        update[row * N + col] = (*out->grad->data)[i];
                    }
                    a->grad = add(a->grad, std::make_shared<Tensor>(update, a->shape));
                };
            }
            return out;
        } else {
            // Batch support for diag_embed style (last dim as diagonal)
            // or batch extraction? Let's implement batch diagonal extraction first.
            int n_last = a->shape[ndim-1];
            int n_prev = a->shape[ndim-2];
            if (n_last == n_prev && diagonal == 0) {
                // Extract diagonal
                int mat_size = n_last * n_last;
                int num_batches = a->numel() / mat_size;
                std::vector<float> out_data(num_batches * n_last);
                for (int b = 0; b < num_batches; ++b) {
                    for (int i = 0; i < n_last; ++i) out_data[b * n_last + i] = (*a->data)[b * mat_size + i * n_last + i];
                }
                std::vector<int> out_shape = a->shape; out_shape.pop_back();
                auto out = std::make_shared<Tensor>(out_data, out_shape);
                out->requires_grad = a->requires_grad;
                if (a->requires_grad) {
                    out->_prev = {a};
                    out->_backward = [a, out, n_last, num_batches, mat_size]() {
                        if (!a->grad) a->zero_grad();
                        if (!out->grad) return;
                        out->grad->ensure_cpu_data();
                        std::vector<float> update(a->numel(), 0.0f);
                        for (int b = 0; b < num_batches; ++b)
                            for (int i = 0; i < n_last; ++i) update[b * mat_size + i * n_last + i] = (*out->grad->data)[b * n_last + i];
                        a->grad = add(a->grad, std::make_shared<Tensor>(update, a->shape));
                    };
                }
                return out;
            } else {
                // Embed diagonal [..., N] -> [..., N, N]
                int n = a->shape[ndim-1];
                int num_batches = a->numel() / n;
                std::vector<float> out_data(num_batches * n * n, 0.0f);
                for (int b = 0; b < num_batches; ++b)
                    for (int i = 0; i < n; ++i) out_data[b * n * n + i * n + i] = (*a->data)[b * n + i];
                std::vector<int> out_shape = a->shape; out_shape.push_back(n);
                auto out = std::make_shared<Tensor>(out_data, out_shape);
                out->requires_grad = a->requires_grad;
                if (a->requires_grad) {
                    out->_prev = {a};
                    out->_backward = [a, out, n, num_batches]() {
                        if (!a->grad) a->zero_grad();
                        if (!out->grad) return;
                        out->grad->ensure_cpu_data();
                        std::vector<float> update(a->numel(), 0.0f);
                        for (int b = 0; b < num_batches; ++b)
                            for (int i = 0; i < n; ++i) update[b * n + i] = (*out->grad->data)[b * n * n + i * n + i];
                        a->grad = add(a->grad, std::make_shared<Tensor>(update, a->shape));
                    };
                }
                return out;
            }
        }
    }
    throw std::invalid_argument("diag: invalid tensor shape or diagonal.");
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> eig(const std::shared_ptr<Tensor>& a) {
    check_square(a);
    int ndim = a->shape.size();
    int n = a->shape[ndim-1];
    int matrix_size = n * n;
    int num_batches = a->numel() / matrix_size;
    
    a->ensure_cpu_data();
    auto a_data_copy = *a->data;
    
    auto w_total_data = std::make_shared<std::vector<float>>(num_batches * n);
    auto v_total_data = std::make_shared<std::vector<float>>(num_batches * n * n);
    
    for (int b = 0; b < num_batches; b++) {
        float* a_batch_ptr = a_data_copy.data() + b * matrix_size;
        float* w_batch_ptr = w_total_data->data() + b * n;
        float* v_batch_ptr = v_total_data->data() + b * matrix_size;
        
        char jobz = 'V', uplo = 'L';
        float work_query;
        int lwork = -1, info;
        ssyev_(&jobz, &uplo, &n, a_batch_ptr, &n, w_batch_ptr, &work_query, &lwork, &info);
        lwork = (int)work_query;
        std::vector<float> work(lwork);
        ssyev_(&jobz, &uplo, &n, a_batch_ptr, &n, w_batch_ptr, work.data(), &lwork, &info);
        
        if (info != 0) throw std::runtime_error("LAPACK ssyev failed at batch " + std::to_string(b));
        
        // Transpose eigenvectors from col-major LAPACK output to row-major DLL tensor
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                v_batch_ptr[i * n + j] = a_batch_ptr[j * n + i];
            }
        }
    }
    
    std::vector<int> w_shape = a->shape; w_shape.pop_back(); 
    auto L = std::make_shared<Tensor>(w_total_data, w_shape);
    auto V = std::make_shared<Tensor>(v_total_data, a->shape);
    
    L->requires_grad = a->requires_grad; V->requires_grad = a->requires_grad;
    
    if (a->requires_grad) {
        L->_prev = {a}; V->_prev = {a};
        int n_copy = n;
        auto eig_backward_helper = [a, L, V, n_copy](const std::shared_ptr<Tensor>& dL, const std::shared_ptr<Tensor>& dV) {
            int n = n_copy;
            int num_batches = L->numel() / n;
            std::vector<float> F_data(num_batches * n * n, 0.0f);
            L->ensure_cpu_data();
            for (int b = 0; b < num_batches; ++b) {
                for (int i = 0; i < n; i++) {
                    float li = (*L->data)[b * n + i];
                    for (int j = 0; j < n; j++) {
                        if (i != j) {
                            float lj = (*L->data)[b * n + j];
                            float diff = lj - li;
                            if (std::abs(diff) < 1e-12f) diff = (diff >= 0) ? 1e-12f : -1e-12f;
                            F_data[b * n * n + i * n + j] = 1.0f / diff;
                        }
                    }
                }
            }
            auto F = std::make_shared<Tensor>(F_data, std::vector<int>{num_batches, n, n});
            // Manual diagonal embedding for batches
            std::vector<int> l_mat_shape = L->shape; l_mat_shape.push_back(n);
            std::vector<float> dl_mat_data(num_batches * n * n, 0.0f);
            dL->ensure_cpu_data();
            for (int b = 0; b < num_batches; ++b) for (int i = 0; i < n; i++) dl_mat_data[b * n * n + i * n + i] = (*dL->data)[b * n + i];
            auto dL_matrix = std::make_shared<Tensor>(dl_mat_data, l_mat_shape);
            
            auto Vt = transpose(V);
            auto dA_mid = add(dL_matrix, mul(F, matmul(Vt, dV)));
            auto dA = matmul(V, matmul(dA_mid, Vt));
            if (!a->grad) a->zero_grad();
            a->grad = add(a->grad, dA);
        };
        L->_backward = [a, L, V, eig_backward_helper]() {
            if (!L->grad) return;
            eig_backward_helper(L->grad, std::make_shared<Tensor>(std::vector<float>(V->numel(), 0.0f), V->shape));
        };
        V->_backward = [a, L, V, eig_backward_helper]() {
            if (!V->grad) return;
            eig_backward_helper(std::make_shared<Tensor>(std::vector<float>(L->numel(), 0.0f), L->shape), V->grad);
        };
    }
    return std::make_tuple(L, V);
}


std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> lu(const std::shared_ptr<Tensor>& a) {
    check_square(a);
    int ndim = a->shape.size();
    int n = a->shape[ndim-1];
    int matrix_size = n * n;
    int num_batches = a->numel() / matrix_size;
    
    a->ensure_cpu_data();
    auto data_total = *a->data;
    
    auto L_total_data = std::make_shared<std::vector<float>>(num_batches * n * n);
    auto U_total_data = std::make_shared<std::vector<float>>(num_batches * n * n);
    auto P_total_data = std::make_shared<std::vector<float>>(num_batches * n * n);
    
    for (int b = 0; b < num_batches; ++b) {
        float* batch_ptr = data_total.data() + b * matrix_size;
        
        // LAPACK col-major is A^T for us. But LU(A^T) = (P^T L U)^T = U^T L^T P.
        // To be simple, let's just transpose each batch to col-major, compute LU, then transpose back.
        std::vector<float> data_cm(matrix_size);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                data_cm[j * n + i] = batch_ptr[i * n + j];
                
        std::vector<int> ipiv(n);
        int info;
        sgetrf_(&n, &n, data_cm.data(), &n, ipiv.data(), &info);
        if (info < 0) throw std::invalid_argument("LAPACK sgetrf: invalid argument at batch " + std::to_string(b));
        
        // Transpose back to row-major
        std::vector<float> data_rm(matrix_size);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                data_rm[i * n + j] = data_cm[j * n + i];
        
        // Extract L and U
        float* L_batch_ptr = L_total_data->data() + b * matrix_size;
        float* U_batch_ptr = U_total_data->data() + b * matrix_size;
        for (int i = 0; i < n; i++) {
            L_batch_ptr[i * n + i] = 1.0f;
            for (int j = 0; j < i; j++) L_batch_ptr[i * n + j] = data_rm[i * n + j];
            for (int j = i; j < n; j++) U_batch_ptr[i * n + j] = data_rm[i * n + j];
        }
        
        // Permutation matrix P
        std::vector<std::vector<float>> P_rows(n, std::vector<float>(n, 0.0f));
        for (int i = 0; i < n; i++) P_rows[i][i] = 1.0f;
        for (int i = 0; i < n; i++) {
            int swap_idx = ipiv[i] - 1;
            if (i != swap_idx) std::swap(P_rows[i], P_rows[swap_idx]);
        }
        float* P_batch_ptr = P_total_data->data() + b * matrix_size;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                P_batch_ptr[j * n + i] = P_rows[i][j]; // P_out = P^T
    }
    
    auto P = std::make_shared<Tensor>(P_total_data, a->shape);
    auto L = std::make_shared<Tensor>(L_total_data, a->shape);
    auto U = std::make_shared<Tensor>(U_total_data, a->shape);
    
    P->requires_grad = false;
    L->requires_grad = a->requires_grad;
    U->requires_grad = a->requires_grad;
    
    if (a->requires_grad) {
        L->_prev = {a}; U->_prev = {a};
        auto lu_backward_helper = [a, P, L, U](const std::shared_ptr<Tensor>& dL, const std::shared_ptr<Tensor>& dU) {
            int n = L->shape[L->shape.size()-1];
            int num_batches = L->numel() / (n * n);
            
            auto Lt_dL = matmul(transpose(L), dL);
            auto dU_Ut = matmul(dU, transpose(U));
            Lt_dL->ensure_cpu_data();
            dU_Ut->ensure_cpu_data();
            
            std::vector<float> S_data(num_batches * n * n, 0.0f);
            for (int b = 0; b < num_batches; ++b) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < i; j++) S_data[b * n * n + i * n + j] = (*Lt_dL->data)[b * n * n + i * n + j];
                    for (int j = i; j < n; j++) S_data[b * n * n + i * n + j] = (*dU_Ut->data)[b * n * n + i * n + j];
                }
            }
            auto S = std::make_shared<Tensor>(S_data, L->shape);
            auto dA = matmul(P, matmul(inverse(transpose(L)), matmul(S, inverse(transpose(U)))));
            if (!a->grad) a->zero_grad();
            a->grad = add(a->grad, dA);
        };
        
        L->_backward = [a, P, L, U, lu_backward_helper]() {
            if (!L->grad) return;
            int n = L->shape[L->shape.size()-1];
            int num_batches = L->numel() / (n*n);
            L->grad->ensure_cpu_data();
            auto dL_data = std::vector<float>(L->numel(), 0.0f);
            for (int b = 0; b < num_batches; ++b)
                for (int i = 1; i < n; i++)
                    for (int j = 0; j < i; j++)
                        dL_data[b * n * n + i * n + j] = (*L->grad->data)[b * n * n + i * n + j];
            lu_backward_helper(std::make_shared<Tensor>(dL_data, L->shape), std::make_shared<Tensor>(std::vector<float>(U->numel(), 0.0f), U->shape));
        };
        
        U->_backward = [a, P, L, U, lu_backward_helper]() {
            if (!U->grad) return;
            int n = U->shape[U->shape.size()-1];
            int num_batches = U->numel() / (n*n);
            U->grad->ensure_cpu_data();
            auto dU_data = std::vector<float>(U->numel(), 0.0f);
            for (int b = 0; b < num_batches; ++b)
                for (int i = 0; i < n; i++)
                    for (int j = i; j < n; j++)
                        dU_data[b * n * n + i * n + j] = (*U->grad->data)[b * n * n + i * n + j];
            lu_backward_helper(std::make_shared<Tensor>(std::vector<float>(L->numel(), 0.0f), L->shape), std::make_shared<Tensor>(dU_data, U->shape));
        };
    }
    return std::make_tuple(P, L, U);
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> qr(const std::shared_ptr<Tensor>& a) {
    int ndim = a->shape.size();
    if (ndim < 2) {
        throw std::invalid_argument("QR decomposition requires at least a 2D matrix.");
    }
    int M = a->shape[ndim-2];
    int N = a->shape[ndim-1];
    int K = std::min(M, N);
    int matrix_size = M * N;
    int num_batches = a->numel() / matrix_size;
    
    a->ensure_cpu_data();
    auto a_data_copy = *a->data;
    
    auto q_total_data = std::make_shared<std::vector<float>>(num_batches * M * K);
    auto r_total_data = std::make_shared<std::vector<float>>(num_batches * K * N);
    
    for (int b = 0; b < num_batches; ++b) {
        float* a_batch_ptr = a_data_copy.data() + b * matrix_size;
        
        std::vector<float> cm_data(M * N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                cm_data[i + j * M] = a_batch_ptr[i * N + j];
                
        std::vector<float> tau(K);
        float work_query;
        int lwork = -1, info;
        
        sgeqrf_(&M, &N, cm_data.data(), &M, tau.data(), &work_query, &lwork, &info);
        lwork = (int)work_query;
        std::vector<float> work(lwork);
        sgeqrf_(&M, &N, cm_data.data(), &M, tau.data(), work.data(), &lwork, &info);
        if (info != 0) throw std::runtime_error("LAPACK sgeqrf failed at batch " + std::to_string(b));
        
        float* r_batch_ptr = r_total_data->data() + b * K * N;
        for (int i = 0; i < K; i++)
            for (int j = i; j < N; j++)
                r_batch_ptr[i * N + j] = cm_data[i + j * M];
                
        lwork = -1;
        sorgqr_(&M, &K, &K, cm_data.data(), &M, tau.data(), &work_query, &lwork, &info);
        if (info != 0) throw std::runtime_error("LAPACK sorgqr query failed at batch " + std::to_string(b));
        lwork = (int)work_query;
        work.resize(lwork);
        sorgqr_(&M, &K, &K, cm_data.data(), &M, tau.data(), work.data(), &lwork, &info);
        if (info != 0) throw std::runtime_error("LAPACK sorgqr failed at batch " + std::to_string(b));
        
        float* q_batch_ptr = q_total_data->data() + b * M * K;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < K; j++)
                q_batch_ptr[i * K + j] = cm_data[i + j * M];
    }
    
    std::vector<int> q_shape = a->shape; q_shape[ndim-1] = K;
    std::vector<int> r_shape = a->shape; r_shape[ndim-2] = K;
    
    auto Q = std::make_shared<Tensor>(q_total_data, q_shape);
    auto R = std::make_shared<Tensor>(r_total_data, r_shape);
    
    Q->requires_grad = a->requires_grad; R->requires_grad = a->requires_grad;
    
    if (a->requires_grad && M >= N) {
        Q->_prev = {a}; R->_prev = {a};
        auto qr_backward_helper = [a, Q, R](const std::shared_ptr<Tensor>& dQ, const std::shared_ptr<Tensor>& dR) {
            int M = Q->shape[Q->shape.size()-2];
            int N = R->shape[R->shape.size()-1]; // Here K = N
            int num_batches = Q->numel() / (M * N);
            
            auto Qt = transpose(Q);
            auto R_dRt = matmul(R, transpose(dR));
            auto dQtQ = matmul(transpose(dQ), Q);
            auto M_mat = sub(R_dRt, dQtQ);
            M_mat->ensure_cpu_data();
            
            std::vector<float> M_sym_data(num_batches * N * N, 0.0f);
            for (int b = 0; b < num_batches; ++b) {
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j <= i; j++) {
                        float val = (*M_mat->data)[b * N * N + i * N + j];
                        M_sym_data[b * N * N + i * N + j] = val;
                        M_sym_data[b * N * N + j * N + i] = val;
                    }
                }
            }
            auto M_sym = std::make_shared<Tensor>(M_sym_data, std::vector<int>{num_batches, N, N}); // Temporarily 3D
            auto R_inv_T = transpose(inverse(R));
            auto dA = matmul(add(dQ, matmul(Q, M_sym)), R_inv_T);
            
            if (M > N) {
                std::vector<float> I_data(num_batches * M * M, 0.0f);
                for (int b = 0; b < num_batches; ++b) for (int i = 0; i < M; i++) I_data[b * M * M + i * M + i] = 1.0f;
                auto I_mat = std::make_shared<Tensor>(I_data, std::vector<int>{num_batches, M, M});
                auto proj = sub(I_mat, matmul(Q, Qt));
                dA = add(dA, matmul(matmul(proj, dQ), R_inv_T));
            }
            if (!a->grad) a->zero_grad();
            a->grad = add(a->grad, dA);
        };
        
        Q->_backward = [a, Q, R, qr_backward_helper]() {
            if (!Q->grad) return;
            qr_backward_helper(Q->grad, std::make_shared<Tensor>(std::vector<float>(R->numel(), 0.0f), R->shape));
        };
        
        R->_backward = [a, Q, R, qr_backward_helper]() {
            if (!R->grad) return;
            int N = R->shape[R->shape.size()-1];
            int num_batches = R->numel() / (N * N);
            R->grad->ensure_cpu_data();
            auto dR_data = std::vector<float>(R->numel(), 0.0f);
            for (int b = 0; b < num_batches; ++b)
                for (int i = 0; i < N; i++)
                    for (int j = i; j < N; j++)
                        dR_data[b * N * N + i * N + j] = (*R->grad->data)[b * N * N + i * N + j];
            qr_backward_helper(std::make_shared<Tensor>(std::vector<float>(Q->numel(), 0.0f), Q->shape), std::make_shared<Tensor>(dR_data, R->shape));
        };
    }
    return std::make_tuple(Q, R);
}

std::shared_ptr<Tensor> matrix_exp(const std::shared_ptr<Tensor>& a) {
    if (a->shape.size() < 2 || a->shape[a->shape.size()-1] != a->shape[a->shape.size()-2]) {
        throw std::invalid_argument("matrix_exp requires square matrices.");
    }
    
    auto forward_impl = [](const std::shared_ptr<Tensor>& X) {
        int ndim = X->shape.size();
        int N = X->shape[ndim-1];
        int matrix_size = N * N;
        int num_batches = X->numel() / matrix_size;
        
        // Compute scaling factor s
        float max_norm = 0.0f;
        X->ensure_cpu_data();
        for (float val : *X->data) if (std::abs(val) > max_norm) max_norm = std::abs(val);
        
        int s = 0;
        if (max_norm > 0.5f) {
            s = (int)std::ceil(std::log2(max_norm / 0.5f));
        }
        if (s < 0) s = 0;
        
        auto A = (s > 0) ? mul_scalar(X, 1.0f / (float)std::pow(2.0, s)) : X;
        
        // Padé m=6 coefficients
        float c0 = 1.0f;
        float c1 = 0.5f;
        float c2 = 15.0f / 132.0f;
        float c3 = 1.0f / 66.0f;
        float c4 = 1.0f / 990.0f;
        float c5 = 1.0f / 27720.0f;
        float c6 = 1.0f / 1316520.0f;
        
        std::vector<float> I_data(num_batches * N * N, 0.0f);
        for(int b=0; b<num_batches; ++b) for(int i=0; i<N; ++i) I_data[b*N*N + i*N + i] = 1.0f;
        auto I = std::make_shared<Tensor>(I_data, X->shape);
        
        auto A2 = matmul(A, A);
        auto A4 = matmul(A2, A2);
        auto A6 = matmul(A4, A2);
        
        // U = c0 I + c2 A2 + c4 A4 + c6 A6
        auto U = add(add(add(mul_scalar(I, c0), mul_scalar(A2, c2)), mul_scalar(A4, c4)), mul_scalar(A6, c6));
        
        // V = c1 A + c3 A3 + c5 A5
        auto A3 = matmul(A2, A);
        auto A5 = matmul(A4, A);
        auto V = add(add(mul_scalar(A, c1), mul_scalar(A3, c3)), mul_scalar(A5, c5));
        
        auto P = add(U, V);
        auto Q = sub(U, V);
        
        auto res = solve(Q, P);
        
        for (int i = 0; i < s; ++i) {
            res = matmul(res, res);
        }
        return res;
    };
    
    auto out = forward_impl(a);
    
    if (a->requires_grad) {
        out->_prev = {a};
        out->_backward = [a, out, forward_impl]() {
            if (!out->grad) return;
            
            // Backward using the block matrix trick
            // H = [[A.T, grad.T], [0, A.T]]
            int ndim = a->shape.size();
            int N = a->shape[ndim-1];
            int num_batches = a->numel() / (N * N);
            
            a->ensure_cpu_data();
            out->grad->ensure_cpu_data();
            
            std::vector<float> h_data(num_batches * (2*N) * (2*N), 0.0f);
            for (int b = 0; b < num_batches; ++b) {
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < N; ++j) {
                        float val_at = (*a->data)[b * N * N + j * N + i]; // A.T
                        float val_g = (*out->grad->data)[b * N * N + i * N + j]; // G
                        
                        h_data[b * (2*N) * (2*N) + i * (2*N) + j] = val_at;
                        h_data[b * (2*N) * (2*N) + i * (2*N) + (j+N)] = val_g;
                        h_data[b * (2*N) * (2*N) + (i+N) * (2*N) + (j+N)] = val_at;
                    }
                }
            }
            
            std::vector<int> h_shape = a->shape;
            h_shape[ndim-1] = 2 * N;
            h_shape[ndim-2] = 2 * N;
            auto H = std::make_shared<Tensor>(h_data, h_shape);
            
            auto expH = forward_impl(H);
            expH->ensure_cpu_data();
            
            std::vector<float> grad_a_data(a->numel());
            for (int b = 0; b < num_batches; ++b) {
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < N; ++j) {
                        // Top-right block is d(exp(A^T))[G] = grad_A
                        grad_a_data[b * N * N + i * N + j] = (*expH->data)[b * (2*N) * (2*N) + i * (2*N) + (j+N)];
                    }
                }
            }
            
            if (!a->grad) a->zero_grad();
            a->grad = add(a->grad, std::make_shared<Tensor>(grad_a_data, a->shape));
        };
    }
    
    return out;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> lu_factor(const std::shared_ptr<Tensor>& a) {
    check_square(a);
    int ndim = a->shape.size();
    int n = a->shape[ndim-1];
    int matrix_size = n * n;
    int num_batches = a->numel() / matrix_size;
    
    a->ensure_cpu_data();
    auto lu_data = std::make_shared<std::vector<float>>(*a->data);
    auto pivots_data = std::make_shared<std::vector<float>>(num_batches * n);
    
    for (int b = 0; b < num_batches; ++b) {
        float* batch_ptr = lu_data->data() + b * matrix_size;
        float* p_ptr = pivots_data->data() + b * n;
        
        std::vector<float> data_cm(matrix_size);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                data_cm[j * n + i] = batch_ptr[i * n + j];
                
        std::vector<int> ipiv(n);
        int info;
        sgetrf_(&n, &n, data_cm.data(), &n, ipiv.data(), &info);
        if (info < 0) throw std::invalid_argument("LAPACK sgetrf failed.");
        
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                batch_ptr[i * n + j] = data_cm[j * n + i];
        
        for (int i = 0; i < n; i++) p_ptr[i] = (float)ipiv[i];
    }
    
    auto LU = std::make_shared<Tensor>(lu_data, a->shape);
    auto pivots = std::make_shared<Tensor>(pivots_data, std::vector<int>{num_batches, n});
    
    LU->requires_grad = a->requires_grad;
    pivots->requires_grad = false;
    
    if (a->requires_grad) {
        LU->_prev = {a};
        LU->_backward = [a, LU, pivots]() {
            if (!a->grad) a->zero_grad();
            if (!LU->grad) return;
            
            int n = LU->shape[LU->shape.size()-1];
            int num_batches = LU->numel() / (n * n);
            
            // Reconstitute dA from dL and dU
            std::vector<float> l_data(LU->numel()), u_data(LU->numel());
            std::vector<float> dl_data(LU->numel()), du_data(LU->numel());
            LU->ensure_cpu_data(); LU->grad->ensure_cpu_data();
            for (int b = 0; b < num_batches; b++) {
                for (int i = 0; i < n; i++) {
                    l_data[b * n * n + i * n + i] = 1.0f;
                    for (int j = 0; j < i; j++) {
                        l_data[b * n * n + i * n + j] = (*LU->data)[b * n * n + i * n + j];
                        dl_data[b * n * n + i * n + j] = (*LU->grad->data)[b * n * n + i * n + j];
                    }
                    for (int j = i; j < n; j++) {
                        u_data[b * n * n + i * n + j] = (*LU->data)[b * n * n + i * n + j];
                        du_data[b * n * n + i * n + j] = (*LU->grad->data)[b * n * n + i * n + j];
                    }
                }
            }
            auto L = std::make_shared<Tensor>(l_data, LU->shape);
            auto U = std::make_shared<Tensor>(u_data, LU->shape);
            auto dL = std::make_shared<Tensor>(dl_data, LU->shape);
            auto dU = std::make_shared<Tensor>(du_data, LU->shape);
            
            // P from pivots
            std::vector<float> p_data(LU->numel());
            pivots->ensure_cpu_data();
            for (int b = 0; b < num_batches; b++) {
                std::vector<int> ipiv(n);
                for (int i = 0; i < n; i++) ipiv[i] = (int)(*pivots->data)[b * n + i];
                std::vector<float> P_batch(n * n, 0.0f);
                for (int i = 0; i < n; i++) P_batch[i * n + i] = 1.0f;
                for (int i = 0; i < n; i++) {
                    int swap = ipiv[i] - 1;
                    if (i != swap) {
                        for (int j = 0; j < n; j++) std::swap(P_batch[i * n + j], P_batch[swap * n + j]);
                    }
                }
                for (int i = 0; i < n * n; i++) p_data[b * n * n + i] = P_batch[i];
            }
            auto P = std::make_shared<Tensor>(p_data, LU->shape);
            
            auto Lt_dL = matmul(transpose(L), dL);
            auto dU_Ut = matmul(dU, transpose(U));
            Lt_dL->ensure_cpu_data(); dU_Ut->ensure_cpu_data();
            std::vector<float> S_data(num_batches * n * n);
            for (int b = 0; b < num_batches; b++) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < i; j++) S_data[b * n * n + i * n + j] = (*Lt_dL->data)[b * n * n + i * n + j];
                    for (int j = i; j < n; j++) S_data[b * n * n + i * n + j] = (*dU_Ut->data)[b * n * n + i * n + j];
                }
            }
            auto S = std::make_shared<Tensor>(S_data, LU->shape);
            auto dA = matmul(transpose(P), matmul(inverse(transpose(L)), matmul(S, inverse(transpose(U)))));
            a->grad = add(a->grad, dA);
        };
    }
    return std::make_tuple(LU, pivots);
}

std::shared_ptr<Tensor> lu_solve(const std::shared_ptr<Tensor>& b, const std::shared_ptr<Tensor>& lu_data, const std::shared_ptr<Tensor>& pivots, bool adjoint) {
    int ndim_a = lu_data->shape.size();
    int n = lu_data->shape[ndim_a - 1];
    int ndim_b = b->shape.size();
    int nrhs = b->shape[ndim_b - 1];
    int num_batches = lu_data->numel() / (n * n);
    
    b->ensure_cpu_data(); lu_data->ensure_cpu_data(); pivots->ensure_cpu_data();
    auto out_data = std::make_shared<std::vector<float>>(*b->data);
    
    for (int b_idx = 0; b_idx < num_batches; ++b_idx) {
        float* lu_batch = lu_data->data->data() + b_idx * n * n;
        float* b_batch = out_data->data() + b_idx * n * nrhs;
        float* p_ptr = pivots->data->data() + b_idx * n;
        
        std::vector<int> ipiv(n);
        for (int i = 0; i < n; i++) ipiv[i] = (int)p_ptr[i];
        
        std::vector<float> lu_cm(n * n);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) lu_cm[j * n + i] = lu_batch[i * n + j];
        std::vector<float> b_cm(n * nrhs);
        for (int i = 0; i < n; i++) for (int j = 0; j < nrhs; j++) b_cm[j * n + i] = b_batch[i * nrhs + j];
        
        char trans = adjoint ? 'T' : 'N';
        int info;
        sgetrs_(&trans, &n, &nrhs, lu_cm.data(), &n, ipiv.data(), b_cm.data(), &n, &info);
        for (int i = 0; i < n; i++) for (int j = 0; j < nrhs; j++) b_batch[i * nrhs + j] = b_cm[j * n + i];
    }
    
    auto out = std::make_shared<Tensor>(out_data, b->shape);
    out->requires_grad = (b->requires_grad || lu_data->requires_grad);
    
    if (out->requires_grad) {
        out->_prev = {b, lu_data, pivots};
        out->_backward = [b, lu_data, pivots, out, adjoint, ndim_a, n, num_batches]() {
            if (!out->grad) return;
            // grad_b = solve(A^T, grad_out) if !adjoint, else solve(A, grad_out)
            auto grad_b = lu_solve(out->grad, lu_data, pivots, !adjoint);
            if (b->requires_grad) {
                if (!b->grad) b->zero_grad();
                b->grad = add(b->grad, grad_b);
            }
            
            if (lu_data->requires_grad) {
                // To compute dLU correctly for lu_factor, we need the gradient w.r.t
                // the LU product BEFORE row permutation P.
                // X = (LU)^-1 P B => dX = -(LU)^-1 d(LU) X
                // grad_LU_prod = - ((LU)^-T grad_X) X^T
                
                // Solve (LU)^T Y = out->grad WITHOUT pivots
                std::vector<float> id_piv_data(num_batches * n);
                for (int b = 0; b < num_batches; b++) for (int i = 0; i < n; i++) id_piv_data[b * n + i] = (float)(i + 1);
                auto id_pivots = std::make_shared<Tensor>(std::make_shared<std::vector<float>>(id_piv_data), pivots->shape);
                
                auto grad_LU_solve = lu_solve(out->grad, lu_data, id_pivots, !adjoint);
                auto X = out;
                std::shared_ptr<Tensor> dA_perm;
                if (!adjoint) dA_perm = mul_scalar(matmul(grad_LU_solve, transpose(X)), -1.0f);
                else dA_perm = mul_scalar(matmul(X, transpose(grad_LU_solve)), -1.0f);
                
                // Map dA_perm to dLU (unit diag L)
                std::vector<float> l_data(lu_data->numel(), 0.0f);
                std::vector<float> u_data(lu_data->numel(), 0.0f);
                for (int b = 0; b < num_batches; b++) {
                    for (int i = 0; i < n; i++) {
                        l_data[b * n * n + i * n + i] = 1.0f;
                        for (int j = 0; j < i; j++) l_data[b * n * n + i * n + j] = (*lu_data->data)[b * n * n + i * n + j];
                        for (int j = i; j < n; j++) u_data[b * n * n + i * n + j] = (*lu_data->data)[b * n * n + i * n + j];
                    }
                }
                auto L = std::make_shared<Tensor>(l_data, lu_data->shape);
                auto U = std::make_shared<Tensor>(u_data, lu_data->shape);
                
                auto dL = matmul(dA_perm, transpose(U));
                auto dU = matmul(transpose(L), dA_perm);
                dL->ensure_cpu_data(); dU->ensure_cpu_data();
                
                std::vector<float> dlu_data(lu_data->numel());
                for (int b = 0; b < num_batches; b++) {
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < i; j++) dlu_data[b * n * n + i * n + j] = (*dL->data)[b * n * n + i * n + j];
                        for (int j = i; j < n; j++) dlu_data[b * n * n + i * n + j] = (*dU->data)[b * n * n + i * n + j];
                    }
                }
                if (!lu_data->grad) lu_data->zero_grad();
                lu_data->grad = add(lu_data->grad, std::make_shared<Tensor>(dlu_data, lu_data->shape));
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> cholesky_solve(const std::shared_ptr<Tensor>& b, const std::shared_ptr<Tensor>& l) {
    int ndim = l->shape.size();
    int n = l->shape[ndim-1];
    int nrhs = b->shape[b->shape.size()-1];
    int num_batches = l->numel() / (n * n);
    
    b->ensure_cpu_data(); l->ensure_cpu_data();
    auto out_data = std::make_shared<std::vector<float>>(*b->data);
    
    for (int b_idx = 0; b_idx < num_batches; ++b_idx) {
        float* l_batch = l->data->data() + b_idx * n * n;
        float* b_batch = out_data->data() + b_idx * n * nrhs;
        
        std::vector<float> l_cm(n * n);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) l_cm[j * n + i] = l_batch[i * n + j];
        std::vector<float> b_cm(n * nrhs);
        for (int i = 0; i < n; i++) for (int j = 0; j < nrhs; j++) b_cm[j * n + i] = b_batch[i * nrhs + j];
        
        char uplo = 'L'; // We expect lower triangular L
        int info;
        spotrs_(&uplo, &n, &nrhs, l_cm.data(), &n, b_cm.data(), &n, &info);
        for (int i = 0; i < n; i++) for (int j = 0; j < nrhs; j++) b_batch[i * nrhs + j] = b_cm[j * n + i];
    }
    
    auto out = std::make_shared<Tensor>(out_data, b->shape);
    out->requires_grad = (b->requires_grad || l->requires_grad);
    
    if (out->requires_grad) {
        out->_prev = {b, l};
        out->_backward = [b, l, out]() {
            if (!out->grad) return;
            auto grad_b = cholesky_solve(out->grad, l);
            if (b->requires_grad) {
                if (!b->grad) b->zero_grad();
                b->grad = add(b->grad, grad_b);
            }
            if (l->requires_grad) {
                // dA = - grad_b @ X^T => dL = -(grad_b @ X^T + X @ grad_b^T) @ L
                auto X = out;
                auto grad_a = mul_scalar(add(matmul(grad_b, transpose(X)), matmul(X, transpose(grad_b))), -1.0f);
                auto grad_l = matmul(grad_a, l);
                if (!l->grad) l->zero_grad();
                l->grad = add(l->grad, grad_l);
            }
        };
    }
    return out;
}

std::shared_ptr<Tensor> lstsq(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    int ndim_a = a->shape.size(), ndim_b = b->shape.size();
    int M = a->shape[ndim_a-2], N = a->shape[ndim_a-1];
    int nrhs = b->shape[ndim_b-1];
    int num_batches = a->numel() / (M * N);
    
    a->ensure_cpu_data(); b->ensure_cpu_data();
    int ldb_max = std::max(M, N);
    auto out_data = std::make_shared<std::vector<float>>(num_batches * ldb_max * nrhs, 0.0f);
    
    for (int b_idx = 0; b_idx < num_batches; ++b_idx) {
        float* a_batch = a->data->data() + b_idx * M * N;
        float* b_batch = b->data->data() + b_idx * M * nrhs;
        
        std::vector<float> a_cm(M * N);
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) a_cm[j * M + i] = a_batch[i * N + j];
        std::vector<float> b_cm(ldb_max * nrhs, 0.0f);
        for (int i = 0; i < M; i++) for (int j = 0; j < nrhs; j++) b_cm[j * ldb_max + i] = b_batch[i * nrhs + j];
        
        char trans = 'N'; int info;
        float work_query; int lwork = -1;
        sgels_(&trans, &M, &N, &nrhs, a_cm.data(), &M, b_cm.data(), &ldb_max, &work_query, &lwork, &info);
        lwork = (int)work_query; std::vector<float> work(lwork);
        sgels_(&trans, &M, &N, &nrhs, a_cm.data(), &M, b_cm.data(), &ldb_max, work.data(), &lwork, &info);
        
        float* out_batch = out_data->data() + b_idx * ldb_max * nrhs;
        for (int i = 0; i < ldb_max; i++) for (int j = 0; j < nrhs; j++) out_batch[i * nrhs + j] = b_cm[j * ldb_max + i];
    }
    
    std::vector<int> out_shape = b->shape; out_shape[ndim_b-2] = N;
    auto x_data = std::make_shared<std::vector<float>>(num_batches * N * nrhs);
    for (int b_idx = 0; b_idx < num_batches; b_idx++) {
        std::memcpy(x_data->data() + b_idx * N * nrhs, out_data->data() + b_idx * ldb_max * nrhs, N * nrhs * sizeof(float));
    }
    auto out = std::make_shared<Tensor>(x_data, out_shape);
    out->requires_grad = (a->requires_grad || b->requires_grad);
    
    if (out->requires_grad) {
        out->_prev = {a, b};
        out->_backward = [a, b, out]() {
            if (!out->grad) return;
            // Differentiable least squares (over-determined case M >= N)
            // grad_b = A^+T grad_X
            // A^+ = (A^T A)^-1 A^T
            // We can compute A^+T grad_X by solving for min ||A^T Y - grad_X|| if M=N
            // Generally, A^+T = A (A^T A)^-1.
            auto AtA = matmul(transpose(a), a);
            auto grad_b = matmul(a, solve(AtA, out->grad));
            if (b->requires_grad) {
                if (!b->grad) b->zero_grad();
                b->grad = add(b->grad, grad_b);
            }
            if (a->requires_grad) {
                // Proper dA for M >= N:
                // dA = - grad_b @ X^T + resid @ (grad_X @ (A^T @ A)^-1)^T ? 
                // Actually: dA = - grad_b @ X^T + (B - AX) @ (grad_X^T @ (A^T A)^-1)
                auto AtA_inv = inverse(AtA);
                auto resid = sub(b, matmul(a, out));
                auto term2 = matmul(resid, matmul(transpose(out->grad), AtA_inv));
                auto dA = add(mul_scalar(matmul(grad_b, transpose(out)), -1.0f), term2);
                
                if (!a->grad) a->zero_grad();
                a->grad = add(a->grad, dA);
            }
        };
    }
    return out;
}
