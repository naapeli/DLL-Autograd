#pragma once
#include "tensor.h"
#include <memory>

/**
 * @brief Computes the inverse of a square matrix.
 */
std::shared_ptr<Tensor> inverse(const std::shared_ptr<Tensor>& a);

/**
 * @brief Computes the determinant of a square matrix.
 */
std::shared_ptr<Tensor> determinant(const std::shared_ptr<Tensor>& a);

/**
 * @brief Solves a linear system AX = B.
 */
std::shared_ptr<Tensor> solve(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

/**
 * @brief Computes the Cholesky decomposition of a symmetric positive-definite matrix.
 */
std::shared_ptr<Tensor> cholesky(const std::shared_ptr<Tensor>& a);

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> svd(const std::shared_ptr<Tensor>& a, bool full_matrices = true);

/**
 * @brief If input is 1D, returns a 2D diagonal matrix. If input is 2D, extracts the diagonal.
 */
std::shared_ptr<Tensor> diag(const std::shared_ptr<Tensor>& a, int diagonal = 0);

/**
 * @brief Computes eigenvalues and eigenvectors of a symmetric matrix.
 * Returns (eigenvalues, eigenvectors) where eigenvectors are columns of V.
 */
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> eig(const std::shared_ptr<Tensor>& a);

/**
 * @brief Computes the LU decomposition with partial pivoting.
 * Returns (P, L, U) such that A = P @ L @ U.
 */
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> lu(const std::shared_ptr<Tensor>& a);

/**
 * @brief Computes the reduced QR decomposition. Returns (Q, R) such that A = Q @ R.
 */
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> qr(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> matrix_exp(const std::shared_ptr<Tensor>& a);

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> lu_factor(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> lu_solve(const std::shared_ptr<Tensor>& b, const std::shared_ptr<Tensor>& lu_data, const std::shared_ptr<Tensor>& pivots, bool adjoint = false);
std::shared_ptr<Tensor> cholesky_solve(const std::shared_ptr<Tensor>& b, const std::shared_ptr<Tensor>& l);
std::shared_ptr<Tensor> lstsq(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
