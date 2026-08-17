#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Matrix {
    std::size_t rows = 0, cols = 0;
    std::vector<double> data;
    Matrix() = default;
    Matrix(std::size_t r, std::size_t c, double value = 0.0)
        : rows(r), cols(c), data(r * c, value) {}
    double &operator()(std::size_t r, std::size_t c) { return data[r * cols + c]; }
    double operator()(std::size_t r, std::size_t c) const { return data[r * cols + c]; }
};

static double parse_finite_double(const std::string &text, const std::string &description) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    while (consumed < text.size() && std::isspace(static_cast<unsigned char>(text[consumed]))) {
        ++consumed;
    }
    if (consumed != text.size() || !std::isfinite(value)) {
        throw std::runtime_error("Invalid finite number for " + description + ": " + text);
    }
    return value;
}

static Matrix read_csv(const std::string &path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("Cannot open " + path);
    std::vector<double> values;
    std::size_t rows = 0, cols = 0;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::string cell;
        std::size_t current_cols = 0;
        while (std::getline(ss, cell, ',')) {
            if (cell.empty())
                throw std::runtime_error("Empty CSV value in " + path);
            values.push_back(parse_finite_double(cell, "CSV value in " + path));
            ++current_cols;
        }
        if (cols == 0)
            cols = current_cols;
        else if (current_cols != cols)
            throw std::runtime_error("Inconsistent CSV columns in " + path);
        ++rows;
    }
    if (rows == 0 || cols == 0)
        throw std::runtime_error("Empty CSV file: " + path);
    Matrix result(rows, cols);
    result.data = std::move(values);
    return result;
}

static void write_csv(const std::string &path, const Matrix &matrix) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error("Cannot write " + path);
    output << std::setprecision(17);
    for (std::size_t i = 0; i < matrix.rows; ++i) {
        for (std::size_t j = 0; j < matrix.cols; ++j) {
            if (j)
                output << ',';
            output << matrix(i, j);
        }
        output << '\n';
    }
    if (!output)
        throw std::runtime_error("Failed while writing " + path);
}

class ExplainerCore {
  public:
    ExplainerCore(const Matrix &X, const Matrix &Y, const Matrix &P, const Matrix &Q,
                  const Matrix &sigma)
        : X_(X), Y_(Y), P_(P), Q_(Q), n_(X.rows), d_(X.cols), m_(Y.cols), sigma_(n_) {
        if (n_ < 2 || d_ == 0)
            throw std::runtime_error("X must contain at least two samples and one feature");
        if (Y.rows != n_ || P.rows != n_ || P.cols != n_ || Q.rows != n_ || Q.cols != n_)
            throw std::runtime_error("X, Y, P and Q shapes are inconsistent");
        if (m_ != 2)
            throw std::runtime_error(
                "This optimized explainer currently supports a 2D embedding only");
        if (!((sigma.rows == n_ && sigma.cols == 1) || (sigma.rows == 1 && sigma.cols == n_)))
            throw std::runtime_error("sigma must contain one value per sample");
        for (std::size_t i = 0; i < n_; ++i) {
            sigma_[i] = sigma.rows == n_ ? sigma(i, 0) : sigma(0, i);
            if (!(sigma_[i] > 0.0) || !std::isfinite(sigma_[i]))
                throw std::runtime_error("sigma values must be finite and positive");
        }
        precompute();
    }

    Matrix compute_all() const {
        // Flattened as N rows and (2*D) columns: row i = gradient[0,:], gradient[1,:].
        Matrix out(n_, 2 * d_, 0.0);
        std::atomic<bool> failed{false};
        std::string error_message;
#pragma omp parallel for schedule(dynamic) if (n_ >= 64)
        for (long long ii = 0; ii < static_cast<long long>(n_); ++ii) {
            if (failed.load(std::memory_order_relaxed))
                continue;
            try {
                compute_one(static_cast<std::size_t>(ii), out);
            } catch (const std::exception &error) {
#pragma omp critical
                {
                    if (!failed.exchange(true))
                        error_message = error.what();
                }
            }
        }
        if (failed)
            throw std::runtime_error(error_message);
        return out;
    }

  private:
    const Matrix &X_;
    const Matrix &Y_;
    const Matrix &P_;
    const Matrix &Q_;
    std::size_t n_, d_, m_;
    std::vector<double> sigma_;
    Matrix xdist2_, ydist2_;
    std::vector<double> S_pj_;
    double S_q_ = 0.0;

    void precompute() {
        xdist2_ = Matrix(n_, n_, 0.0);
        ydist2_ = Matrix(n_, n_, 0.0);
#pragma omp parallel for schedule(static) if (n_ >= 64)
        for (long long ii = 0; ii < static_cast<long long>(n_); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            for (std::size_t j = i + 1; j < n_; ++j) {
                double dx2 = 0.0, dy2 = 0.0;
                for (std::size_t k = 0; k < d_; ++k) {
                    double z = X_(i, k) - X_(j, k);
                    dx2 += z * z;
                }
                for (std::size_t k = 0; k < m_; ++k) {
                    double z = Y_(i, k) - Y_(j, k);
                    dy2 += z * z;
                }
                xdist2_(i, j) = xdist2_(j, i) = dx2;
                ydist2_(i, j) = ydist2_(j, i) = dy2;
            }
        }
        S_q_ = 0.0;
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t j = 0; j < n_; ++j)
                if (i != j)
                    S_q_ += 1.0 / (1.0 + ydist2_(i, j));
        if (!(S_q_ > 0.0) || !std::isfinite(S_q_))
            throw std::runtime_error("Cannot normalize low-dimensional affinities");

        // Mirrors Python: sum(axis=0) of exp(-D/(2*sigma**2)), where sigma broadcasts over columns.
        S_pj_.assign(n_, -1.0);
#pragma omp parallel for schedule(static) if (n_ >= 64)
        for (long long jj = 0; jj < static_cast<long long>(n_); ++jj) {
            const std::size_t j = static_cast<std::size_t>(jj);
            double sum = 0.0;
            const double denom = 2.0 * sigma_[j] * sigma_[j];
            for (std::size_t l = 0; l < n_; ++l)
                sum += std::exp(-xdist2_(l, j) / denom);
            S_pj_[j] += sum;
        }
    }

    void compute_one(std::size_t i, Matrix &out) const {
        // Hessian d2C/dy_i^2 (2x2)
        double term1[2][2] = {{0, 0}, {0, 0}}, term2[2][2] = {{0, 0}, {0, 0}},
               term3[2][2] = {{0, 0}, {0, 0}};
        double Sq_d[2] = {0, 0};
        for (std::size_t j = 0; j < n_; ++j) {
            const double dy0 = Y_(i, 0) - Y_(j, 0), dy1 = Y_(i, 1) - Y_(j, 1);
            const double e = 1.0 + ydist2_(i, j), inv_e2 = 1.0 / (e * e);
            Sq_d[0] += inv_e2 * dy0;
            Sq_d[1] += inv_e2 * dy1;
        }
        Sq_d[0] *= -4.0;
        Sq_d[1] *= -4.0;

        for (std::size_t j = 0; j < n_; ++j) {
            const double dy[2] = {Y_(i, 0) - Y_(j, 0), Y_(i, 1) - Y_(j, 1)};
            const double e = 1.0 + ydist2_(i, j), E = 1.0 / e, E2 = E * E;
            const double v = P_(i, j) - Q_(i, j);
            double vd[2];
            for (int a = 0; a < 2; ++a)
                vd[a] = ((2.0 * dy[a] * E2 * S_q_) + (Sq_d[a] * E)) / (S_q_ * S_q_);
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b) {
                    term1[a][b] += vd[a] * E * dy[b];
                    term2[a][b] += (2.0 * dy[a]) * (v * E2) * dy[b];
                }
            if (j != i) {
                term3[0][0] += v * E;
                term3[1][1] += v * E;
            }
        }
        double H00 = 4.0 * (term1[0][0] - term2[0][0] + term3[0][0]);
        double H01 = 4.0 * (term1[0][1] - term2[0][1] + term3[0][1]);
        double H10 = 4.0 * (term1[1][0] - term2[1][0] + term3[1][0]);
        double H11 = 4.0 * (term1[1][1] - term2[1][1] + term3[1][1]);

        // xy derivative has shape D x 2.
        std::vector<double> B0(d_, 0.0), B1(d_, 0.0);
        std::vector<double> exp_ij(n_), exp_ji(n_);
        double S_pi = 0.0;
        const double si2 = sigma_[i] * sigma_[i];
        for (std::size_t j = 0; j < n_; ++j) {
            exp_ij[j] = std::exp(-xdist2_(i, j) / (2.0 * si2));
            exp_ji[j] = std::exp(-xdist2_(i, j) / (2.0 * sigma_[j] * sigma_[j]));
            if (j != i)
                S_pi += exp_ij[j];
        }
        if (!(S_pi > 0.0) || !std::isfinite(S_pi))
            throw std::runtime_error("Degenerate high-dimensional affinities at sample " +
                                     std::to_string(i));
        std::vector<double> Spi_d(d_, 0.0);
        for (std::size_t j = 0; j < n_; ++j)
            for (std::size_t k = 0; k < d_; ++k)
                Spi_d[k] += -(X_(i, k) - X_(j, k)) / si2 * exp_ij[j];

        for (std::size_t j = 0; j < n_; ++j) {
            const double E = 1.0 / (1.0 + ydist2_(i, j));
            const double dy0 = Y_(i, 0) - Y_(j, 0), dy1 = Y_(i, 1) - Y_(j, 1);
            const double sj2 = sigma_[j] * sigma_[j];
            const double Spj = S_pj_[j];
            if (!(Spj > 0.0) || !std::isfinite(Spj))
                throw std::runtime_error("Degenerate high-dimensional affinity normalization");
            for (std::size_t k = 0; k < d_; ++k) {
                const double xij = X_(i, k) - X_(j, k);
                const double xji = -xij;
                const double Spj_d = (exp_ji[j] / sj2) * xji;
                const double Pji_d =
                    ((-S_pi * xij / si2 * exp_ij[j]) - (exp_ij[j] * Spi_d[k])) / (S_pi * S_pi);
                const double Pij_d =
                    ((Spj * exp_ji[j] / sj2 * xji) - (exp_ji[j] * Spj_d)) / (Spj * Spj);
                const double vd = (Pji_d + Pij_d) / (2.0 * static_cast<double>(n_));
                B0[k] += 4.0 * vd * E * dy0;
                B1[k] += 4.0 * vd * E * dy1;
            }
        }

        double det = H00 * H11 - H01 * H10;
        const double scale =
            std::max({std::abs(H00), std::abs(H01), std::abs(H10), std::abs(H11), 1.0});
        if (std::abs(det) < 1e-14 * scale * scale) {
            const double ridge = 1e-10 * scale;
            H00 += ridge;
            H11 += ridge;
            det = H00 * H11 - H01 * H10;
        }
        if (std::abs(det) < std::numeric_limits<double>::epsilon())
            throw std::runtime_error("Singular 2x2 Hessian at sample " + std::to_string(i));
        const double inv00 = H11 / det, inv01 = -H01 / det, inv10 = -H10 / det, inv11 = H00 / det;
        for (std::size_t k = 0; k < d_; ++k) {
            out(i, k) = -(inv00 * B0[k] + inv01 * B1[k]);
            out(i, d_ + k) = -(inv10 * B0[k] + inv11 * B1[k]);
        }
    }
};

int main(int argc, char **argv) {
    try {
        if (argc != 7) {
            std::cerr
                << "Usage: tsne_explainer X.csv Y.csv P.csv Q.csv sigma.csv out_gradients.csv\n";
            return 2;
        }
        Matrix X = read_csv(argv[1]), Y = read_csv(argv[2]), P = read_csv(argv[3]),
               Q = read_csv(argv[4]), sigma = read_csv(argv[5]);
        ExplainerCore explainer(X, Y, P, Q, sigma);
        Matrix gradients = explainer.compute_all();
        write_csv(argv[6], gradients);
        std::cout << "Computed gradients for " << X.rows << " samples and " << X.cols
                  << " features.\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
