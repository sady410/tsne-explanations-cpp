#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

struct Matrix {
    size_t rows{}, cols{};
    std::vector<double> a;
    Matrix() = default;
    Matrix(size_t r, size_t c, double v = 0.0) : rows(r), cols(c), a(r * c, v) {}
    double &operator()(size_t i, size_t j) { return a[i * cols + j]; }
    double operator()(size_t i, size_t j) const { return a[i * cols + j]; }
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
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open " + path);
    std::vector<double> vals;
    std::string line;
    size_t cols = 0, rows = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        std::vector<double> row;
        size_t start = 0;
        while (true) {
            size_t p = line.find(',', start);
            row.push_back(
                parse_finite_double(line.substr(start, p == std::string::npos ? p : p - start),
                                    "CSV value in " + path));
            if (p == std::string::npos)
                break;
            start = p + 1;
        }
        if (!cols)
            cols = row.size();
        if (row.size() != cols)
            throw std::runtime_error("Inconsistent CSV columns");
        vals.insert(vals.end(), row.begin(), row.end());
        ++rows;
    }
    if (rows == 0 || cols == 0)
        throw std::runtime_error("Empty CSV file: " + path);
    Matrix m(rows, cols);
    m.a = std::move(vals);
    return m;
}

static void write_csv(const Matrix &m, const std::string &path) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("Cannot write " + path);
    f << std::setprecision(17);
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            if (j)
                f << ',';
            f << m(i, j);
        }
        f << '\n';
    }
    if (!f)
        throw std::runtime_error("Failed while writing " + path);
}

static void write_vector_csv(const std::vector<double> &v, const std::string &path) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("Cannot write " + path);
    f << std::setprecision(17);
    for (double x : v)
        f << x << '\n';
    if (!f)
        throw std::runtime_error("Failed while writing " + path);
}

class CustomTSNE {
  public:
    Matrix X, Y, P, Q;
    std::vector<double> sigma;
    explicit CustomTSNE(Matrix x) : X(std::move(x)) {}

    static std::pair<double, std::vector<double>> Hbeta(const std::vector<double> &D, double beta) {
        std::vector<double> p(D.size());
        double sumP = 0.0, weighted = 0.0;
        for (size_t k = 0; k < D.size(); ++k) {
            p[k] = std::exp(-D[k] * beta);
            sumP += p[k];
            weighted += D[k] * p[k];
        }
        sumP = std::max(sumP, std::numeric_limits<double>::epsilon());
        double H = std::log(sumP) + beta * weighted / sumP;
        for (double &x : p)
            x /= sumP;
        return {H, std::move(p)};
    }

    std::pair<Matrix, std::vector<double>> x2p(double tol, double perplexity) const {
        const size_t n = X.rows, d = X.cols;
        Matrix D(n, n, 0.0);
#pragma omp parallel for schedule(static) if (n >= 64)
        for (long long i = 0; i < (long long)n; ++i)
            for (size_t j = (size_t)i + 1; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < d; ++k) {
                    const double z = X((size_t)i, k) - X(j, k);
                    sum += z * z;
                }
                D((size_t)i, j) = D(j, (size_t)i) = sum;
            }

        Matrix cond(n, n, 0.0);
        std::vector<double> beta(n, 1.0);
        const double logU = std::log(perplexity);
#pragma omp parallel for schedule(dynamic) if (n >= 64)
        for (long long ii = 0; ii < (long long)n; ++ii) {
            const size_t i = (size_t)ii;
            std::vector<double> Di;
            std::vector<size_t> idx;
            Di.reserve(n - 1);
            idx.reserve(n - 1);
            for (size_t j = 0; j < n; ++j)
                if (j != i) {
                    Di.push_back(D(i, j));
                    idx.push_back(j);
                }
            double bmin = -std::numeric_limits<double>::infinity(),
                   bmax = std::numeric_limits<double>::infinity();
            auto hp = Hbeta(Di, beta[i]);
            double hdiff = hp.first - logU;
            int tries = 0;
            while ((std::isnan(hdiff) || std::abs(hdiff) > tol) && tries < 50) {
                if (hdiff > 0) {
                    bmin = beta[i];
                    beta[i] = std::isinf(bmax) ? beta[i] * 2.0 : (beta[i] + bmax) / 2.0;
                } else {
                    bmax = beta[i];
                    beta[i] = std::isinf(bmin) ? beta[i] / 2.0 : (beta[i] + bmin) / 2.0;
                }
                hp = Hbeta(Di, beta[i]);
                hdiff = hp.first - logU;
                ++tries;
            }
            for (size_t k = 0; k < idx.size(); ++k)
                cond(i, idx[k]) = hp.second[k];
        }
        // Hbeta uses exp(-beta * distance), while the paper defines the
        // Gaussian kernel as exp(-distance / (2 * sigma^2)).
        std::vector<double> sig(n);
        for (size_t i = 0; i < n; ++i)
            sig[i] = std::sqrt(1.0 / (2.0 * beta[i]));
        return {std::move(cond), std::move(sig)};
    }

    void update_low_dimensional_affinities(Matrix &num) {
        const size_t n = Y.rows, no_dims = Y.cols;
        double qsum = 0.0;
#pragma omp parallel for reduction(+ : qsum) schedule(static) if (n >= 64)
        for (long long ii = 0; ii < (long long)n; ++ii) {
            const size_t i = (size_t)ii;
            for (size_t j = 0; j < n; ++j) {
                if (i == j) {
                    num(i, j) = 0.0;
                    continue;
                }
                double dist = 0.0;
                for (size_t k = 0; k < no_dims; ++k) {
                    double z = Y(i, k) - Y(j, k);
                    dist += z * z;
                }
                num(i, j) = 1.0 / (1.0 + dist);
                qsum += num(i, j);
            }
        }
        if (!(qsum > 0.0) || !std::isfinite(qsum))
            throw std::runtime_error("Cannot normalize low-dimensional affinities");
        for (size_t z = 0; z < Q.a.size(); ++z)
            Q.a[z] = std::max(num.a[z] / qsum, 1e-12);
    }

    void run(size_t no_dims, double perplexity, const Matrix &init, int max_iter = 400) {
        const size_t n = X.rows;
        if (n < 2)
            throw std::runtime_error("X must contain at least two samples");
        if (X.cols == 0)
            throw std::runtime_error("X must contain at least one feature");
        if (!(perplexity > 0.0 && perplexity < static_cast<double>(n)))
            throw std::runtime_error(
                "Perplexity must be positive and smaller than the sample count");
        if (max_iter <= 0)
            throw std::runtime_error("Iterations must be positive");
        if (init.rows != n || init.cols != no_dims)
            throw std::runtime_error("Bad Y_init shape");
        Y = init;
        Matrix dY(n, no_dims, 0.0), iY(n, no_dims, 0.0), gains(n, no_dims, 1.0);
        auto ps = x2p(1e-5, perplexity);
        Matrix cond = std::move(ps.first);
        sigma = std::move(ps.second);
        P = Matrix(n, n, 0.0);
        double total = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) {
                P(i, j) = cond(i, j) + cond(j, i);
                total += P(i, j);
            }
        for (double &v : P.a)
            v = std::max(v / total * 4.0, 1e-12);

        Q = Matrix(n, n, 0.0);
        Matrix num(n, n, 0.0);
        for (int iter = 0; iter < max_iter; ++iter) {
            update_low_dimensional_affinities(num);

#pragma omp parallel for schedule(static) if (n >= 64)
            for (long long ii = 0; ii < (long long)n; ++ii) {
                const size_t i = (size_t)ii;
                for (size_t k = 0; k < no_dims; ++k) {
                    double g = 0.0;
                    for (size_t j = 0; j < n; ++j)
                        g += (P(j, i) - Q(j, i)) * num(j, i) * (Y(i, k) - Y(j, k));
                    dY(i, k) = g;
                }
            }
            const double momentum = iter < 20 ? 0.5 : 0.8;
            for (size_t z = 0; z < Y.a.size(); ++z) {
                const bool sign_change = (dY.a[z] > 0) != (iY.a[z] > 0);
                gains.a[z] = sign_change ? gains.a[z] + 0.2 : gains.a[z] * 0.8;
                gains.a[z] = std::max(gains.a[z], 0.01);
                iY.a[z] = momentum * iY.a[z] - 500.0 * gains.a[z] * dY.a[z];
                Y.a[z] += iY.a[z];
            }
            for (size_t k = 0; k < no_dims; ++k) {
                double mean = 0.0;
                for (size_t i = 0; i < n; ++i)
                    mean += Y(i, k);
                mean /= n;
                for (size_t i = 0; i < n; ++i)
                    Y(i, k) -= mean;
            }
            if ((iter + 1) % 100 == 0) {
                double C = 0.0;
                for (size_t z = 0; z < P.a.size(); ++z)
                    C += P.a[z] * std::log(P.a[z] / Q.a[z]);
                std::cout << "iteration=" << (iter + 1) << " kl=" << std::setprecision(17) << C
                          << "\n";
            }
            if (iter == 100)
                for (double &v : P.a)
                    v /= 4.0;
        }
        // Keep exported Q consistent with the final, post-update embedding Y.
        update_low_dimensional_affinities(num);
    }
};

static void print_usage() {
    std::cerr << "Usage: tsne X.csv perplexity out_prefix [options]\n"
              << "Options:\n"
              << "  --iterations N   Optimization iterations (default: 400)\n"
              << "  --seed N         Random initialization seed (default: 42)\n"
              << "  --init FILE      Use a custom initialization CSV instead\n"
              << "  --help            Show this message\n";
}

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        print_usage();
        return 0;
    }
    if (argc < 4) {
        print_usage();
        return 2;
    }
    try {
        const std::string x_path = argv[1];
        const double perplexity = parse_finite_double(argv[2], "perplexity");
        const std::string pre = argv[3];
        int iters = 400;
        unsigned long long seed = 42;
        std::string init_path;
        for (int i = 4; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--iterations" && i + 1 < argc) {
                std::size_t consumed = 0;
                const std::string value = argv[++i];
                iters = std::stoi(value, &consumed);
                if (consumed != value.size())
                    throw std::runtime_error("Invalid iterations: " + value);
            } else if (option == "--seed" && i + 1 < argc) {
                std::size_t consumed = 0;
                const std::string value = argv[++i];
                if (!value.empty() && value.front() == '-')
                    throw std::runtime_error("Seed cannot be negative");
                seed = std::stoull(value, &consumed);
                if (consumed != value.size())
                    throw std::runtime_error("Invalid seed: " + value);
            } else if (option == "--init" && i + 1 < argc)
                init_path = argv[++i];
            else if (option == "--help") {
                print_usage();
                return 0;
            } else
                throw std::runtime_error("Unknown or incomplete option: " + option);
        }

        Matrix X = read_csv(x_path);
        Matrix init;
        if (init_path.empty()) {
            init = Matrix(X.rows, 2);
            std::mt19937_64 generator(seed);
            std::normal_distribution<double> distribution(0.0, 1e-4);
            for (double &value : init.a)
                value = distribution(generator);
            std::cout << "initialization=random seed=" << seed << "\n";
        } else {
            init = read_csv(init_path);
            std::cout << "initialization=" << init_path << "\n";
        }
        const auto t0 = std::chrono::steady_clock::now();
        CustomTSNE t(std::move(X));
        t.run(init.cols, perplexity, init, iters);
        const double sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        write_csv(t.Y, pre + "_Y.csv");
        write_csv(t.P, pre + "_P.csv");
        write_csv(t.Q, pre + "_Q.csv");
        write_vector_csv(t.sigma, pre + "_sigma.csv");
        std::cout << "seconds=" << std::setprecision(17) << sec << "\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
