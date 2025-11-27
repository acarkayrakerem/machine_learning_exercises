#include <cstdio>
#include <cstdint>
#include <thread>
#include <vector>
#include <chrono>

static inline uint64_t denom_u64(uint64_t i) {
    return 16ull * i * i - 1ull;
}

static inline uint64_t delta4_u64(uint64_t i) {
    return 128ull * i + 256ull;
}

static inline double sum_deltas_range(uint64_t start, uint64_t end) {
    if (start > end) return 0.0;
    const double neg2 = -2.0;
    uint64_t count = end - start + 1;
    uint64_t blocks = count / 4;
    uint64_t rem = count % 4;

    // Initialize four lanes
    uint64_t i0 = start;
    uint64_t i1 = i0 + 1;
    uint64_t i2 = i0 + 2;
    uint64_t i3 = i0 + 3;

    uint64_t d0 = denom_u64(i0);
    uint64_t d1 = denom_u64(i1);
    uint64_t d2 = denom_u64(i2);
    uint64_t d3 = denom_u64(i3);

    uint64_t e0 = delta4_u64(i0);
    uint64_t e1 = delta4_u64(i1);
    uint64_t e2 = delta4_u64(i2);
    uint64_t e3 = delta4_u64(i3);

    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;

    for (uint64_t k = 0; k < blocks; ++k) {
        s0 += neg2 / static_cast<double>(d0); d0 += e0; e0 += 512ull;
        s1 += neg2 / static_cast<double>(d1); d1 += e1; e1 += 512ull;
        s2 += neg2 / static_cast<double>(d2); d2 += e2; e2 += 512ull;
        s3 += neg2 / static_cast<double>(d3); d3 += e3; e3 += 512ull;
    }

    double sum = s0 + s1 + s2 + s3;

    uint64_t i = start + blocks * 4;
    for (uint64_t r = 0; r < rem; ++r, ++i) {
        sum += neg2 / static_cast<double>(denom_u64(i));
    }
    return sum;
}

static double calculate(uint64_t iterations, int param1, int param2) {
    if (param1 == 4 && param2 == 1) {
        unsigned int T = std::thread::hardware_concurrency();
        if (T == 0) T = 8;
        std::vector<std::thread> threads;
        std::vector<double> partials(T, 0.0);
        threads.reserve(T);

        uint64_t base = 1;
        uint64_t q = iterations / T;
        uint64_t r = iterations % T;

        for (unsigned int t = 0; t < T; ++t) {
            uint64_t len = q + (t < r ? 1 : 0);
            uint64_t start = base;
            uint64_t end = start + len - 1;
            base += len;
            threads.emplace_back([start, end, &partials, t]() {
                partials[t] = sum_deltas_range(start, end);
            });
        }
        for (auto &th : threads) th.join();

        double total = 1.0;
        for (double v : partials) total += v;
        return total;
    } else {
        // Generic fallback matching Python semantics
        double result = 1.0;
        const double p1 = static_cast<double>(param1);
        const double p2 = static_cast<double>(param2);
        for (uint64_t i = 1; i <= iterations; ++i) {
            double j = i * p1 - p2;
            result -= 1.0 / j;
            j = i * p1 + p2;
            result += 1.0 / j;
        }
        return result;
    }
}

int main() {
    const uint64_t iterations = 200000000ull;
    const int param1 = 4;
    const int param2 = 1;

    auto start_time = std::chrono::high_resolution_clock::now();
    double result = calculate(iterations, param1, param2) * 4.0;
    auto end_time = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    std::printf("Result: %.12f\n", result);
    std::printf("Execution Time: %.6f seconds\n", elapsed);
    return 0;
}
