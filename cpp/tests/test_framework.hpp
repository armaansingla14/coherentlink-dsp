#pragma once
// Minimal self-registering test framework (no external dependencies).

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace testing {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

inline int& failures() {
    static int f = 0;
    return f;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

}  // namespace testing

#define TEST(name)                                                   \
    static void name();                                              \
    static ::testing::Registrar registrar_##name(#name, name);       \
    static void name()

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);             \
            ++::testing::failures();                                                  \
        }                                                                             \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                 \
    do {                                                                                      \
        const double va_ = (a), vb_ = (b);                                                    \
        if (!(std::fabs(va_ - vb_) <= (tol))) {                                               \
            std::printf("  FAIL %s:%d: %s=%.12g vs %s=%.12g\n", __FILE__, __LINE__, #a, va_, #b, vb_); \
            ++::testing::failures();                                                          \
        }                                                                                     \
    } while (0)

#define CHECK_THROWS(expr)                         \
    do {                                           \
        bool thrown_ = false;                      \
        try {                                      \
            (void)(expr);                          \
        } catch (...) {                            \
            thrown_ = true;                        \
        }                                          \
        CHECK(thrown_);                            \
    } while (0)
