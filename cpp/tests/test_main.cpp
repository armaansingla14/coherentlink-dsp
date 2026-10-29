#include "test_framework.hpp"

int main() {
    int failed_tests = 0;
    for (const auto& t : testing::registry()) {
        const int before = testing::failures();
        t.fn();
        const bool ok = testing::failures() == before;
        if (!ok) ++failed_tests;
        std::printf("[%s] %s\n", ok ? " OK " : "FAIL", t.name.c_str());
    }
    std::printf("%zu tests, %d failed\n", testing::registry().size(), failed_tests);
    return failed_tests == 0 ? 0 : 1;
}
