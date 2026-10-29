// Command-line front end used by the Python parity tests.
//   clink_tool crc <hex-bytes>
//   clink_tool fir <taps.txt> <input.txt> <output.txt>   (one int16 per line)
//   clink_tool bus <requests.txt> <responses.txt>         (one hex frame per line)
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "clink/crc16.hpp"
#include "clink/fir_q15.hpp"
#include "clink/regfile.hpp"
#include "clink/regmap_table.hpp"

namespace {

std::vector<std::uint8_t> parse_hex(const std::string& s) {
    std::vector<std::uint8_t> out;
    for (std::size_t i = 0; i + 1 < s.size(); i += 2) out.push_back(static_cast<std::uint8_t>(std::stoul(s.substr(i, 2), nullptr, 16)));
    return out;
}

std::vector<std::int16_t> read_ints(const char* path) {
    std::ifstream in(path);
    std::vector<std::int16_t> v;
    long x;
    while (in >> x) v.push_back(static_cast<std::int16_t>(x));
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace clink;
    if (argc >= 3 && std::strcmp(argv[1], "crc") == 0) {
        auto bytes = parse_hex(argv[2]);
        std::printf("%04X\n", crc16(bytes.data(), bytes.size()));
        return 0;
    }
    if (argc == 5 && std::strcmp(argv[1], "fir") == 0) {
        FirQ15 fir(read_ints(argv[2]));
        auto y = fir.filter(read_ints(argv[3]));
        std::ofstream out(argv[4]);
        for (auto v : y) out << v << '\n';
        std::fprintf(stderr, "saturations=%zu\n", fir.saturations());
        return 0;
    }
    if (argc == 4 && std::strcmp(argv[1], "bus") == 0) {
        RegisterFile rf(kDspRegisters);
        std::ifstream in(argv[2]);
        std::ofstream out(argv[3]);
        std::string line;
        while (std::getline(in, line)) {
            auto bytes = parse_hex(line);
            if (bytes.size() != kRequestLen) {
                std::cerr << "bad frame length\n";
                return 1;
            }
            Request req{};
            std::copy(bytes.begin(), bytes.end(), req.begin());
            Response r = rf.handle(req);
            char buf[3];
            for (auto b : r) {
                std::snprintf(buf, sizeof buf, "%02X", b);
                out << buf;
            }
            out << '\n';
        }
        return 0;
    }
    std::fprintf(stderr, "usage: clink_tool crc <hex> | fir <taps> <in> <out> | bus <req> <resp>\n");
    return 2;
}
