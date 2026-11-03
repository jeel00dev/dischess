// Minimal --key value / --flag command-line parsing for the tools.
#pragma once
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace dis {
class Args {
  public:
    // `switches` are boolean options that never consume the following argument.
    Args(int argc, char **argv, std::string usage, std::set<std::string> switches = {})
        : usage_(std::move(usage)) {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "-h" || a == "--help") {
                std::cout << usage_ << '\n';
                std::exit(0);
            }
            if (a.rfind("--", 0) != 0) {
                positional_.push_back(a);
                continue;
            }
            a = a.substr(2);
            if (auto eq = a.find('='); eq != std::string::npos) {
                values_[a.substr(0, eq)].push_back(a.substr(eq + 1));
            } else if (!switches.count(a) && i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                values_[a].push_back(argv[++i]);
            } else {
                values_[a].push_back("true");
            }
        }
    }
    bool has(const std::string &k) const { return values_.count(k); }
    std::string str(const std::string &k, const std::string &def = "") const {
        auto it = values_.find(k);
        return it == values_.end() ? def : it->second.back();
    }
    std::string required(const std::string &k) const {
        if (!has(k))
            throw std::runtime_error("missing --" + k + "\n\n" + usage_);
        return str(k);
    }
    std::vector<std::string> all(const std::string &k) const {
        auto it = values_.find(k);
        return it == values_.end() ? std::vector<std::string>{} : it->second;
    }
    long long integer(const std::string &k, long long def) const {
        return has(k) ? std::stoll(str(k)) : def;
    }
    double real(const std::string &k, double def) const {
        return has(k) ? std::stod(str(k)) : def;
    }
    bool flag(const std::string &k) const {
        return has(k) && str(k) != "false" && str(k) != "0";
    }
    const std::vector<std::string> &positional() const { return positional_; }

  private:
    std::string usage_;
    std::map<std::string, std::vector<std::string>> values_;
    std::vector<std::string> positional_;
};

template <class F> int run_tool(F &&body) {
    try {
        return body();
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
} // namespace dis
