// The trainer's record->feature path must produce exactly the features the
// engine's inference code uses (dv5::active_features on the unpacked board),
// in the same order, with the same output bucket and material term.
#include "check.hpp"
#include "common/records.hpp"
#include <algorithm>
using namespace obs;

int main() {
    auto positions = random_positions(20000, 11);
    for (const char *arch : {"linear2048", "linear512", "wide1024"}) {
        auto c = dv5::config_by_id(arch);
        int checked = 0;
        for (auto &b : positions) {
            Record r = pack(b, 0, 0);
            dis::PackedFeatures pf;
            CHECK(dis::extract_features(c, r, pf));
            Board u = unpack(r);
            CHECK_EQ(pf.head, dv5::output_head(u));
            CHECK_EQ(pf.material, u.material());
            for (int view = 0; view < 2; ++view) {
                auto ref = dv5::active_features(c, u, u.turn ^ view);
                CHECK_EQ(ref.count, pf.count);
                for (int i = 0; i < std::min(ref.count, pf.count); ++i)
                    CHECK_EQ(ref.indices[i], pf.index[view][i]);
            }
            ++checked;
        }
        std::cout << arch << ": " << checked << " positions identical\n";
    }
    // Result/flag bit packing.
    Record r;
    dis::set_record_flags(r, RECORD_QUIET_PV);
    dis::set_record_result(r, dis::Result::Draw);
    CHECK(dis::record_result(r) == dis::Result::Draw);
    CHECK_EQ(int(dis::record_flags(r)), int(RECORD_QUIET_PV));
    CHECK(dis::result_for(dis::Result::Win, 1) == dis::Result::Loss);
    return finish("test_features");
}
