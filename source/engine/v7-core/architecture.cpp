#include "architecture.hpp"
#include <algorithm>
#include <stdexcept>
namespace dv5 {
const std::vector<Config> &candidates() {
    static const std::vector<Config> all = {
        {"wide1024", 1024, 16, 8, Inputs::Pieces, Head::Linear, true, {0, 0, 0}},
        {"linear512", 512, 16, 8, Inputs::Pieces, Head::Linear, true, {0, 0, 0}},
        {"compact256", 256, 16, 8, Inputs::Pieces, Head::Pair, true, {16, 32, 0}},
        {"compact512", 512, 16, 8, Inputs::Pieces, Head::Pair, true, {16, 32, 0}},
        {"compact1024", 1024, 16, 8, Inputs::Pieces, Head::Pair, true, {16, 32, 0}},
        {"classic-kp256", 256, 64, 8, Inputs::HalfKP, Head::Classic, false, {32, 32, 0}},
        {"king32-compact512", 512, 32, 8, Inputs::Pieces, Head::Pair, true, {16, 32, 0}},
        {"king32-compact1024", 1024, 32, 8, Inputs::Pieces, Head::Pair, true, {16, 32, 0}},
        {"mixed512", 512, 16, 8, Inputs::Pieces, Head::Mixed, false, {32, 32, 0}},
        {"deep512", 512, 16, 8, Inputs::Pieces, Head::Deep, true, {64, 32, 16}},
        {"relations256", 256, 16, 8, Inputs::Relations, Head::Pair, true, {16, 32, 0}},
        {"pawnpairs512", 512, 16, 8, Inputs::Pawns, Head::Pair, true, {16, 32, 0}},
        {"relations-pawns256", 256, 16, 8, Inputs::RelationsPawns, Head::Pair, true, {16, 32, 0}},
        {"compact1536", 1536, 16, 8, Inputs::Pieces, Head::Pair, true, {16, 32, 0}},
        {"compact2048", 2048, 16, 8, Inputs::Pieces, Head::Pair, true, {16, 32, 0}},
        {"linear2048", 2048, 16, 8, Inputs::Pieces, Head::Linear, true, {0, 0, 0}}};
    return all;
}
Config config_by_id(const std::string &id) {
    for (const auto &c : candidates())
        if (c.id == id)
            return c;
    throw std::runtime_error("Unknown V5 architecture: " + id);
}
void Config::validate() const {
    if (width < 256 || width > 2048 || width % 256 || heads != 8 ||
        (buckets != 16 && buckets != 32 && buckets != 64) || (halfkp() != (buckets == 64)))
        throw std::runtime_error("Invalid V5 dimensions");
    if (head != Head::Linear && (!hidden[0] || !hidden[1]))
        throw std::runtime_error("Missing dense layer");
}
int Config::features() const {
    return piece_features() + (relations() ? relation_dimensions(pawns()) : 0) + (pawns() ? 4560 : 0);
}
Layout::Layout(Config c) : config(std::move(c)) {
    config.validate();
    const auto &q = config;
    size_t next = size_t(q.features()) * q.width;
    ft_bias = next;
    next += q.width;
    if (q.direct) {
        direct_weights = next;
        next += size_t(q.heads) * 2 * q.width;
        direct_bias = next;
        next += q.heads;
    }
    head_input = q.head == Head::Linear ? 0 : (q.head == Head::Classic ? 2 * q.width : q.width);
    int previous = head_input;
    int act_next = 2 * q.width + head_input, delta_next = 2 * q.width;
    for (int i = 0; i < 3 && q.hidden[i]; ++i) {
        Layer l;
        l.input = previous;
        l.neurons = q.hidden[i];
        l.activation = q.head == Head::Classic ? Activation::Clipped
                       : q.head == Head::Mixed ? Activation::Mixed
                                               : Activation::Squared;
        l.outputs = l.neurons * (l.activation == Activation::Mixed ? 2 : 1);
        l.weights = next;
        next += size_t(q.heads) * l.input * l.neurons;
        l.bias = next;
        next += size_t(q.heads) * l.neurons;
        z_offset[i] = act_next;
        act_next += l.neurons;
        delta_offset[i] = delta_next;
        delta_next += l.neurons;
        previous = l.outputs;
        layers.push_back(l);
    }
    if (!layers.empty()) {
        output_input = previous + (q.head == Head::Mixed ? layers[0].outputs : 0);
        output_weights = next;
        next += size_t(q.heads) * output_input;
        output_bias = next;
        next += q.heads;
    }
    parameters = next;
    factor_parameters = size_t(q.factor_features()) * q.width;
    activation_stride = act_next;
    delta_stride = delta_next;
}
std::string describe(const Config &c) {
    return c.id + " (" + std::to_string(c.features()) + "->" + std::to_string(c.width) + ")x2";
}
uint64_t hash_bytes(const void *ptr, size_t n) {
    auto p = static_cast<const unsigned char *>(ptr);
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < n; ++i)
        h = (h ^ p[i]) * 1099511628211ULL;
    return h;
}
uint64_t mix64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
} // namespace dv5
