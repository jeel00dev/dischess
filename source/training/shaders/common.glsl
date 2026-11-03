// Shared declarations for every DisEngine trainer kernel.
// Must match dis::train::Push and the binding table in trainer.cpp.
//
// Parameter vector layout (identical to dv5::FloatModel::weights):
//   [0, features*W)            feature-transformer rows (bucketed)
//   [ft_bias, +W)              accumulator bias
//   [direct_w, +heads*2*W)     linear SCReLU head, [head][view][W]
//   [direct_b, +heads)         head bias
//   [factor, +planes*W)        factorizer rows (shared by all king buckets)
layout(push_constant) uniform Push {
    uint width;        // accumulator width W
    uint batch;        // positions in this batch
    uint ft_bias;
    uint direct_w;
    uint direct_b;
    uint factor;       // factor block offset
    uint planes;       // 768
    uint count;        // kernel-specific element count
    uint factorized;   // 1 when the factor block exists
    uint chunks;       // dense-gradient position chunks
    uint chunk_size;
    uint pad0;
    float cp_scale;    // 600: network output units -> centipawns
    float eval_scale;  // 400: centipawns -> win probability
    float power;       // loss exponent (2 = MSE in sigmoid space)
    float inv_batch;
    float step;        // Adam step size incl. bias correction
    float beta1;
    float beta2;
    float eps;         // eps incl. bias correction
    float decay;       // lr * weight_decay (decoupled)
    float clip;        // weight clamp
    float pad1;
    float pad2;
} pc;

struct Pos {
    float material;    // obs::Board::material(), side-to-move POV
    float target;      // blended training target in [0,1]
    uint head;         // output bucket 0..7
    uint count;        // active features per view
};

layout(std430, binding = 0) buffer WeightsBuf { float W[]; };
layout(std430, binding = 1) buffer GradBuf { float G[]; };
layout(std430, binding = 2) buffer Moment1Buf { float M1[]; };
layout(std430, binding = 3) buffer Moment2Buf { float M2[]; };
layout(std430, binding = 4) buffer AccBuf { float ACC[]; };      // [batch][2][W] pre-activation
layout(std430, binding = 5) buffer OutBuf { float OUT[]; };      // dL/dy | loss | prediction(cp)
layout(std430, binding = 6) readonly buffer FeatBuf { uint FEATS[]; };   // [batch][2][32]
layout(std430, binding = 7) readonly buffer PosBuf { Pos POS[]; };
layout(std430, binding = 8) readonly buffer SegBuf { uvec4 SEG[]; };     // dest, begin, end, -
layout(std430, binding = 9) readonly buffer EntryBuf { uint ENTRIES[]; }; // b*2+view
layout(std430, binding = 10) buffer ScratchBuf { float SCRATCH[]; };
