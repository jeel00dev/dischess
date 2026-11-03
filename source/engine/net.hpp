// Team-owned glue header: selects the network backend compiled into the engine.
// The dv5 network (source/v7-core) serves both retained architectures
// (linear512 and linear2048) from the same DISV5Q01 container and exposes the
// engine's eval API (load/refresh/advance/evaluate) on the obs::Board used by
// the search. All search/UCI/movegen code is this team's own implementation.
#pragma once
#ifndef DISENGINE_NET_V7
#error "engine build requires -DDISENGINE_NET_V7 (dv5 network backend); see build.sh"
#endif
#include "v7-core/network.hpp"
namespace obs {
  using dv5::Network;      // identical public API as the historical V5 network
  using dv5::Accumulator;  // same refresh/advance/evaluate semantics
  using dv5::RefreshCache; // per-thread accumulator refresh cache
}
