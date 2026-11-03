#!/usr/bin/env bash
# Builds the tournament executable engine/disengine from source/engine.
#
# Portable by default: x86-64-v3 (AVX2, any x86 CPU from ~2013 on) and fully
# static, so it runs on the organisers' machine without matching libraries.
#   DIS_MARCH=native   tune for this machine instead
#   DIS_STATIC=0       dynamic linking (if static libc is unavailable)
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
src="$root/source/engine"
out="$root/engine/disengine"
cxx="${CXX:-g++}"
march="${DIS_MARCH:-x86-64-v3}"
flags=(-std=c++20 -O3 -flto=auto -march="$march" -DNDEBUG -DDISENGINE_NET_V7 -I"$src")
link=(-pthread)
[ "${DIS_STATIC:-1}" = 1 ] && link+=(-static)
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
"$cxx" "${flags[@]}" "$src/main.cpp" "$src/search.cpp" "$src/chess.cpp" \
       "$src/v7-core/network.cpp" "$src/v7-core/features.cpp" "$src/v7-core/architecture.cpp" \
       "${link[@]}" -o "$tmp/disengine"
mkdir -p "$root/engine"
mv -f "$tmp/disengine" "$out"   # atomic replace: safe while a running bot uses the old binary
echo "built $out ($march${DIS_STATIC:+, static=${DIS_STATIC:-1}})"
