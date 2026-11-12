#pragma once

// Procedural trees: a C++ port of the generation algorithm of EZ-Tree
// (https://github.com/dgreenheck/ez-tree, MIT License, Copyright (c) 2024
// Daniel Greenheck). The skeleton code follows tree.js (growBranch,
// generateChildBranches, generateLeaves): branches grow level by level in
// sections whose orientation is perturbed by `gnarliness`, twisted, and
// bent by a growth force; child branches and leaves are stratified along
// their parent and around it. Presets carry the original option values.
//
// Meshing is this engine's own: branches become tapered tubes (vertex
// material 0, bark) and every leaf a card (material 7) whose normal slot
// carries (u, v, kind) for the procedural leaf cut-out in veg_leaf.glsl.

#include "game/level.h"

#include <glm/glm.hpp>

#include <cstdint>

namespace qlike {

struct EzTreeOptions {
    uint32_t seed = 0;
    bool evergreen = false;
    int levels = 3;
    float angle[4]      = { 0.0f, 70.0f, 60.0f, 60.0f };     // child branch angle (deg), per level
    int   children[4]   = { 7, 7, 5, 0 };
    glm::vec3 force_dir{0.0f, 1.0f, 0.0f};
    float force_strength = 0.01f;
    float gnarliness[4] = { 0.15f, 0.2f, 0.3f, 0.02f };
    float length[4]     = { 20.0f, 20.0f, 10.0f, 1.0f };
    float radius[4]     = { 1.5f, 0.7f, 0.7f, 0.7f };
    int   sections[4]   = { 12, 10, 8, 6 };
    int   segments[4]   = { 8, 6, 4, 3 };
    float start[4]      = { 0.0f, 0.4f, 0.3f, 0.3f };
    float taper[4]      = { 0.7f, 0.7f, 0.7f, 0.7f };
    float twist[4]      = { 0.0f, 0.0f, 0.0f, 0.0f };
    // Leaves.
    float leaf_angle = 10.0f;
    int   leaf_count = 1;
    float leaf_start = 0.0f;
    float leaf_size = 2.5f;
    float leaf_size_variance = 0.7f;
    bool  leaf_double = true;        // two crossed cards per leaf
    float leaf_kind = 0.0f;          // atlas cell: 0 oak, 1 spruce, 2 birch, 3 pine (odd = conifer)
};

EzTreeOptions ez_preset_oak();
EzTreeOptions ez_preset_pine();
EzTreeOptions ez_preset_bush();

// Continuous level of detail. Every element of a tree (a branch, a leaf
// card) has a PRIORITY in 0..1: the trunk 0, limbs low, twigs high, leaves
// spread uniformly above their branch. An instance drawn at detail d shows
// the elements with priority <= d; veg.vert shrinks the ones just below d
// to nothing and enlarges the remaining leaves to keep the crown covered,
// so detail changes continuously with distance instead of in steps. The
// draw buckets are prefixes of that ordering: bucket k holds the elements
// with priority <= kVegDetail[k].
constexpr float kVegDetail[4] = { 1.0f, 0.5f, 0.22f, 0.09f };

// Generates the tree and appends the elements with priority <= max_detail
// to `m`, uniformly scaled so it stands `target_height` metres tall (base
// at the origin, +Y up).
//   branch vertex : normal = outward normal, u = floor(priority * 1023) + radius (< 1 m), v = 8
//   leaf card     : normal = corner - leaf origin, u = 4 * floor(priority * 1023) + corner (0..3),
//                   v = 7 for atlas cell 0, 8 + cell otherwise (9 spruce, 10 birch, 11 pine)
void build_ez_tree(game::LevelMesh& m, const EzTreeOptions& o, float max_detail, float target_height);

}  // namespace qlike
