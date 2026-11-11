#pragma once

// Procedural self-propelled anti-aircraft gun (tracked hull, rotating
// turret with search + tracking radar, twin 35 mm guns on the turret
// flanks). Hard-surface geometry with chamfered plates, road wheels,
// individual track links, hatches, smoke dischargers and stowage.
//
// Parts are built in their own pivot space; the engine composes
//   hull   = T(pos) * Ry(body_yaw)
//   turret = hull   * T(turret_pivot) * Ry(turret_yaw)
//   guns   = turret * T(gun_pivot)    * Rx(-pitch)
//   radar  = turret * T(radar_pivot)  * Ry(spin)
// Each part is split per material (camo metal / dark steel / rubber).

#include "game/level.h"

#include <glm/vec3.hpp>
#include <vector>

namespace qlike {

struct AaModel {
    enum Part { kHull = 0, kTurret, kGuns, kRadar, kPartCount };
    // meshes[part] = one LevelMesh per material.
    std::vector<game::LevelMesh> meshes[kPartCount];
    glm::vec3 turret_pivot{0.0f}, gun_pivot{0.0f}, radar_pivot{0.0f};
    glm::vec3 muzzle[2]{};          // in gun space
    glm::vec3 half_extent{1.7f, 1.4f, 3.6f};   // hull collision (centre y = half)
    // Missile launcher only: one missile (tail at origin, nose +Z) and the
    // four rail positions in launch-frame ("guns") space.
    std::vector<game::LevelMesh> missile;
    glm::vec3 rail[4]{};
};

AaModel build_aa_model();
AaModel build_sam_model();

} // namespace qlike
