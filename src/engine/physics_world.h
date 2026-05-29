#pragma once

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <memory>

namespace qlike {

// Thin Jolt wrapper. Owns a PhysicsSystem ready for non-player rigid bodies
// (dynamic crates, projectiles, ragdolls). The player movement is *not* run
// through this — Quake-feel needs the custom kinematic code.
class PhysicsWorld {
public:
    PhysicsWorld();
    ~PhysicsWorld();

    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;

    // Advance simulation by dt seconds. Internally subdivides if dt is large.
    void step(float dt);

    // Body count snapshot — handy for log diagnostics.
    int body_count() const;

    // Add a static box AABB to the world (for matching the level geometry to Jolt).
    void add_static_box(glm::vec3 center, glm::vec3 half_extents);

    // Batch-add many static boxes in one BodyInterface call. ~10x faster
    // than calling add_static_box per brush at level load (Jolt's
    // AddBodiesPrepare/Finalize amortizes the body-mutex acquire and the
    // broadphase tree rebuild). Bodies stay disabled until prepare returns.
    struct StaticBox { glm::vec3 center; glm::vec3 half_extents; };
    void add_static_boxes(const StaticBox* boxes, size_t count);

    // Voxel-building collision: replace the single static body that
    // represents the destructible voxel shape with a fresh one built from
    // `count` world-space boxes (one StaticCompoundShape → one Jolt body,
    // so it never threatens the body cap regardless of box count). Pass
    // count == 0 to just remove the existing body. Called at voxel init and
    // rebuilt after every carve. Projectiles collide with this for
    // shoot-to-destroy; the player uses the same boxes via game::collision.
    void set_voxel_collision(const StaticBox* boxes, size_t count);

    // Add a procedural heightmap as a single Jolt HeightFieldShape static
    // body. `dim` is the cell count per side; `samples` must be of size
    // (dim+1) * (dim+1) and laid out row-major (x outer, z inner — matching
    // Heightmap::at). `origin_xz` is the world-space corner the (0,0) cell
    // sits at; `cell_size` is the world distance between adjacent samples.
    void add_static_heightfield(const float* samples, int dim,
                                glm::vec2 origin_xz, float cell_size);

    // Engine-side body handle (cached BodyID). Storing this on each
    // dynamic prop / particle / projectile lets the read-only queries skip
    // the unordered_map lookup in physics_world.cpp. 0 = invalid.
    using BodyHandle = uint32_t;
    BodyHandle handle_of(uint32_t id) const;
    bool get_body_world_matrix_h(BodyHandle h, glm::mat4& out) const;
    bool is_body_active_h(BodyHandle h) const;
    glm::vec3 get_linear_velocity_h(BodyHandle h) const;
    void apply_impulse_h(BodyHandle h, glm::vec3 impulse);

    // Spawn a dynamic box; returns an opaque body id (uint32_t).
    // `euler_radians` is XYZ Euler order (pitch, yaw, roll) — pass {0,0,0} for upright.
    uint32_t add_dynamic_box(glm::vec3 center, glm::vec3 half_extents,
                             glm::vec3 euler_radians = glm::vec3(0.0f),
                             float density = 1000.0f);

    // Spawn a dynamic cylinder (axis = Y in local space) with continuous
    // collision detection (Jolt's `LinearCast` motion quality). The body is
    // sweep-cast at the start of each physics step, so a fast projectile
    // CANNOT tunnel through a thinner box between ticks. Used for bullets.
    //   `orientation` is the quaternion that rotates the local +Y axis to the
    //   firing direction (so the cylinder points along travel).
    //   `linear_velocity` is set at creation (m/s, world space).
    //   `mass` is in kg (explicit, not density-derived).
    uint32_t add_dynamic_cylinder_ccd(glm::vec3 center,
                                      float radius, float half_length,
                                      glm::quat orientation,
                                      glm::vec3 linear_velocity,
                                      float mass);

    // Spawn a small dynamic sphere — used as a hit-effect particle. Bounces
    // realistically off everything in the world via Jolt's collision response
    // (this is exactly the "particles bounce off material and ground/walls"
    // requirement). `restitution` 0..1 (0=no bounce, 1=perfectly elastic).
    uint32_t add_dynamic_sphere(glm::vec3 center, float radius,
                                glm::vec3 linear_velocity,
                                float mass,
                                float restitution = 0.5f,
                                float friction = 0.4f);

    // World matrix (translation + rotation) for rendering. False if id missing.
    bool get_body_world_matrix(uint32_t id, glm::mat4& out) const;

    // True if the body is awake (Jolt is still simulating it). When a dynamic
    // body settles and Jolt sleeps it, its transform stops changing — callers
    // can skip recomputing per-frame derived state (AABB, render matrix). False
    // for unknown ids and for sleeping bodies.
    bool is_body_active(uint32_t id) const;

    // Current linear velocity in world space (m/s). Returns (0,0,0) if id is
    // missing. Used by projectile bookkeeping to detect bounce / impact.
    glm::vec3 get_linear_velocity(uint32_t id) const;

    // Remove and destroy a previously-added dynamic body.
    void remove_body(uint32_t id);

    struct RayHit {
        bool hit = false;
        bool dynamic = false;     // true if a MOVING-layer body was hit
        uint32_t body_id = 0;     // qlike id (only valid if `dynamic`)
        glm::vec3 position{0.0f}; // world-space hit point
        glm::vec3 normal{0.0f};   // surface normal at hit
        float distance = 0.0f;
    };

    // Cast a world-space ray. Returns the closest hit (statics + dynamics).
    RayHit raycast(glm::vec3 origin, glm::vec3 direction, float max_distance) const;

    // Apply a directional impulse (kg·m/s) to a dynamic body.
    void apply_impulse(uint32_t id, glm::vec3 impulse);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace qlike
