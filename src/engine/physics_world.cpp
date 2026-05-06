#include "engine/physics_world.h"

#include "engine/log.h"

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/NarrowPhaseQuery.h>
#include <Jolt/Physics/Collision/CollisionCollector.h>
#include <Jolt/Physics/Collision/CollidePointResult.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <thread>
#include <unordered_map>
#include <vector>

JPH_SUPPRESS_WARNINGS

namespace qlike {

namespace {

namespace Layers {
    static constexpr JPH::ObjectLayer NON_MOVING = 0;
    static constexpr JPH::ObjectLayer MOVING     = 1;
    static constexpr JPH::ObjectLayer NUM        = 2;
}

namespace BPLayers {
    static constexpr JPH::BroadPhaseLayer NON_MOVING(0);
    static constexpr JPH::BroadPhaseLayer MOVING(1);
    static constexpr unsigned NUM = 2;
}

class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface {
public:
    BPLayerInterfaceImpl() {
        map_[Layers::NON_MOVING] = BPLayers::NON_MOVING;
        map_[Layers::MOVING]     = BPLayers::MOVING;
    }
    unsigned int GetNumBroadPhaseLayers() const override { return BPLayers::NUM; }
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer l) const override {
        return map_[l];
    }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer) const override {
        return "qlike-bp";
    }
#endif
private:
    JPH::BroadPhaseLayer map_[Layers::NUM]{};
};

class ObjectVsBPFilterImpl final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer obj, JPH::BroadPhaseLayer bp) const override {
        switch (obj) {
            case Layers::NON_MOVING: return bp == BPLayers::MOVING;
            case Layers::MOVING:     return true;
            default:                 return false;
        }
    }
};

class ObjectPairFilterImpl final : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override {
        switch (a) {
            case Layers::NON_MOVING: return b == Layers::MOVING;
            case Layers::MOVING:     return true;
            default:                 return false;
        }
    }
};

void trace_callback(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    log::infof("[jolt] %s", buf);
}

#ifdef JPH_ENABLE_ASSERTS
bool assert_callback(const char* expr, const char* msg, const char* file, JPH::uint line) {
    log::errorf("[jolt assert] %s:%u (%s) %s", file, line, expr, msg ? msg : "");
    return true;
}
#endif

std::atomic<int> g_init_count{0};

void global_init_once() {
    if (g_init_count.fetch_add(1) == 0) {
        JPH::RegisterDefaultAllocator();
        JPH::Trace = &trace_callback;
#ifdef JPH_ENABLE_ASSERTS
        JPH::AssertFailed = &assert_callback;
#endif
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
        log::info("[jolt] global init complete");
    }
}

void global_shutdown_if_last() {
    if (g_init_count.fetch_sub(1) == 1) {
        JPH::UnregisterTypes();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
        log::info("[jolt] global shutdown complete");
    }
}

} // namespace

struct PhysicsWorld::Impl {
    JPH::PhysicsSystem system;
    // 32 MB temp scratch — at 100 dyn props with up to 6 collision_steps
    // the constraint solver fits comfortably; previously 16 MB was tight
    // enough to occasionally fall back to the heap allocator.
    JPH::TempAllocatorImpl temp_alloc{ 32 * 1024 * 1024 };
    JPH::JobSystemThreadPool job_system{
        JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers,
        std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1) };

    BPLayerInterfaceImpl bp_layer_iface;
    ObjectVsBPFilterImpl obj_vs_bp;
    ObjectPairFilterImpl obj_pairs;

    std::unordered_map<uint32_t, JPH::BodyID> ids;
    uint32_t next_id = 1;
};

PhysicsWorld::PhysicsWorld() {
    global_init_once();
    impl_ = std::make_unique<Impl>();
    impl_->system.Init(
        /*maxBodies*/        2048,
        /*numBodyMutexes*/   0,
        /*maxBodyPairs*/     4096,
        /*maxContactConstraints*/ 2048,
        impl_->bp_layer_iface,
        impl_->obj_vs_bp,
        impl_->obj_pairs);
    impl_->system.SetGravity(JPH::Vec3(0, -25.0f, 0));
    // Tune the constraint solver. Default is 10 velocity / 2 position
    // iterations — fine for tall stacks of crates, overkill for our
    // shallow piles. 4 vel / 1 pos still resolves crate-on-floor stably
    // and shaves a measurable slice of step time at 100 dyn bodies.
    JPH::PhysicsSettings ps = impl_->system.GetPhysicsSettings();
    ps.mNumVelocitySteps = 4;
    ps.mNumPositionSteps = 1;
    impl_->system.SetPhysicsSettings(ps);
    log::info("PhysicsWorld initialized (Jolt)");
}

PhysicsWorld::~PhysicsWorld() {
    impl_.reset();
    global_shutdown_if_last();
}

void PhysicsWorld::step(float dt) {
    if (dt <= 0.0f) return;
    constexpr float kFixed = 1.0f / 60.0f;
    int collision_steps = 1;
    int subdivisions = std::max(1, static_cast<int>(std::ceil(dt / kFixed)));
    float sub_dt = dt / subdivisions;
    for (int i = 0; i < subdivisions; ++i) {
        impl_->system.Update(sub_dt, collision_steps,
                             &impl_->temp_alloc, &impl_->job_system);
    }
}

int PhysicsWorld::body_count() const {
    return static_cast<int>(impl_->system.GetNumBodies());
}

void PhysicsWorld::add_static_box(glm::vec3 c, glm::vec3 he) {
    // Jolt's default convex radius is 0.05 — fails on shapes thinner than that
    // (the lantern post halves are 0.09; the cap half-y is 0.05). Use a small
    // radius scaled to the box.
    float min_he = std::min({he.x, he.y, he.z});
    float convex_radius = std::min(0.04f, min_he * 0.4f);
    JPH::BoxShapeSettings ss(JPH::Vec3(he.x, he.y, he.z), convex_radius);
    ss.SetEmbedded();
    JPH::ShapeSettings::ShapeResult sr = ss.Create();
    if (sr.HasError()) {
        log::errorf("[jolt] static box shape: %s", sr.GetError().c_str());
        return;
    }
    JPH::BodyCreationSettings bcs(sr.Get(),
                                  JPH::RVec3(c.x, c.y, c.z),
                                  JPH::Quat::sIdentity(),
                                  JPH::EMotionType::Static,
                                  Layers::NON_MOVING);
    impl_->system.GetBodyInterface().CreateAndAddBody(bcs, JPH::EActivation::DontActivate);
}

void PhysicsWorld::add_static_boxes(const StaticBox* boxes, size_t count) {
    if (count == 0) return;
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    // Two-step: first allocate all bodies (bi.CreateBody, no add), then
    // batch-add via AddBodiesPrepare + AddBodiesFinalize. The prepare/
    // finalize pair takes the broadphase mutex once for the whole batch
    // instead of per body, which is the dominant cost when 150+ static
    // brushes are loaded at level start.
    std::vector<JPH::BodyID> ids;
    ids.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        glm::vec3 he = boxes[i].half_extents;
        float min_he = std::min({he.x, he.y, he.z});
        float convex_radius = std::min(0.04f, min_he * 0.4f);
        JPH::BoxShapeSettings ss(JPH::Vec3(he.x, he.y, he.z), convex_radius);
        ss.SetEmbedded();
        JPH::ShapeSettings::ShapeResult sr = ss.Create();
        if (sr.HasError()) {
            log::errorf("[jolt] static box shape (batch %zu): %s",
                        i, sr.GetError().c_str());
            continue;
        }
        glm::vec3 c = boxes[i].center;
        JPH::BodyCreationSettings bcs(sr.Get(),
                                      JPH::RVec3(c.x, c.y, c.z),
                                      JPH::Quat::sIdentity(),
                                      JPH::EMotionType::Static,
                                      Layers::NON_MOVING);
        if (JPH::Body* body = bi.CreateBody(bcs)) {
            ids.push_back(body->GetID());
        }
    }
    if (ids.empty()) return;
    JPH::BodyInterface::AddState state =
        bi.AddBodiesPrepare(ids.data(), static_cast<int>(ids.size()));
    bi.AddBodiesFinalize(ids.data(), static_cast<int>(ids.size()),
                          state, JPH::EActivation::DontActivate);
}

uint32_t PhysicsWorld::add_dynamic_box(glm::vec3 c, glm::vec3 he,
                                       glm::vec3 euler_rad, float density) {
    float min_he = std::min({he.x, he.y, he.z});
    float convex_radius = std::min(0.04f, min_he * 0.4f);
    JPH::BoxShapeSettings ss(JPH::Vec3(he.x, he.y, he.z), convex_radius);
    ss.SetEmbedded();
    JPH::ShapeSettings::ShapeResult sr = ss.Create();
    if (sr.HasError()) {
        log::errorf("[jolt] dyn box shape: %s", sr.GetError().c_str());
        return 0;
    }
    glm::quat gq = glm::quat(euler_rad);
    JPH::Quat jq(gq.x, gq.y, gq.z, gq.w);

    JPH::BodyCreationSettings bcs(sr.Get(),
                                  JPH::RVec3(c.x, c.y, c.z), jq,
                                  JPH::EMotionType::Dynamic,
                                  Layers::MOVING);
    bcs.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateMassAndInertia;
    bcs.mMassPropertiesOverride.mMass = density * he.x * he.y * he.z * 8.0f;

    JPH::BodyID bid = impl_->system.GetBodyInterface().CreateAndAddBody(
        bcs, JPH::EActivation::Activate);
    uint32_t id = impl_->next_id++;
    impl_->ids.emplace(id, bid);
    return id;
}

uint32_t PhysicsWorld::add_dynamic_cylinder_ccd(glm::vec3 c, float radius,
                                                float half_length,
                                                glm::quat orientation,
                                                glm::vec3 linvel, float mass) {
    // Jolt's CylinderShape requires its convex radius to be smaller than both
    // the cylinder's radius and half-length. Default convex radius is 0.05;
    // for a 6-cm bullet that's already too large.
    float convex_radius = std::min({0.02f, radius * 0.4f, half_length * 0.4f});
    JPH::CylinderShapeSettings ss(half_length, radius, convex_radius);
    ss.SetEmbedded();
    JPH::ShapeSettings::ShapeResult sr = ss.Create();
    if (sr.HasError()) {
        log::errorf("[jolt] dyn cylinder shape: %s", sr.GetError().c_str());
        return 0;
    }
    JPH::Quat jq(orientation.x, orientation.y, orientation.z, orientation.w);

    JPH::BodyCreationSettings bcs(sr.Get(),
                                  JPH::RVec3(c.x, c.y, c.z), jq,
                                  JPH::EMotionType::Dynamic,
                                  Layers::MOVING);
    // *** Industry-standard CCD for fast small bodies. *** Jolt sweep-casts
    // the cylinder along its motion at the start of each step and reports the
    // first contact instead of integrating-then-resolving. Eliminates tunnel-
    // through against any thicker target.
    bcs.mMotionQuality = JPH::EMotionQuality::LinearCast;
    bcs.mLinearVelocity = JPH::Vec3(linvel.x, linvel.y, linvel.z);
    bcs.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
    bcs.mMassPropertiesOverride.mMass = mass;
    // Jolt's per-body default cap is 500 m/s — anything above asserts/clamps
    // on creation. Raise it so the bullet-speed slider's 600 m/s upper bound
    // lands inside the legal range with comfortable headroom.
    bcs.mMaxLinearVelocity = 1000.0f;

    JPH::BodyID bid = impl_->system.GetBodyInterface().CreateAndAddBody(
        bcs, JPH::EActivation::Activate);
    uint32_t id = impl_->next_id++;
    impl_->ids.emplace(id, bid);
    return id;
}

uint32_t PhysicsWorld::add_dynamic_sphere(glm::vec3 c, float radius,
                                          glm::vec3 linvel, float mass,
                                          float restitution, float friction) {
    JPH::SphereShapeSettings ss(radius);
    ss.SetEmbedded();
    JPH::ShapeSettings::ShapeResult sr = ss.Create();
    if (sr.HasError()) {
        log::errorf("[jolt] sphere shape: %s", sr.GetError().c_str());
        return 0;
    }
    JPH::BodyCreationSettings bcs(sr.Get(),
                                  JPH::RVec3(c.x, c.y, c.z),
                                  JPH::Quat::sIdentity(),
                                  JPH::EMotionType::Dynamic,
                                  Layers::MOVING);
    bcs.mLinearVelocity = JPH::Vec3(linvel.x, linvel.y, linvel.z);
    bcs.mRestitution = restitution;
    bcs.mFriction = friction;
    bcs.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
    bcs.mMassPropertiesOverride.mMass = mass;
    // Particles are tiny + numerous — Discrete motion quality is fine and
    // far cheaper than the LinearCast path used for bullets.
    bcs.mMotionQuality = JPH::EMotionQuality::Discrete;

    JPH::BodyID bid = impl_->system.GetBodyInterface().CreateAndAddBody(
        bcs, JPH::EActivation::Activate);
    uint32_t id = impl_->next_id++;
    impl_->ids.emplace(id, bid);
    return id;
}

void PhysicsWorld::remove_body(uint32_t id) {
    auto it = impl_->ids.find(id);
    if (it == impl_->ids.end()) return;
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    bi.RemoveBody(it->second);
    bi.DestroyBody(it->second);
    impl_->ids.erase(it);
}

PhysicsWorld::RayHit PhysicsWorld::raycast(glm::vec3 origin, glm::vec3 direction,
                                           float max_distance) const {
    RayHit out{};
    JPH::RRayCast ray(JPH::RVec3(origin.x, origin.y, origin.z),
                      JPH::Vec3(direction.x, direction.y, direction.z) * max_distance);
    JPH::RayCastResult res;
    bool hit = impl_->system.GetNarrowPhaseQuery().CastRay(ray, res);
    if (!hit) return out;

    out.hit = true;
    out.distance = res.mFraction * max_distance;
    out.position = origin + direction * out.distance;

    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    JPH::BodyLockRead lock(impl_->system.GetBodyLockInterface(), res.mBodyID);
    if (lock.Succeeded()) {
        const JPH::Body& body = lock.GetBody();
        JPH::Vec3 n = body.GetWorldSpaceSurfaceNormal(res.mSubShapeID2,
                                                      JPH::RVec3(out.position.x,
                                                                 out.position.y,
                                                                 out.position.z));
        out.normal = glm::vec3(n.GetX(), n.GetY(), n.GetZ());
        if (body.GetMotionType() == JPH::EMotionType::Dynamic) {
            out.dynamic = true;
            for (const auto& [id, bid] : impl_->ids) {
                if (bid == res.mBodyID) { out.body_id = id; break; }
            }
        }
    }
    (void)bi;
    return out;
}

void PhysicsWorld::apply_impulse(uint32_t id, glm::vec3 impulse) {
    auto it = impl_->ids.find(id);
    if (it == impl_->ids.end()) return;
    impl_->system.GetBodyInterface().AddImpulse(
        it->second, JPH::Vec3(impulse.x, impulse.y, impulse.z));
}

// Read-only queries use GetBodyInterfaceNoLock(): all our physics
// mutations (Update, AddImpulse, remove_body, spawn) happen on the main
// thread between Update() calls, so no other thread is racing with these
// reads. GetBodyInterface() takes a body mutex per call, which at 100+
// bodies × multiple queries/frame is a measurable fraction of CPU time.

glm::vec3 PhysicsWorld::get_linear_velocity(uint32_t id) const {
    auto it = impl_->ids.find(id);
    if (it == impl_->ids.end()) return glm::vec3(0.0f);
    JPH::Vec3 v = impl_->system.GetBodyInterfaceNoLock().GetLinearVelocity(it->second);
    return glm::vec3(v.GetX(), v.GetY(), v.GetZ());
}

bool PhysicsWorld::is_body_active(uint32_t id) const {
    auto it = impl_->ids.find(id);
    if (it == impl_->ids.end()) return false;
    return impl_->system.GetBodyInterfaceNoLock().IsActive(it->second);
}

bool PhysicsWorld::get_body_world_matrix(uint32_t id, glm::mat4& out) const {
    auto it = impl_->ids.find(id);
    if (it == impl_->ids.end()) return false;
    // Single GetWorldTransform call returns position+rotation as one
    // RMat44, vs separate GetPosition + GetRotation which lock the body
    // mutex twice.
    JPH::RMat44 m = impl_->system.GetBodyInterfaceNoLock().GetWorldTransform(it->second);
    out = glm::mat4(
        m(0,0), m(1,0), m(2,0), m(3,0),
        m(0,1), m(1,1), m(2,1), m(3,1),
        m(0,2), m(1,2), m(2,2), m(3,2),
        m(0,3), m(1,3), m(2,3), m(3,3));
    return true;
}

// --- BodyHandle fast-path: caller stores the cached BodyID once, all
// subsequent reads skip the unordered_map.find. JPH::BodyID's underlying
// storage is a uint32 (index + sequence), so we can just round-trip it.

PhysicsWorld::BodyHandle PhysicsWorld::handle_of(uint32_t id) const {
    auto it = impl_->ids.find(id);
    if (it == impl_->ids.end()) return 0;
    return it->second.GetIndexAndSequenceNumber();
}

bool PhysicsWorld::get_body_world_matrix_h(BodyHandle h, glm::mat4& out) const {
    if (h == 0) return false;
    JPH::BodyID bid(h);
    JPH::RMat44 m = impl_->system.GetBodyInterfaceNoLock().GetWorldTransform(bid);
    out = glm::mat4(
        m(0,0), m(1,0), m(2,0), m(3,0),
        m(0,1), m(1,1), m(2,1), m(3,1),
        m(0,2), m(1,2), m(2,2), m(3,2),
        m(0,3), m(1,3), m(2,3), m(3,3));
    return true;
}

bool PhysicsWorld::is_body_active_h(BodyHandle h) const {
    if (h == 0) return false;
    return impl_->system.GetBodyInterfaceNoLock().IsActive(JPH::BodyID(h));
}

glm::vec3 PhysicsWorld::get_linear_velocity_h(BodyHandle h) const {
    if (h == 0) return glm::vec3(0.0f);
    JPH::Vec3 v = impl_->system.GetBodyInterfaceNoLock().GetLinearVelocity(JPH::BodyID(h));
    return glm::vec3(v.GetX(), v.GetY(), v.GetZ());
}

void PhysicsWorld::apply_impulse_h(BodyHandle h, glm::vec3 impulse) {
    if (h == 0) return;
    impl_->system.GetBodyInterface().AddImpulse(
        JPH::BodyID(h), JPH::Vec3(impulse.x, impulse.y, impulse.z));
}

} // namespace qlike
