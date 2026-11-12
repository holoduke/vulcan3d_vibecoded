#include "engine/eztree.h"

#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

#undef min
#undef max

namespace qlike {
namespace {

constexpr float kPi = 3.14159265358979f;

struct Rng {
    uint32_t s;
    float u() { s = s * 1664525u + 1013904223u; return float((s >> 8) & 0xFFFFFFu) / 16777216.0f; }
    // ez-tree's rng.random(max, min).
    float random(float mx = 1.0f, float mn = 0.0f) { return mn + (mx - mn) * u(); }
};

struct Section { glm::vec3 origin; glm::quat q; float radius; };
struct Branch { glm::vec3 origin; glm::quat q; float length, radius; int level, sections, segments; float prio; };
struct SkBranch { std::vector<Section> sections; int segments; int level; float prio; };
struct Leaf { glm::vec3 origin; glm::quat q; float size; float prio; };

glm::quat axis_q(glm::vec3 axis, float ang) { return glm::angleAxis(ang, axis); }

struct Gen {
    const EzTreeOptions& o;
    Rng rng;
    std::deque<Branch> queue;
    std::vector<SkBranch> branches;
    std::vector<Leaf> leaves;

    Rng prng{ 0x51ED270Bu };          // priorities: separate stream, the shape stays that of the seed
    float cur_prio = 0.0f;            // priority of the branch being grown
    void record_leaf(glm::vec3 origin, glm::quat q) {
        const float size = o.leaf_size * (1.0f + rng.random(o.leaf_size_variance, -o.leaf_size_variance));
        leaves.push_back({ origin, q, size, prng.u() });   // uniform: the leaf count is proportional to the detail
    }
    // Children appear after their parent; thin orders come in late.
    float child_prio(int level) {
        const float lo = o.evergreen ? 0.0f : (level == 1 ? 0.0f : level == 2 ? 0.08f : 0.18f);
        const float hi = o.evergreen ? 0.85f : (level == 1 ? 0.10f : level == 2 ? 0.32f : 0.55f);
        return std::min(0.9f, cur_prio + lo + (hi - lo) * prng.u());
    }
    std::vector<int> shuffled(int count) {
        std::vector<int> a(static_cast<size_t>(count));
        for (int k = 0; k < count; ++k) a[size_t(k)] = k;
        for (int k = count - 1; k > 0; --k) std::swap(a[size_t(k)], a[size_t(std::min(k, int(rng.random() * float(k + 1))))]);
        return a;
    }
    // Position / orientation / radius at fraction `f` along a branch.
    void at(const std::vector<Section>& sec, float f, glm::vec3& origin, glm::quat& q, float& radius) {
        const int last = int(sec.size()) - 1;
        const int idx = std::clamp(int(std::floor(f * float(last))), 0, last);
        const Section& a = sec[size_t(idx)];
        const Section& b = sec[size_t(std::min(idx + 1, last))];
        const float alpha = std::clamp(f * float(last) - float(idx), 0.0f, 1.0f);
        origin = glm::mix(a.origin, b.origin, alpha);
        radius = (1.0f - alpha) * a.radius + alpha * b.radius;
        q = glm::slerp(a.q, b.q, alpha);
    }

    void grow(const Branch& br) {
        glm::quat q = br.q;
        glm::vec3 origin = br.origin;
        const float section_len = br.length / float(br.sections);
        std::vector<Section> sec;
        for (int i = 0; i <= br.sections; ++i) {
            float r = br.radius;
            const float f = float(i) / float(br.sections);
            if (i == br.sections && br.level == o.levels) r = 0.001f;
            else if (!o.evergreen) r *= 1.0f - o.taper[br.level] * f;
            else r *= 1.0f - f;                              // evergreens taper to a point
            sec.push_back({ origin, q, r });
            origin += q * glm::vec3(0.0f, section_len, 0.0f);

            // Random perturbation, stronger on thin branches.
            const float g = std::max(1.0f, 1.0f / std::sqrt(std::max(r, 1e-4f))) * o.gnarliness[br.level];
            q = q * axis_q(glm::vec3(1, 0, 0), rng.random(g, -g)) * axis_q(glm::vec3(0, 0, 1), rng.random(g, -g));
            q = q * axis_q(glm::vec3(0, 1, 0), o.twist[br.level]);
            // Growth force: rotate the growth direction toward (or away from)
            // force_dir by strength / radius, never past it.
            const glm::vec3 up = q * glm::vec3(0, 1, 0);
            const glm::vec3 target = glm::normalize(o.force_dir);
            glm::vec3 axis = glm::cross(up, target);
            const float sin_full = glm::length(axis);
            if (sin_full > 1e-6f) {
                axis /= sin_full;
                const float full = std::atan2(sin_full, glm::dot(up, target));
                const float step = o.force_strength / std::max(r, 1e-3f);
                q = axis_q(axis, std::clamp(step, -full, full)) * q;
            }
            q = glm::normalize(q);
        }
        branches.push_back({ sec, br.segments, br.level, br.prio });
        cur_prio = br.prio;

        // Deciduous: a terminal branch continues from the end of the parent.
        if (!o.evergreen) {
            const Section& lastS = sec.back();
            if (br.level < o.levels)
                queue.push_back({ lastS.origin, lastS.q, o.length[br.level + 1], lastS.radius, br.level + 1,
                                  br.sections, br.segments, child_prio(br.level + 1) });
            else
                record_leaf(lastS.origin, lastS.q);
        }
        if (br.level == o.levels) gen_leaves(sec);
        else if (br.level < o.levels) gen_children(o.children[br.level], br.level + 1, sec);
    }

    void gen_children(int count, int level, const std::vector<Section>& sec) {
        if (count <= 0) return;
        const float radial_offset = rng.random();
        const float start_min = o.start[level];
        const float step = (1.0f - start_min) / float(count);
        const std::vector<int> slots = shuffled(count);
        for (int i = 0; i < count; ++i) {
            const float st = start_min + (float(i) + rng.random()) * step;
            glm::vec3 origin; glm::quat pq; float pr;
            at(sec, st, origin, pq, pr);
            const float jitter = rng.random(0.5f, -0.5f);
            const float radial = 2.0f * kPi * (radial_offset + (float(slots[size_t(i)]) + jitter) / float(count));
            const glm::quat cq = pq * axis_q(glm::vec3(0, 1, 0), radial) *
                                 axis_q(glm::vec3(1, 0, 0), glm::radians(o.angle[level]));
            const float len = o.length[level] * (o.evergreen ? 1.0f - st : 1.0f);
            queue.push_back({ origin, cq, len, o.radius[level] * pr, level, o.sections[level], o.segments[level], child_prio(level) });
        }
    }

    void gen_leaves(const std::vector<Section>& sec) {
        const float radial_offset = rng.random();
        const int count = o.leaf_count;
        if (count <= 0) return;
        const float step = (1.0f - o.leaf_start) / float(count);
        const std::vector<int> slots = shuffled(count);
        for (int i = 0; i < count; ++i) {
            const float st = o.leaf_start + (float(i) + rng.random()) * step;
            glm::vec3 origin; glm::quat pq; float pr;
            at(sec, st, origin, pq, pr);
            const float jitter = rng.random(0.5f, -0.5f);
            const float radial = 2.0f * kPi * (radial_offset + (float(slots[size_t(i)]) + jitter) / float(count));
            record_leaf(origin, pq * axis_q(glm::vec3(0, 1, 0), radial) *
                                     axis_q(glm::vec3(1, 0, 0), glm::radians(o.leaf_angle)));
        }
    }

    void run() {
        queue.push_back({ glm::vec3(0.0f), glm::quat(1, 0, 0, 0), o.length[0], o.radius[0], 0, o.sections[0], o.segments[0], 0.0f });
        while (!queue.empty()) {
            const Branch b = queue.front();
            queue.pop_front();
            grow(b);
        }
    }
};

void push(game::LevelMesh& m, glm::vec3 p, glm::vec3 n, float u, float v) {
    m.verts.push_back({ p, n, u, v });
    m.bmin = glm::min(m.bmin, p);
    m.bmax = glm::max(m.bmax, p);
}

}  // namespace

EzTreeOptions ez_preset_oak() {
    // presets/oak_medium.json
    EzTreeOptions o;
    o.seed = 35729; o.evergreen = false; o.levels = 3;
    const float angle[4] = { 0, 54, 58, 32 };           const int children[4] = { 6, 4, 3, 0 };
    const float gn[4] = { 0.0f, -0.1f, -0.15f, 0.09f };  const float len[4] = { 37.24f, 11.08f, 12.39f, 7.16f };
    const float rad[4] = { 1.41f, 0.9f, 0.69f, 1.19f };  const int sec[4] = { 8, 6, 3, 1 };
    const int seg[4] = { 7, 5, 3, 3 };                   const float st[4] = { 0, 0.49f, 0.06f, 0.12f };
    const float tp[4] = { 0.73f, 0.42f, 0.69f, 0.75f };  const float tw[4] = { -0.23f, 0.42f, 0, 0 };
    for (int i = 0; i < 4; ++i) {
        o.angle[i] = angle[i]; o.children[i] = children[i]; o.gnarliness[i] = gn[i]; o.length[i] = len[i];
        o.radius[i] = rad[i]; o.sections[i] = sec[i]; o.segments[i] = seg[i]; o.start[i] = st[i];
        o.taper[i] = tp[i]; o.twist[i] = tw[i];
    }
    o.force_strength = 0.02f;
    o.leaf_angle = 42; o.leaf_count = 18; o.leaf_start = 0.16f; o.leaf_size = 2.5f; o.leaf_size_variance = 0.7f;
    o.leaf_double = true; o.leaf_kind = 0.0f;
    return o;
}

EzTreeOptions ez_preset_pine() {
    // presets/pine_medium.json
    EzTreeOptions o;
    o.seed = 13977; o.evergreen = true; o.levels = 1;
    const float angle[4] = { 0, 110, 16, 60 };          const int children[4] = { 82, 3, 5, 0 };
    const float gn[4] = { 0.05f, 0.08f, 0, 0 };          const float len[4] = { 50, 23.87f, 14.08f, 1 };
    const float rad[4] = { 1.05f, 0.36f, 0.7f, 0.7f };   const int sec[4] = { 12, 10, 8, 6 };
    const int seg[4] = { 8, 6, 4, 3 };                   const float st[4] = { 0, 0.27f, 0.14f, 0.3f };
    for (int i = 0; i < 4; ++i) {
        o.angle[i] = angle[i]; o.children[i] = children[i]; o.gnarliness[i] = gn[i]; o.length[i] = len[i];
        o.radius[i] = rad[i]; o.sections[i] = sec[i]; o.segments[i] = seg[i]; o.start[i] = st[i];
        o.taper[i] = 0.7f; o.twist[i] = 0.0f;
    }
    o.force_strength = -0.003f;
    o.leaf_angle = 39; o.leaf_count = 30; o.leaf_start = 0.09f; o.leaf_size = 1.435f; o.leaf_size_variance = 0.201f;
    o.leaf_double = true; o.leaf_kind = 1.0f;
    return o;
}

void build_ez_tree(game::LevelMesh& m, const EzTreeOptions& o, float max_detail, float target_height) {
    Gen g{ o, Rng{ o.seed * 2654435761u + 977u }, {}, {}, {} };
    g.run();
    float top = 1e-3f;
    for (const auto& b : g.branches) for (const auto& s : b.sections) top = std::max(top, s.origin.y);
    for (const auto& l : g.leaves) top = std::max(top, l.origin.y);
    const float sc = target_height / top;
    auto qprio = [](float p) { return std::floor(glm::clamp(p, 0.0f, 1.0f) * 1023.0f); };

    // ---- Branches: tubes through the section rings ----
    const int stride = max_detail > 0.6f ? 1 : 2;
    for (const auto& b : g.branches) {
        if (b.prio > max_detail) continue;
        const int segs = std::max(3, int(std::lround(float(b.segments) * glm::mix(0.5f, 1.0f, max_detail))));
        std::vector<Section> ring;
        for (size_t i = 0; i < b.sections.size(); i += size_t(stride)) ring.push_back(b.sections[i]);
        if ((b.sections.size() - 1) % size_t(stride) != 0) ring.push_back(b.sections.back());
        if (ring.size() < 2) continue;
        const uint32_t base = uint32_t(m.verts.size());
        const float qp = qprio(b.prio);
        for (const Section& s : ring) {
            const glm::vec3 c = s.origin * sc;
            const float r = glm::clamp(std::max(s.radius, 0.004f) * sc, 0.002f, 0.98f);
            for (int j = 0; j < segs; ++j) {
                const float a = 2.0f * kPi * float(j) / float(segs);
                const glm::vec3 n = glm::normalize(s.q * glm::vec3(std::cos(a), 0.0f, std::sin(a)));
                push(m, c + n * r, n, qp + r, 8.0f);
            }
        }
        for (size_t i = 0; i + 1 < ring.size(); ++i)
            for (int j = 0; j < segs; ++j) {
                const uint32_t v1 = base + uint32_t(i) * uint32_t(segs) + uint32_t(j);
                const uint32_t v2 = base + uint32_t(i) * uint32_t(segs) + uint32_t((j + 1) % segs);
                const uint32_t v3 = v1 + uint32_t(segs), v4 = v2 + uint32_t(segs);
                m.idx.insert(m.idx.end(), { v1, v3, v2, v2, v3, v4 });
            }
    }

    // ---- Leaves: cards. A second, crossed card per leaf joins in at the
    // upper half of the detail range. ----
    const float mat = o.leaf_kind >= 1.0f ? 9.0f : 7.0f;
    for (const Leaf& l : g.leaves) {
        const float size = l.size * sc;
        const glm::vec3 origin = l.origin * sc;
        const int cards = o.leaf_double ? 2 : 1;
        for (int cIdx = 0; cIdx < cards; ++cIdx) {
            const float prio = cIdx == 0 ? l.prio : 0.5f + 0.5f * l.prio;
            if (prio > max_detail) continue;
            const float qp = qprio(prio) * 4.0f;
            const glm::quat q = l.q * axis_q(glm::vec3(0, 1, 0), cIdx == 0 ? 0.0f : kPi * 0.5f);
            const glm::vec3 x = q * glm::vec3(size * 0.5f, 0.0f, 0.0f), y = q * glm::vec3(0.0f, size, 0.0f);
            const uint32_t k = uint32_t(m.verts.size());
            // corner id = u + 2 v
            push(m, origin - x,     -x,     qp + 0.0f, mat);
            push(m, origin + x,      x,     qp + 1.0f, mat);
            push(m, origin - x + y, -x + y, qp + 2.0f, mat);
            push(m, origin + x + y,  x + y, qp + 3.0f, mat);
            m.idx.insert(m.idx.end(), { k, k + 1, k + 3, k, k + 3, k + 2 });
        }
    }
}

}  // namespace qlike
