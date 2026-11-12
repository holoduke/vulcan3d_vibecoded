#pragma once

// Parametric, stochastic, bracketed L-systems with a 3D turtle
// (Prusinkiewicz & Lindenmayer, "The Algorithmic Beauty of Plants").
//
// A plant is a string of modules (symbol + up to three parameters). Each
// derivation step rewrites every module in parallel through the rule
// registered for its symbol; rules are C++ callables, so they can test
// parameters (parametric), draw random numbers (stochastic) and emit any
// successor string. The turtle then walks the final string:
//
//   F(l)   move forward l drawing a segment      f(l)  move without drawing
//   +(a) -(a)   turn left / right  (about U)      &(a) ^(a)  pitch down / up (about L)
//   \(a) /(a)   roll left / right  (about H)      |          turn around
//   [ ]    push / pop state                       !(w)  set the branch radius
//   $      roll until the left vector is horizontal (keeps leaves level)
//
// Angles are degrees. Any other symbol listed in `organs` is reported with
// the turtle frame (leaf, flower, needle spray ...); the caller turns
// segments and organs into geometry. After every F the heading bends
// toward `tropism` (gravity sag, or light seeking with a +Y vector).

#include <glm/glm.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace qlike::lsys {

struct Module {
    char  s = 'F';
    float a = 0.0f, b = 0.0f, c = 0.0f;
};
using Str = std::vector<Module>;

struct Rng {
    uint32_t st = 1u;
    float u() {                                   // 0..1
        st = st * 1664525u + 1013904223u;
        return float((st >> 8) & 0xFFFFFFu) / 16777215.0f;
    }
    float range(float lo, float hi) { return lo + (hi - lo) * u(); }
    float jit(float v, float rel) { return v * (1.0f + rel * (2.0f * u() - 1.0f)); }
};

using Rule = std::function<void(const Module&, Rng&, Str&)>;

struct Grammar {
    Str axiom;
    std::unordered_map<char, Rule> rules;

    Str derive(int steps, uint32_t seed) const {
        Rng rng{ seed * 2654435761u + 12345u };
        Str cur = axiom, next;
        for (int i = 0; i < steps; ++i) {
            next.clear();
            next.reserve(cur.size() * 2);
            for (const Module& m : cur) {
                auto it = rules.find(m.s);
                if (it == rules.end()) next.push_back(m);
                else it->second(m, rng, next);
            }
            cur.swap(next);
        }
        return cur;
    }
};

struct Segment {
    glm::vec3 a{0.0f}, b{0.0f};
    float r0 = 0.01f, r1 = 0.01f;
    int   depth = 0;          // bracket nesting level
    float d0 = 0.0f, d1 = 0.0f;   // path length from the root at a / b
};
struct Organ {
    char s = 'L';
    glm::vec3 pos{0.0f}, H{0.0f, 1.0f, 0.0f}, L{1.0f, 0.0f, 0.0f}, U{0.0f, 0.0f, 1.0f};
    float a = 0.0f, b = 0.0f, c = 0.0f;
    int   depth = 0;
    float dist = 0.0f;        // path length from the root
    float radius = 0.0f;      // branch radius where it sits
};
struct Shape {
    std::vector<Segment> segs;
    std::vector<Organ>   organs;
};

struct TurtleParams {
    float angle = 25.0f;              // default for + - & ^ \ / without a parameter
    float radius = 0.02f;             // initial branch radius
    float taper = 1.0f;               // radius multiplier along each F
    glm::vec3 tropism{0.0f, -1.0f, 0.0f};
    float bend = 0.0f;                // tropism susceptibility (per unit length)
    std::string organs;               // symbols reported as organs
};

inline Shape interpret(const Str& str, const TurtleParams& tp) {
    struct State { glm::vec3 p, H, L, U; float r; int depth; float dist; };
    State t{ glm::vec3(0.0f), glm::vec3(0, 1, 0), glm::vec3(-1, 0, 0), glm::vec3(0, 0, 1), tp.radius, 0, 0.0f };
    std::vector<State> stack;
    Shape out;
    const float kRad = 0.017453292f;
    auto rot = [](glm::vec3& x, glm::vec3& y, float ang) {     // rotate x toward y
        const float c = std::cos(ang), s = std::sin(ang);
        const glm::vec3 nx = x * c + y * s, ny = y * c - x * s;
        x = nx; y = ny;
    };
    auto ang = [&](const Module& m) { return (m.a != 0.0f ? m.a : tp.angle) * kRad; };
    for (const Module& m : str) {
        switch (m.s) {
        case 'F': case 'f': {
            const float len = m.a;
            const glm::vec3 b = t.p + t.H * len;
            const float r1 = t.r * tp.taper;
            if (m.s == 'F') out.segs.push_back({ t.p, b, t.r, r1, t.depth, t.dist, t.dist + len });
            t.p = b; t.r = r1; t.dist += len;
            if (tp.bend != 0.0f) {
                const glm::vec3 perp = tp.tropism - t.H * glm::dot(tp.tropism, t.H);
                t.H = glm::normalize(t.H + perp * (tp.bend * len));
                t.L = glm::normalize(t.L - t.H * glm::dot(t.L, t.H));
                t.U = glm::cross(t.H, t.L);
            }
            break;
        }
        case '+': rot(t.H, t.L,  ang(m)); break;
        case '-': rot(t.H, t.L, -ang(m)); break;
        case '&': rot(t.H, t.U,  ang(m)); break;
        case '^': rot(t.H, t.U, -ang(m)); break;
        case '\\': rot(t.L, t.U,  ang(m)); break;
        case '/':  rot(t.L, t.U, -ang(m)); break;
        case '|': t.H = -t.H; t.L = -t.L; break;
        case '$': {
            const glm::vec3 v(0.0f, 1.0f, 0.0f);
            glm::vec3 l = glm::cross(v, t.H);
            if (glm::dot(l, l) > 1e-6f) { t.L = glm::normalize(l); t.U = glm::cross(t.H, t.L); }
            break;
        }
        case '!': t.r = m.a; break;
        case '[': stack.push_back(t); t.depth += 1; break;
        case ']': if (!stack.empty()) { t = stack.back(); stack.pop_back(); } break;
        default:
            if (tp.organs.find(m.s) != std::string::npos)
                out.organs.push_back({ m.s, t.p, t.H, t.L, t.U, m.a, m.b, m.c, t.depth, t.dist, t.r });
            break;
        }
    }
    return out;
}

}  // namespace qlike::lsys
