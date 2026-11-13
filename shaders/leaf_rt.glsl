// Leaf-card cut-out for inline ray queries. Include after defining
// LEAF_ATLAS (a sampler2D holding the 2x2 leaf atlas).
//
// The vegetation BLAS keeps its leaf cards in a NON-OPAQUE geometry (two
// triangles per card, corners in a fixed order), so the card uv follows
// from the triangle's parity and the barycentrics; the instance's SBT
// record offset carries the atlas cell + 1 (0 = not a card instance).
// Every other geometry is flagged opaque and never reaches the loop body.
#ifndef QLIKE_LEAF_RT_GLSL
#define QLIKE_LEAF_RT_GLSL
bool leaf_card_hit(rayQueryEXT rq) {
    uint cell1 = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(rq, false);
    if (cell1 == 0u) return true;
    vec2 b = rayQueryGetIntersectionBarycentricsEXT(rq, false);
    vec2 uv = ((rayQueryGetIntersectionPrimitiveIndexEXT(rq, false) & 1) == 0) ? vec2(b.x + b.y, b.y)
                                                                              : vec2(b.x, b.x + b.y);
    float cell = float(cell1 - 1u);
    vec2 auv = vec2(mod(cell, 2.0) * 0.5 + clamp(uv.x, 0.006, 0.994) * 0.5,
                    (cell >= 2.0 ? 0.5 : 0.0) + (1.0 - clamp(uv.y, 0.006, 0.994)) * 0.5);
    return textureLod(LEAF_ATLAS, auv, 1.5).a > 0.33;
}
// Drives a query to completion, alpha-testing leaf cards (use in place of
// `while (rayQueryProceedEXT(rq)) {}` for queries WITHOUT gl_RayFlagsOpaqueEXT).
#define RQ_PROCEED_LEAVES(rq) \
    while (rayQueryProceedEXT(rq)) { \
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT && \
            leaf_card_hit(rq)) rayQueryConfirmIntersectionEXT(rq); \
    }
#endif
