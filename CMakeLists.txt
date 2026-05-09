cmake_minimum_required(VERSION 3.24)
project(quake_like CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:preprocessor)
    # /Zi + /DEBUG so the StackWalk-based crash dumper can resolve symbols
    # to file:line in Release. Without these we still get module+offset, but
    # not function names — useless for "where did this throw come from".
    add_compile_options(/Zi)
    add_link_options(/DEBUG)

    # Release-only: aggressive optimisation. /O2 is the default but we
    # explicitly opt into the additional flags below.
    #   /Ob3       max inlining (above /Ob2 default)
    #   /Oi        intrinsic functions (memcpy, sin, etc → inline asm)
    #   /Ot        favour speed over size
    #   /GL        whole-program optimisation (paired with /LTCG link)
    #   /Gy        function-level linking (lets /OPT:REF strip unused)
    #   /Gw        same for global data
    #   /arch:AVX2 enable AVX2 instructions (every x64 CPU since ~2013)
    #   /Qpar      auto-parallelise simple loops
    #
    # /fp:fast removed — bisected to be the cause of a sky-coloured
    # cyan-wash flicker on the user's GPU. /fp:fast lets MSVC reorder
    # FP ops + treat denormals as zero + assume operands are not NaN;
    # somewhere in the per-frame uniform / scene UBO compute, that
    # assumption produces a wrong value that scales the lit term
    # screen-wide. Keep IEEE-conformant FP for now; the perf delta vs
    # /fp:precise is marginal in this codebase.
    add_compile_options(
        $<$<CONFIG:Release>:/Ob3>
        $<$<CONFIG:Release>:/Oi>
        $<$<CONFIG:Release>:/Ot>
        $<$<CONFIG:Release>:/GL>
        $<$<CONFIG:Release>:/Gy>
        $<$<CONFIG:Release>:/Gw>
        $<$<CONFIG:Release>:/arch:AVX2>
        $<$<CONFIG:Release>:/Qpar>
    )
    # /DEBUG turns OPT:REF/ICF off by default; re-enable both so unused
    # functions + identical COMDATs get folded out.
    add_link_options(
        $<$<CONFIG:Release>:/LTCG>
        $<$<CONFIG:Release>:/OPT:REF>
        $<$<CONFIG:Release>:/OPT:ICF>
    )
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    # -ffast-math symmetrically dropped (matches MSVC /fp:fast removal).
    add_compile_options(
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=x86-64-v3>   # AVX2 baseline
        $<$<CONFIG:Release>:-flto>
    )
    add_link_options($<$<CONFIG:Release>:-flto>)
endif()

find_package(Vulkan REQUIRED COMPONENTS glslc)

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE)

FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
    GIT_SHALLOW TRUE)

FetchContent_Declare(vk_bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG v1.3.295
    GIT_SHALLOW TRUE)

FetchContent_Declare(vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.1.0
    GIT_SHALLOW TRUE)

FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE)

# Jolt Physics — used for non-player rigid bodies (the player is custom Quake-style).
set(TARGET_HELLO_WORLD OFF CACHE BOOL "" FORCE)
set(TARGET_PERFORMANCE_TEST OFF CACHE BOOL "" FORCE)
set(TARGET_SAMPLES OFF CACHE BOOL "" FORCE)
set(TARGET_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(TARGET_VIEWER OFF CACHE BOOL "" FORCE)
set(USE_STATIC_MSVC_RUNTIME_LIBRARY OFF CACHE BOOL "" FORCE)
set(INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
set(CROSS_PLATFORM_DETERMINISTIC OFF CACHE BOOL "" FORCE)
set(DOUBLE_PRECISION OFF CACHE BOOL "" FORCE)
# Skip Jolt's install/export plumbing — it tries to write JoltConfig-$config.cmake
# with a literal $config when its export() runs from inside FetchContent, which
# trips a "$<CONFIG:$Config>" generator-expression error during Generate.
set(ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(JoltPhysics
    GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
    GIT_TAG v5.2.0
    SOURCE_SUBDIR Build
    GIT_SHALLOW TRUE)

FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.5
    GIT_SHALLOW TRUE)

# tinygltf — header-only glTF 2.0 loader. Used to load the optional viewmodel
# asset (drop a .glb at assets/gun.glb to activate). Opted out of its loader
# for stb_image_write / json header generation; we only need the parser.
set(TINYGLTF_HEADER_ONLY ON CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_GL_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_VALIDATOR_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_BUILDER_EXAMPLE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG v2.9.3
    GIT_SHALLOW TRUE)

# tinyexr — OpenEXR loader. Used to load the HDR equirect skybox (.exr) so
# the sun pixel keeps its real radiance and blooms naturally instead of
# being a clamped LDR disc + procedural overlay. tinyexr ships its own CMake
# target that bundles miniz — we link it directly rather than re-instantiate
# TINYEXR_IMPLEMENTATION ourselves (which would also need miniz on our include
# path).
set(TINYEXR_BUILD_SAMPLE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(tinyexr
    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
    GIT_TAG v1.0.8
    GIT_SHALLOW TRUE)

# miniaudio — single-header audio engine (MIT-0). Built-in 3D spatial
# sources with WAV/MP3/FLAC decoding; Vorbis comes via the bundled
# stb_vorbis. We pull just the source tree and wire a header-only
# INTERFACE target ourselves; the implementation goes in vma_impl.cpp's
# sibling (audio.cpp) so MINIAUDIO_IMPLEMENTATION lives in exactly one TU.
FetchContent_Declare(miniaudio
    GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG 0.11.21
    GIT_SHALLOW TRUE)

# FastNoiseLite — single-header noise (MIT). Used by the procedural
# terrain generator (src/engine/terrain.cpp).
FetchContent_Declare(fastnoiselite
    GIT_REPOSITORY https://github.com/Auburn/FastNoiseLite.git
    GIT_TAG v1.1.1
    GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(glfw glm vk_bootstrap vma Catch2 JoltPhysics imgui tinygltf tinyexr miniaudio fastnoiselite)

# Dear ImGui ships without CMakeLists; build our own target from its sources.
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)

# qlike_core: pure logic with no Vulkan/GLFW deps. Tests link this directly.
add_library(qlike_core STATIC
    src/game/physics.cpp
    src/game/collision.cpp
    src/game/level.cpp
    src/game/player.cpp
    src/engine/frustum.cpp
)
target_include_directories(qlike_core PUBLIC src)
target_link_libraries(qlike_core PUBLIC glm::glm)
target_compile_definitions(qlike_core PUBLIC
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_ENABLE_EXPERIMENTAL
    GLM_FORCE_INLINE             # aggressive inline for hot paths
    # GLM_FORCE_INTRINSICS intentionally NOT defined — it routes vec
    # construction through SIMD intrinsics that MSVC can't evaluate at
    # constexpr (breaks `static constexpr glm::vec3` in player.h etc.).
    # The scalar path with /arch:AVX2 + /fp:fast gets auto-vectorised
    # by the compiler anyway.
)

# qlike_engine: rendering, windowing, Vulkan. Depends on qlike_core.
add_library(qlike_engine STATIC
    src/engine/window.cpp
    src/engine/vk_initializers.cpp
    src/engine/vk_engine.cpp
    src/engine/vk_engine/helpers.cpp
    src/engine/vk_engine/settings.cpp
    src/engine/vk_engine/descriptors.cpp
    src/engine/vk_engine/rt.cpp
    src/engine/vk_engine/post.cpp
    src/engine/vk_engine/combat.cpp
    src/engine/vk_engine/screenshot.cpp
    src/engine/vk_engine/setup.cpp
    src/engine/vk_engine/world.cpp
    src/engine/vk_ui.cpp
    src/engine/vk_pipelines.cpp
    src/engine/mesh.cpp
    src/engine/gltf_loader.cpp
    src/engine/skybox.cpp
    src/engine/texture.cpp
    src/engine/physics_world.cpp
    src/engine/vma_impl.cpp
    src/engine/log.cpp
    src/engine/crash_handler.cpp
    src/engine/audio.cpp
    src/engine/terrain.cpp
    src/engine/grass.cpp
)
target_include_directories(qlike_engine PUBLIC src)
target_include_directories(qlike_engine PRIVATE
    ${tinygltf_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}
    ${miniaudio_SOURCE_DIR}
    ${fastnoiselite_SOURCE_DIR}/Cpp)
target_link_libraries(qlike_engine PUBLIC
    qlike_core
    Vulkan::Vulkan
    glfw
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    Jolt
    imgui
    tinyexr
)
target_compile_definitions(qlike_engine PUBLIC
    VK_USE_PLATFORM_WIN32_KHR
)

add_executable(quake_like src/main.cpp)
target_link_libraries(quake_like PRIVATE qlike_engine)
target_compile_definitions(qlike_engine PRIVATE
    QLIKE_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders"
)

# --- Shader compilation ---
file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/shaders/*.vert"
    "${CMAKE_SOURCE_DIR}/shaders/*.frag"
    "${CMAKE_SOURCE_DIR}/shaders/*.comp")

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

set(COMPILED_SHADERS)
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPV_OUT "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPV_OUT}
        # -O = SPIR-V optimisation (perf preset). Enables dead-code
        # elimination, common-subexpression elim, loop unrolling.
        # Equivalent to running spirv-opt -O on the output.
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O ${SHADER} -o ${SPV_OUT}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader ${SHADER_NAME}"
        VERBATIM)
    list(APPEND COMPILED_SHADERS ${SPV_OUT})
endforeach()

if(COMPILED_SHADERS)
    add_custom_target(shaders ALL DEPENDS ${COMPILED_SHADERS})
    add_dependencies(quake_like shaders)
endif()

enable_testing()
add_subdirectory(tests)
