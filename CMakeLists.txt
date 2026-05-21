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
        $<$<CONFIG:Release>:-flto>
    )
    # AVX2 baseline only on x86_64 — Apple Silicon (arm64) doesn't have it.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        add_compile_options($<$<CONFIG:Release>:-march=x86-64-v3>)
    endif()
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

# AMD FidelityFX-FSR2 — vendored as a git submodule under
# external/FidelityFX-FSR2 (MIT license; LICENSE.txt preserved). Use as-is
# via the SDK's public API; no port, no clean-room reimplementation.
#
# Phase 2 (build wiring):
#   - DX12 backend disabled; we only need Vulkan.
#   - FSR2's CMakeLists hard-FATAL_ERRORs unless CMAKE_GENERATOR_PLATFORM
#     or CMAKE_EXE_LINKER_FLAGS picks x64/Win32 — set the linker-flag
#     hint temporarily to satisfy that check under Ninja, then restore.
#   - EXCLUDE_FROM_ALL so we only build it when something links to it.
#   - Auto-shader-compilation calls FidelityFX_SC.exe (bundled at
#     external/FidelityFX-FSR2/tools/sc/), which then invokes the
#     glslangValidator next to it — both are in the submodule, no
#     external DXC dependency required for the Vulkan path.
if(EXISTS "${CMAKE_SOURCE_DIR}/external/FidelityFX-FSR2/src/ffx-fsr2-api/CMakeLists.txt")
    set(FFX_FSR2_API_DX12 OFF CACHE BOOL "" FORCE)
    set(FFX_FSR2_API_VK   ON  CACHE BOOL "" FORCE)
    set(FSR2_BUILD_AS_DLL OFF CACHE BOOL "" FORCE)

    # Satisfy FSR2's platform check (FATAL_ERROR on non-x64) — we use
    # Ninja so CMAKE_GENERATOR_PLATFORM is unset; spoof via linker flag.
    set(_qlike_orig_exe_linker_flags "${CMAKE_EXE_LINKER_FLAGS}")
    if(NOT CMAKE_GENERATOR_PLATFORM)
        set(CMAKE_EXE_LINKER_FLAGS "/machine:x64")
    endif()

    add_subdirectory(external/FidelityFX-FSR2/src/ffx-fsr2-api EXCLUDE_FROM_ALL)

    # Restore — the FSR2 CMake only reads CMAKE_EXE_LINKER_FLAGS at
    # configure-time inside its top-level if() block.
    set(CMAKE_EXE_LINKER_FLAGS "${_qlike_orig_exe_linker_flags}")
else()
    message(STATUS "FidelityFX-FSR2 submodule not present — run "
                   "`git submodule update --init --recursive` to enable.")
endif()

# AMD FidelityFX SDK (FSR3 + Frame Generation) — vendored at
# external/FidelityFX-SDK (MIT, LICENSE.txt preserved). The v1.1.4 SDK
# ships a stable C ABI behind a prebuilt DLL (amd_fidelityfx_vk.dll +
# amd_fidelityfx_vk.lib in PrebuiltSignedDLL/). We link the import lib,
# include the ffx-api headers, and copy the DLL alongside the exe at
# build time. No add_subdirectory needed — the sample-framework CMake
# in the SDK isn't designed for drop-in use.
#
# The DLL is the FIRST runtime dependency in the project; everything
# else is statically linked. This is the cost of FSR3's chosen ABI.
set(FFX_SDK_ROOT "${CMAKE_SOURCE_DIR}/external/FidelityFX-SDK")
if(EXISTS "${FFX_SDK_ROOT}/PrebuiltSignedDLL/amd_fidelityfx_vk.lib")
    add_library(ffx_sdk_vk SHARED IMPORTED)
    set_target_properties(ffx_sdk_vk PROPERTIES
        IMPORTED_LOCATION "${FFX_SDK_ROOT}/PrebuiltSignedDLL/amd_fidelityfx_vk.dll"
        IMPORTED_IMPLIB   "${FFX_SDK_ROOT}/PrebuiltSignedDLL/amd_fidelityfx_vk.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${FFX_SDK_ROOT}/ffx-api/include")
    message(STATUS "FidelityFX SDK 1.1.4 prebuilt VK lib + headers found")
else()
    message(STATUS "FidelityFX SDK not present at external/FidelityFX-SDK — "
                   "FSR3 disabled. Run `git submodule update --init --recursive` "
                   "and check out v1.1.4 in the submodule.")
endif()

# tinyexr enables -Weverything -Werror; clang on Apple Silicon flags
# /usr/local/include via -Wpoison-system-directories. Disable warnings for it.
if(TARGET tinyexr)
    target_compile_options(tinyexr PRIVATE -w)
endif()
if(TARGET miniz)
    target_compile_options(miniz PRIVATE -w)
endif()

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
    src/engine/vk_engine/bloom.cpp
    src/engine/vk_engine/taa.cpp
    src/engine/vk_engine/fsr2.cpp
    src/engine/vk_engine/fsr3.cpp
    src/engine/vk_engine/restir.cpp
    src/engine/vk_engine/sun_shadow.cpp
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

# FSR2 — link only if the submodule was processed above.
if(TARGET ffx_fsr2_api_x64)
    target_link_libraries(qlike_engine PUBLIC ffx_fsr2_api_x64)
    target_include_directories(qlike_engine PRIVATE
        ${CMAKE_SOURCE_DIR}/external/FidelityFX-FSR2/src/ffx-fsr2-api)
    target_compile_definitions(qlike_engine PUBLIC QLIKE_HAVE_FSR2=1)
endif()
if(TARGET ffx_fsr2_api_vk_x64)
    target_link_libraries(qlike_engine PUBLIC ffx_fsr2_api_vk_x64)
endif()
if(TARGET ffx_sdk_vk)
    # Runtime LoadLibrary path — the SDK's headers hardcode FFX_API_ENTRY
    # as __declspec(dllexport), so static linkage to the import lib
    # produces NULL function pointers in the consumer. AMD's recommended
    # consumer integration is ffx_api_loader.h + GetProcAddress at
    # runtime. So we DON'T link the .lib here, only pull in the headers
    # and rely on the staged DLL alongside the exe.
    target_include_directories(qlike_engine PUBLIC
        $<TARGET_PROPERTY:ffx_sdk_vk,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(qlike_engine PUBLIC QLIKE_HAVE_FSR3=1)
    # Compile fsr3.cpp with /EHa so __try/__except can catch hardware
    # AVs (NULL function calls, bad ptr derefs) coming back from the
    # SDK DLL. /EHsc (the project default) only catches synchronous
    # C++ exceptions and would let the SDK's AVs kill the process
    # before our SEH handler can flip the fatal flag.
    set_source_files_properties(src/engine/vk_engine/fsr3.cpp
        PROPERTIES COMPILE_FLAGS "/EHa")
endif()
if(WIN32)
    target_compile_definitions(qlike_engine PUBLIC VK_USE_PLATFORM_WIN32_KHR)
elseif(APPLE)
    target_compile_definitions(qlike_engine PUBLIC VK_USE_PLATFORM_METAL_EXT)
elseif(UNIX)
    target_compile_definitions(qlike_engine PUBLIC VK_USE_PLATFORM_XLIB_KHR)
endif()

add_executable(quake_like src/main.cpp)
target_link_libraries(quake_like PRIVATE qlike_engine)

# Stage the FSR3 DLL next to quake_like.exe so it loads at process start.
# Must be on the exe target (not qlike_engine, which is a static lib —
# its POST_BUILD doesn't carry over to the exe's output dir).
if(TARGET ffx_sdk_vk)
    add_custom_command(TARGET quake_like POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FFX_SDK_ROOT}/PrebuiltSignedDLL/amd_fidelityfx_vk.dll"
            "$<TARGET_FILE_DIR:quake_like>/amd_fidelityfx_vk.dll"
        COMMENT "Staging amd_fidelityfx_vk.dll alongside quake_like.exe")
endif()
target_compile_definitions(qlike_engine PRIVATE
    QLIKE_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders"
)

# --- Shader compilation ---
file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/shaders/*.vert"
    "${CMAKE_SOURCE_DIR}/shaders/*.frag"
    "${CMAKE_SOURCE_DIR}/shaders/*.tesc"
    "${CMAKE_SOURCE_DIR}/shaders/*.tese"
    "${CMAKE_SOURCE_DIR}/shaders/*.comp")
# (grass_raymarch.frag picked up by the glob above)

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
