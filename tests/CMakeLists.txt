add_executable(qlike_tests
    test_physics.cpp
    test_collision.cpp
    test_frustum.cpp
)

target_link_libraries(qlike_tests PRIVATE
    qlike_core
    Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(qlike_tests)
