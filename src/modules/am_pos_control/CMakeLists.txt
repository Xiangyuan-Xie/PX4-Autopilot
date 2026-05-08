############################################################################
#
#   Copyright (c) 2026.
#
############################################################################

add_compile_options(-Wno-float-equal)
get_directory_property(FLAGS COMPILE_OPTIONS)
list(REMOVE_ITEM FLAGS "-Wcast-align")
set_directory_properties(PROPERTIES COMPILE_OPTIONS "${FLAGS}")

px4_add_module(
	MODULE am_pos_control
	MAIN am_pos_control
	COMPILE_FLAGS
	SRCS
		am_pos_control.cpp
		am_pos_control.hpp
		rl_tools_adapter.cpp
		rl_tools_adapter.hpp
	DEPENDS
		px4_work_queue
		mathlib
)

target_include_directories(am_pos_control PRIVATE ${PX4_SOURCE_DIR}/third_party/rl_tools/include)
target_compile_features(am_pos_control PRIVATE cxx_std_17)
target_compile_options(am_pos_control PRIVATE -Wno-unused-local-typedefs)

add_library(am_pos_control_policy_blob INTERFACE)
target_include_directories(am_pos_control_policy_blob INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}
	${PX4_SOURCE_DIR}/third_party/rl_tools/include
)
target_compile_features(am_pos_control_policy_blob INTERFACE cxx_std_17)
target_compile_options(am_pos_control_policy_blob INTERFACE -Wno-unused-local-typedefs)

px4_add_unit_gtest(SRC am_pos_control_policy_blob_test.cpp LINKLIBS am_pos_control_policy_blob)
px4_add_unit_gtest(SRC am_pos_control_test.cpp LINKLIBS am_pos_control_policy_blob)
