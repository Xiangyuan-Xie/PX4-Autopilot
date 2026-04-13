############################################################################
#
#   Copyright (c) 2026.
#
############################################################################

add_compile_options(-Wno-float-equal)
get_directory_property(FLAGS COMPILE_OPTIONS)
list(REMOVE_ITEM FLAGS "-Wcast-align")
set_directory_properties(PROPERTIES COMPILE_OPTIONS "${FLAGS}")

px4_add_library(rl_mc_arm_control_manual_reference
	rl_mc_arm_control_manual_reference.cpp
)
target_link_libraries(rl_mc_arm_control_manual_reference PUBLIC mathlib motion_planning)
target_include_directories(rl_mc_arm_control_manual_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

px4_add_module(
	MODULE rl_mc_arm_control
	MAIN rl_mc_arm_control
	COMPILE_FLAGS
	SRCS
		rl_mc_arm_control.cpp
		rl_mc_arm_control.hpp
		rl_mc_arm_control_manual_reference.hpp
		rl_tools_adapter.cpp
		rl_tools_adapter.hpp
	DEPENDS
		px4_work_queue
		mathlib
		motion_planning
		rl_mc_arm_control_manual_reference
)
