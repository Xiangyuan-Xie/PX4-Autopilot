############################################################################
#
#   Copyright (c) 2026.
#
############################################################################

px4_add_library(FlightTaskManualAmPose
	FlightTaskManualAmPose.cpp
	FlightTaskManualAmPose.hpp
)

target_link_libraries(FlightTaskManualAmPose
	PRIVATE
		WeatherVane
	PUBLIC
		FlightTask
		Sticks
)

target_include_directories(FlightTaskManualAmPose PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
