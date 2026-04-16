############################################################################
#
#   Copyright (c) 2026.
#
############################################################################

px4_add_library(FlightTaskManualAmPosition
	FlightTaskManualAmPosition.cpp
	FlightTaskManualAmPosition.hpp
)

target_link_libraries(FlightTaskManualAmPosition
	PRIVATE
		WeatherVane
	PUBLIC
		FlightTask
		Sticks
)

target_include_directories(FlightTaskManualAmPosition PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
