###############################################################################
##
## CMakeLists file for unit test
##
###############################################################################

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

SET(SOURCES
  t-init
  t-hello
  t-eigenproblem
  t-ex1
  )

FOREACH(current ${SOURCES})
  ADD_EXECUTABLE(${current} ${current}.cpp ${COMMON_SOURCES})
  ADD_TEST(${current} ${current})
  TARGET_LINK_LIBRARIES(${current} ${PROJECT_NAME} petsc_cxx ${PETSC_LIBRARIES} ${SLEPC_LIBRARIES})
  INSTALL(TARGETS ${current} RUNTIME DESTINATION share/${PROJECT_NAME}/test COMPONENT test)
ENDFOREACH()

######################################################################################
