###############################################################################
##
## CMakeLists file for unit test
##
###############################################################################

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

SET(SOURCES
  #t-init
  #t-hello
  #t-eigenproblem
  t-ex1
  t-arnoldi
  t-slepc-ex1
  # t-slepc-ex2
  # t-slepc-ex3
  # t-slepc-ex4
  # t-slepc-ex5
  # t-slepc-ex7
  # t-slepc-ex8
  # t-slepc-ex9
  # t-slepc-ex10
  # t-slepc-ex11
  # t-slepc-ex12
  # t-slepc-ex13
  # t-slepc-ex14
  # t-slepc-ex15
  # t-slepc-ex16
  # t-slepc-ex17
  # t-slepc-ex18
  )

FOREACH(current ${SOURCES})
  ADD_EXECUTABLE(${current} ${current}.cpp ${COMMON_SOURCES})
  ADD_TEST(${current} ${current})
  TARGET_LINK_LIBRARIES(${current} ${PROJECT_NAME} petsc_cxx ${PETSC_LIBRARIES} ${SLEPC_LIBRARIES})
  INSTALL(TARGETS ${current} RUNTIME DESTINATION share/${PROJECT_NAME}/test COMPONENT test)
ENDFOREACH()

######################################################################################
