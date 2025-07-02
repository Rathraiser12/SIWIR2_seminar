cmake_minimum_required(VERSION 3.16)
project(cylinder_eigen LANGUAGES CXX)

# default to Release if not set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(deal.II 9 REQUIRED COMPONENTS PETSc SLEPc MPI)
deal_ii_initialize_cached_variables()

add_executable(cylinder_eigen cylinder_eigen.cc)
deal_ii_setup_target(cylinder_eigen)