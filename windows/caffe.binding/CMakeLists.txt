if(NOT HAVE_BINGING)
  message(STATUS "caffe binding interface is disabled. Building without it...")
  return()
endif()

# ---[ caffe_binding project
project(caffe_binding C CXX)

file(GLOB_RECURSE caffe_binding_srcs ${PROJECT_SOURCE_DIR}/*.cpp)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_library(caffe_binding SHARED ${caffe_binding_srcs})
caffe_default_properties(caffe_binding)
set_target_properties(caffe_binding PROPERTIES PREFIX "" OUTPUT_NAME "caffe_binding")
target_link_libraries(caffe_binding PUBLIC ${Caffe_LINK})

if(UNIX OR APPLE)
    set(__linkname "${PROJECT_SOURCE_DIR}/caffe_binding.so")
    add_custom_command(TARGET caffe_binding POST_BUILD
                       COMMAND ln -sf $<TARGET_LINKER_FILE:caffe_binding> "${__linkname}"
                       COMMENT "Creating symlink ${__linkname} -> ${PROJECT_BINARY_DIR}/lib/caffe_binding${Caffe_POSTFIX}.so")
endif()

# ---[ Install
# caffe_binding.so
install(TARGETS caffe_binding  DESTINATION caffe_binding)

