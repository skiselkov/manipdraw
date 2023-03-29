# Copyright 2023 Saso Kiselkov. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# “Software”), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

# WARNING: Don't forget to set CMAKE_OSX_SYSROOT to a suitable SDK, otherwise
# we will be missing IOSurface headers. Typical command line for that:
# cmake -DCMAKE_OSX_SYSROOT=/Applications/Xcode.app/Contents/Developer/\
#	Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk .

cmake_minimum_required(VERSION 3.9)
project(manipdraw)

if(WIN32)
	set(PLAT_SHORT "win64")
	set(PLAT_LONG "win-64")
	set(PLAT_ONLY "win")
	set(PLUGIN_BIN_OUTDIR "win_x64")
elseif(APPLE)
	set(PLAT_SHORT "mac64")
	set(PLAT_LONG "mac-64")
	set(PLAT_ONLY "mac")
	set(PLUGIN_BIN_OUTDIR "mac_x64")
else()
	set(PLAT_SHORT "lin64")
	set(PLAT_LONG "linux-64")
	set(PLAT_ONLY "linux")
	set(PLUGIN_BIN_OUTDIR "lin_x64")
endif()

set(LIBACFUTILS "${CMAKE_SOURCE_DIR}/libacfutils")
set(LIBRAIN "${CMAKE_SOURCE_DIR}/librain")
set(CGLM "${LIBACFUTILS}/cglm/cglm-0.7.9")
set(GLEW "${LIBACFUTILS}/glew/glew-1.13.0-${PLAT_LONG}/install")
set(XPSDK "${LIBACFUTILS}/SDK")
set(ZLIB "${LIBACFUTILS}/zlib/zlib-${PLAT_LONG}")

FILE(GLOB LIBRAIN_SRCS ${LIBRAIN}/src/obj8.c)
FILE(GLOB LIBRAIN_HDRS ${LIBRAIN}/src/obj8.h ${LIBRAIN}/src/glpriv.h)

set(ALL_SRC
    manipdraw.c
    ${LIBRAIN_SRCS}
    ${LIBRAIN_HDRS})
LIST(SORT ALL_SRC)

add_library(manipdraw SHARED ${ALL_SRC})

include_directories(manipdraw PUBLIC
    "${XPSDK}/CHeaders/XPLM"
    "${XPSDK}/CHeaders/Widgets"
    "${XPSDK}/CHeaders/Wrappers"
    "${CGLM}/include"
    "${LIBACFUTILS}/src"
    "${GLEW}/include"
    "${LIBRAIN}/src"
    )

#compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -W -Wall -Wextra -Werror \
    --std=c11 -Wno-missing-field-initializers -fvisibility=hidden")
if(UNIX AND NOT APPLE)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-format-truncation \
	    -fno-gnu-unique")
endif()

add_definitions(-D_GNU_SOURCE -D_FILE_OFFSET_BITS=64)
# Needed for C11's gmtime_s
add_definitions(-D__STDC_WANT_LIB_EXT1__=1)

add_definitions(-DXPLM200=1 -DXPLM210=1 -DXPLM300=1 -DXPLM301=1 -DXPLM302=1)
add_definitions(-DXPLM_DEPRECATED)
add_definitions(-DGLEW_BUILD=GLEW_STATIC -DGLEW_MX)
if(WIN32)
	add_definitions(-DAPL=0 -DIBM=1 -DLIN=0)
elseif(APPLE)
	add_definitions(-DAPL=1 -DIBM=0 -DLIN=0)
else()
	add_definitions(-DAPL=0 -DIBM=0 -DLIN=1)
endif()

execute_process(COMMAND libacfutils/pkg-config-deps ${PLAT_LONG} --cflags
    OUTPUT_VARIABLE DEP_CFLAGS)
string(REGEX REPLACE "\n$" "" DEP_CFLAGS "${DEP_CFLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${DEP_CFLAGS}")

set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELEASE} -DDEBUG -g \
    -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG -O0 -g")

# linking
# Careful: order matters in target_link_libraries. Dependencies must be
# ordered as first symbol callers and then as symbol providers (i.e.
# libpng16 depends on libz, so linking must be first: libpng, then libz).
#link_libraries(m)

if(WIN32)
	find_library(XPLM_LIBRARY XPLM_64 "${XPSDK}/Libraries/Win")
elseif(APPLE)
	find_library(XPLM_LIBRARY XPLM "${XPSDK}/Libraries/Mac")
else()
	set(XPLM_LIBRARY "")
endif()

set(LIBACFUTILS_LIBRARY "${LIBACFUTILS}/qmake/${PLAT_SHORT}/libacfutils.a")
if(WIN32)
	set(GLEW_LIBRARY "${GLEW}/lib/libglew32mx.a")
elseif(APPLE)
	set(GLEW_LIBRARY "${GLEW}/lib/libGLEWmx.a")
else()
	set(GLEW_LIBRARY "${GLEW}/lib64/libGLEWmx.a")
endif()
set(ZLIB_LIBRARY "${ZLIB}/lib/libz.a")

if(APPLE)
	set(MATH_LIBRARY "")
	set(LIBC_NONSHARED "")
elseif(UNIX)
	find_library(MATH_LIBRARY "m")
	find_library(LIBC_NONSHARED c_nonshared)
endif()

if(APPLE)
	find_library(OPENGL_FRAMEWORK OpenGL)
	find_library(IOSURFACE_FRAMEWORK IOSurface)
	set(EXTRA_PLATFORM_LIBS
	    ${OPENGL_FRAMEWORK}
	    ${IOSURFACE_FRAMEWORK})
else()
	set(EXTRA_PLATFORM_LIBS "")
endif()

target_link_libraries(manipdraw
    ${XPLM_LIBRARY}
    ${LIBACFUTILS_LIBRARY}
    ${GLEW_LIBRARY}
    ${ZLIB_LIBRARY}
    ${EXTRA_PLATFORM_LIBS}
    ${MATH_LIBRARY}
    ${LIBC_NONSHARED}
    )

set_target_properties(manipdraw PROPERTIES PREFIX "")
set_target_properties(manipdraw PROPERTIES SUFFIX "")
set_target_properties(manipdraw PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_SOURCE_DIR}/../${PLUGIN_BIN_OUTDIR}" )
set_target_properties(manipdraw PROPERTIES LIBRARY_OUTPUT_DIRECTORY
    "${CMAKE_SOURCE_DIR}/../${PLUGIN_BIN_OUTDIR}" )
set_target_properties(manipdraw PROPERTIES OUTPUT_NAME "manipdraw.xpl")

if(UNIX AND NOT APPLE)
	set_target_properties(manipdraw PROPERTIES LINK_FLAGS
	    "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic -nodefaultlibs \
	    -undefined_warning -fPIC -fvisibility=hidden -fno-gnu-unique")
endif()
