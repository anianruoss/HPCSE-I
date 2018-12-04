 # - Find ZFP
     # Find the native NETPBM includes and library
     # This module defines
     #  ZFP_INCLUDE_DIR, where to find jpeglib.h, etc.
     #  ZFP_LIBRARIES, the libraries needed to use ZFP.
     #  ZFP_FOUND, If false, do not try to use ZF{.
     # also defined, but not for general use are
     #  ZFP_LIBRARY, where to find the ZFP library.
     
 FIND_PATH(ZFP_INCLUDE_DIR zfp.h
PATHS
 $ENV{ZFP_ROOT_DIR}/include
 /opt/zfp/include
  
 )
 
 SET(ZFP_NAMES ${ZFP_NAMES} zfp)
 FIND_LIBRARY(ZFP_LIBRARY
   NAMES ${ZFP_NAMES}
   PATHS $ENV{ZFP_ROOT_DIR}/lib /opt/zfp/lib
   )

MESSAGE(${ZFP_INCLUDE_DIR} ${ZFP_ROOT_DIR} ${ZFP_LIBRARY} "ZFP")
 
 IF (ZFP_LIBRARY AND ZFP_INCLUDE_DIR)
     SET(ZFP_LIBRARIES ${ZFP_LIBRARY})
     SET(ZFP_FOUND "YES")
 ELSE (ZFP_LIBRARY AND ZFP_INCLUDE_DIR)
   SET(ZFP_FOUND "NO")
 ENDIF (ZFP_LIBRARY AND ZFP_INCLUDE_DIR)
 
 
 IF (ZFP_FOUND)
    IF (NOT ZFP_FIND_QUIETLY)
       MESSAGE(STATUS "Found ZFP: ${ZFP_LIBRARIES}")
    ENDIF (NOT ZFP_FIND_QUIETLY)
 ELSE (ZFP_FOUND)
    IF (ZFP_FIND_REQUIRED)
       MESSAGE(FATAL_ERROR "Could not find ZFP library")
    ENDIF (ZFP_FIND_REQUIRED)
 ENDIF (ZFP_FOUND)
 

 MARK_AS_ADVANCED(
   ZFP_LIBRARY
   ZFP_INCLUDE_DIR
   )
 
