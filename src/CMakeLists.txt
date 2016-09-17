#find_library(MGL mgl PATHS /usr/local/lib)

ADD_EXECUTABLE( eyeLike main.cpp classify.cpp findEyeCenter.cpp findEyeCenter.h findEyeCorner.cpp findEyeCorner.h classify.hpp constants.h helpers.cpp helpers.h)
TARGET_LINK_LIBRARIES( eyeLike ${OpenCV_LIBS} )
#target_link_libraries( eyeLike ${MGL})
