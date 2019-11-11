# Copyright (C) 2019 Fondazione Istituto Italiano di Tecnologia (IIT)  
# All Rights Reserved.
# Authors: Ugo Pattacini <ugo.pattacini@iit.it>

project(skeletonLocker)
set(doc_files ${PROJECT_NAME}.xml)

source_group("DOC Files" FILES ${doc_files})

add_executable(${PROJECT_NAME} src/main.cpp ${doc_files})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME} ${YARP_LIBRARIES} AssistiveRehab)
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

file(GLOB ini app/conf/*.ini)
file(GLOB scripts app/scripts/*.template)
yarp_install(FILES ${ini} DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${PROJECT_NAME})
yarp_install(FILES ${scripts} DESTINATION ${ICUBCONTRIB_APPLICATIONS_TEMPLATES_INSTALL_DIR})
