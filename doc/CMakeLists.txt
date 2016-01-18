find_package(Doxygen)
IF(DOXYGEN_FOUND)
  CONFIGURE_FILE(Doxyfile.in "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile" @ONLY)
  add_custom_target(
    doc
    COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
  )
ENDIF()
