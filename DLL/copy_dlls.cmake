file(GLOB DLL_FILES "${SOURCE_DIR}/*.dll")
foreach(DLL ${DLL_FILES})
    file(COPY "${DLL}" DESTINATION "${DEST_DIR}")
endforeach()
