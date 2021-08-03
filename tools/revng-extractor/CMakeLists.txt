llvm_map_components_to_libnames(llvm_extractor_pdb_libs 
	core
	BinaryFormat
	DebugInfoCodeView
	DebugInfoMSF
	DebugInfoPDB
	Object
	ObjectYAML
	debuginfodwarf
	mc 
	Support
	Core
	${LLVM_TARGETS_TO_BUILD})

revng_add_executable(revng-extractor
  StreamManager.cpp
  InputFile.cpp
  Main.cpp
  ParametersExtractor.cpp
  DwarfExtractor.cpp
  )

target_link_libraries(revng-extractor
                      PRIVATE
					  revngSupport
					  ${llvm_extractor_pdb_libs})

