#ifndef _CODEGENERATOR_H
#define _CODEGENERATOR_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdint>
#include <string>
#include <memory>

// LLVM includes
#include "llvm/ADT/ArrayRef.h"

// Local includes
#include "binaryfile.h"
#include "revamb.h"

// Forward declarations
namespace llvm {

class LLVMContext;
class Function;
class GlobalVariable;
class Module;
class Value;
class StructType;
class DataLayout;

namespace object {
class ObjectFile;
};

};

class DebugHelper;

/// Translator from binary code to LLVM IR.
class CodeGenerator {
public:
  /// Create a new code generator translating code from an architecture to
  /// another, writing the corresponding LLVM IR and other useful information to
  /// the specified paths.
  ///
  /// \param Binary reference to a BinaryFile object describing the input.
  /// \param Target target architecture.
  /// \param Output path where the generate LLVM IR must be saved.
  /// \param Helpers path of the LLVM IR file containing the QEMU helpers.
  /// \param DebugInfo type of debug information to generate.
  /// \param Debug path where the debugging source file must be written. If an
  ///        empty string, the output file name plus ".S", if \p DebugInfo is
  ///        DebugInfoType::OriginalAssembly, or ".ptc", if \p DebugInfo is
  ///        DebugInfoType::PTC.
  /// \param LinkingInfo path where the information about how the linking should
  ///        be stored. If an empty string, the output file name with a
  ///        ".li.csv" suffix will be used.
  /// \param Coverage path where the information about instruction coverage
  ///        should be stored. If an empty string, the output file name with a
  ///        ".coverage.csv" suffix will be used.
  /// \param EnableOSRA specify whether OSRA should be used to discover
  ///        additional jump targets or not.
  /// \param EnableLinking specifying whether linking to QEMU helpers should be
  ///        performed or not.
  CodeGenerator(BinaryFile &Binary,
                Architecture &Target,
                std::string Output,
                std::string Helpers,
                std::string EarlyLinked,
                DebugInfoType DebugInfo,
                std::string Debug,
                std::string LinkingInfo,
                std::string Coverage,
                std::string BBSummary,
                bool EnableOSRA,
                bool DetectFunctionBoundaries,
                bool EnableLinking,
                bool ExternalCSVs);

  ~CodeGenerator();

  /// \brief Creates an LLVM function for the code in the specified memory area.
  /// If debug information has been requested, the debug source files will be
  /// create in this phase.
  ///
  /// \param VirtualAddress the address from where the translation should start.
  void translate(uint64_t VirtualAddress);

  /// Serialize the generated LLVM IR to the specified output path.
  void serialize();

private:
  /// \brief Parse the ELF headers.
  /// Collect useful information such as the segments' boundaries, their
  /// permissions, the address of program headers and the like.
  /// From this information it produces the .li.csv file containing information
  /// useful for linking.
  /// This function parametric w.r.t. endianess and pointer size.
  ///
  /// \param TheBinary the LLVM ObjectFile representing the ELF file.
  /// \param LinkingInfo path where the .li.csv file should be created.
  template<typename T>
  void parseELF(llvm::object::ObjectFile *TheBinary,
                std::string LinkingInfo,
                bool UseSections);

  /// \brief Import a helper function definition
  ///
  /// Queries the HelpersModule for a function and adds it to TheModule.
  ///
  /// \param Name name of the imported function
  llvm::Function *importHelperFunctionDefinition(llvm::StringRef Name);

private:
  Architecture TargetArchitecture;
  llvm::LLVMContext& Context;
  std::unique_ptr<llvm::Module> TheModule;
  std::unique_ptr<llvm::Module> HelpersModule;
  std::unique_ptr<llvm::Module> EarlyLinkedModule;
  std::string OutputPath;
  std::unique_ptr<DebugHelper> Debug;
  BinaryFile &Binary;

  unsigned OriginalInstrMDKind;
  unsigned PTCInstrMDKind;
  unsigned DbgMDKind;

  std::string CoveragePath;
  bool EnableOSRA;
  std::string BBSummaryPath;
  std::string FunctionListPath;
  bool DetectFunctionBoundaries;
  bool EnableLinking;
  bool ExternalCSVs;
};

#endif // _CODEGENERATOR_H
