#ifndef CODEGENERATOR_H
#define CODEGENERATOR_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdint>
#include <memory>
#include <string>

// LLVM includes
#include "llvm/ADT/ArrayRef.h"

// Local libraries includes
#include "revng/Support/revng.h"

// Local includes
#include "BinaryFile.h"

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

}; // namespace llvm

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
  CodeGenerator(BinaryFile &Binary,
                Architecture &Target,
                llvm::LLVMContext &TheContext,
                std::string Output,
                std::string Helpers,
                std::string EarlyLinked);

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
  void parseELF(llvm::object::ObjectFile *TheBinary, bool UseSections);

  /// \brief Import a helper function definition
  ///
  /// Queries the HelpersModule for a function and adds it to TheModule.
  ///
  /// \param Name name of the imported function
  llvm::Function *importHelperFunctionDeclaration(llvm::StringRef Name);

private:
  Architecture TargetArchitecture;
  llvm::LLVMContext &Context;
  std::unique_ptr<llvm::Module> TheModule;
  std::unique_ptr<llvm::Module> HelpersModule;
  std::unique_ptr<llvm::Module> EarlyLinkedModule;
  std::string OutputPath;
  std::unique_ptr<DebugHelper> Debug;
  BinaryFile &Binary;

  unsigned OriginalInstrMDKind;
  unsigned PTCInstrMDKind;
  unsigned DbgMDKind;

  std::string FunctionListPath;
};

#endif // CODEGENERATOR_H
