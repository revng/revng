#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"

#include "revng/Support/revng.h"

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
  void translate(llvm::Optional<uint64_t> RawVirtualAddress);

  /// Serialize the generated LLVM IR to the specified output path.
  void serialize();

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

  std::string FunctionListPath;

  std::set<MetaAddress> NoMoreCodeBoundaries;
};
