#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"

#include "revng/Model/Binary.h"
#include "revng/Model/RawBinaryView.h"

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

/// Translator from binary code to LLVM IR.
class CodeGenerator {
public:
  /// Create a new code generator translating code from an architecture to
  /// another, writing the corresponding LLVM IR and other useful information to
  /// the specified paths.
  CodeGenerator(const RawBinaryView &RawBinary,
                llvm::Module *TheModule,
                const TupleTree<model::Binary> &Model,
                std::string Helpers,
                std::string EarlyLinked,
                model::Architecture::Values TargetArchitecture);

  ~CodeGenerator();

  /// Creates an LLVM function for the code in the specified memory area.
  ///
  /// \param VirtualAddress the address from where the translation should start.
  void translate(llvm::Optional<uint64_t> RawVirtualAddress);

private:
  const RawBinaryView &RawBinary;
  llvm::Module *TheModule;
  llvm::LLVMContext &Context;
  std::unique_ptr<llvm::Module> HelpersModule;
  std::unique_ptr<llvm::Module> EarlyLinkedModule;
  const TupleTree<model::Binary> &Model;

  unsigned OriginalInstrMDKind;
  unsigned PTCInstrMDKind;

  std::string FunctionListPath;

  std::set<MetaAddress> NoMoreCodeBoundaries;

  model::Architecture::Values TargetArchitecture;
};
