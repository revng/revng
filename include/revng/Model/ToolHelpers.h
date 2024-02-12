#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>
#include <memory>
#include <string>
#include <type_traits>

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/Model/Binary.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Assert.h"

namespace ModelOutputType {

enum Values {
  Invalid,
  YAML,
  LLVMIR,
  BitCode
};

inline bool requiresModule(Values V) {
  switch (V) {
  case YAML:
    return false;

  case LLVMIR:
  case BitCode:
    return true;

  default:
    revng_abort();
  }
}

inline llvm::sys::fs::OpenFlags getFlags(Values V) {
  using namespace llvm::sys::fs;
  switch (V) {
  case YAML:
  case LLVMIR:
    return OF_Text;

  case BitCode:
    return OF_None;

  default:
    revng_abort();
  }
}

} // namespace ModelOutputType

struct NoOutputYAMLOption {
  template<typename... Args>
  NoOutputYAMLOption(Args...) {}
};

struct OutputYAMLOption {
protected:
  llvm::cl::opt<bool> OutputYAML;

public:
  OutputYAMLOption(llvm::cl::OptionCategory &Category) :
    OutputYAML("Y",
               llvm::cl::desc("Write output as YAML"),
               llvm::cl::cat(Category)) {}
};

template<bool YAML = true>
class ModelOutputOptions
  : public std::conditional_t<YAML, OutputYAMLOption, NoOutputYAMLOption> {
private:
  using Base = std::conditional_t<YAML, OutputYAMLOption, NoOutputYAMLOption>;

private:
  llvm::cl::opt<bool> OutputAssembly;

  llvm::cl::opt<std::string> OutputFilename;

public:
  ModelOutputOptions(llvm::cl::OptionCategory &Category) :
    Base(Category),
    OutputAssembly("S",
                   llvm::cl::desc("Write output as LLVM assembly"),
                   llvm::cl::cat(Category)),
    OutputFilename("o",
                   llvm::cl::init("-"),
                   llvm::cl::desc("Override output filename"),
                   llvm::cl::value_desc("filename"),
                   llvm::cl::cat(Category)) {}

public:
  ModelOutputType::Values getDesiredOutput() const {
    if constexpr (YAML) {
      if (this->OutputYAML && OutputAssembly)
        return ModelOutputType::Invalid;

      if (this->OutputYAML)
        return ModelOutputType::YAML;

      if (OutputAssembly)
        return ModelOutputType::LLVMIR;

      return ModelOutputType::YAML;
    } else {
      if (OutputAssembly)
        return ModelOutputType::LLVMIR;

      return ModelOutputType::YAML;
    }

    revng_abort();
  }

  std::string getPath() const { return OutputFilename; }
};

inline void writeModel(const model::Binary &Model, llvm::Module &M) {
  Model.verify(true);

  llvm::NamedMDNode *NamedMD = M.getNamedMetadata(ModelMetadataName);
  revng_check(not NamedMD, "The model has already been serialized");

  std::string Buffer;
  {
    llvm::raw_string_ostream Stream(Buffer);
    serialize(Stream, Model);
  }

  llvm::LLVMContext &Context = M.getContext();
  auto Tuple = llvm::MDTuple::get(Context,
                                  { llvm::MDString::get(Context, Buffer) });

  NamedMD = M.getOrInsertNamedMetadata(ModelMetadataName);
  NamedMD->addOperand(Tuple);
}
