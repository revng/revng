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
#include "revng/Model/SerializeModelPass.h"
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
  ModelOutputType::Values getDesiredOutput(bool HasModule) const {
    if constexpr (YAML) {
      if (this->OutputYAML && OutputAssembly)
        return ModelOutputType::Invalid;

      if (OutputAssembly and not HasModule)
        return ModelOutputType::Invalid;

      if (this->OutputYAML)
        return ModelOutputType::YAML;

      if (OutputAssembly and not HasModule)
        return ModelOutputType::LLVMIR;

      if (HasModule)
        return ModelOutputType::BitCode;
      else
        return ModelOutputType::YAML;
    } else {
      if (OutputAssembly and not HasModule)
        return ModelOutputType::Invalid;

      if (OutputAssembly and not HasModule)
        return ModelOutputType::LLVMIR;

      if (HasModule)
        return ModelOutputType::BitCode;
      else
        return ModelOutputType::YAML;
    }

    revng_abort();
  }

  std::string getPath() const { return OutputFilename; }
};

class ModelInModule {
private:
  std::unique_ptr<llvm::LLVMContext> Context;
  std::unique_ptr<llvm::Module> Module;

public:
  TupleTree<model::Binary> Model;

public:
  static llvm::Expected<ModelInModule> load(const llvm::MemoryBuffer &MB) {
    if (MB.getBuffer().startswith("---")) {
      return loadYAML(MB);
    } else {
      return loadModule(MB);
    }
  }

  static llvm::Expected<ModelInModule>
  loadModule(const llvm::MemoryBuffer &MB) {
    using namespace llvm;
    ModelInModule Result;

    Result.Context = std::make_unique<llvm::LLVMContext>();

    auto MaybeModule = parseIR(*Result.Context, MB);
    if (not MaybeModule)
      return MaybeModule.takeError();

    Result.Module = std::move(*MaybeModule);
    if (hasModel(*Result.Module))
      Result.Model = loadModel(*Result.Module);

    return Result;
  }

  static llvm::Expected<ModelInModule> loadYAML(const llvm::MemoryBuffer &MB) {
    using namespace llvm;
    ModelInModule Result;
    auto MaybeModel = TupleTree<model::Binary>::deserialize(MB.getBuffer());

    if (not MaybeModel)
      return errorOrToExpected(ErrorOr<ModelInModule>(MaybeModel.getError()));

    Result.Model = std::move(*MaybeModel);

    return Result;
  }

  static llvm::Expected<ModelInModule> load(const llvm::Twine &Path) {
    auto MaybeBuffer = read(Path);
    if (not MaybeBuffer)
      return MaybeBuffer.takeError();
    return load(*MaybeBuffer->get());
  }

  static llvm::Expected<ModelInModule> loadModule(const llvm::Twine &Path) {
    auto MaybeBuffer = read(Path);
    if (not MaybeBuffer)
      return MaybeBuffer.takeError();
    return loadModule(*MaybeBuffer->get());
  }

  static llvm::Expected<ModelInModule> loadYAML(const llvm::Twine &Path) {
    auto MaybeBuffer = read(Path);
    if (not MaybeBuffer)
      return MaybeBuffer.takeError();
    return loadYAML(*MaybeBuffer->get());
  }

public:
  bool hasModule() const { return static_cast<bool>(Module); }
  const llvm::Module &getModule() const {
    revng_assert(hasModule());
    return *Module;
  }

  TupleTree<model::Binary> &getWriteableModel() {
    Model.evictCachedReferences();
    return Model;
  }

  const model::Binary &getReadOnlyModel() {
    Model.cacheReferences();
    return *std::as_const(Model);
  }

public:
  llvm::Error save(const llvm::Twine &Path, ModelOutputType::Values Type) {
    using namespace llvm;

    Model->verify(true);

    if (ModelOutputType::requiresModule(Type)) {
      if (not hasModule()) {
        return llvm::createStringError(inconvertibleErrorCode(),
                                       "Cannot produce module: input was YAML");
      }

      updateModule();
    }

    std::error_code EC;
    llvm::ToolOutputFile OutputFile(Path.str(),
                                    EC,
                                    ModelOutputType::getFlags(Type));

    if (EC)
      return createStringError(EC, EC.message());

    auto &Stream = OutputFile.os();

    switch (Type) {
    case ModelOutputType::YAML:
      Model.serialize(Stream);
      break;

    case ModelOutputType::LLVMIR:
      Module->print(Stream, nullptr);
      break;

    case ModelOutputType::BitCode:
      WriteBitcodeToFile(*Module, Stream);
      break;

    default:
      revng_abort();
    }

    OutputFile.keep();

    return Error::success();
  }

private:
  static llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  read(const llvm::Twine &Path) {
    return llvm::errorOrToExpected(llvm::MemoryBuffer::getFileOrSTDIN(Path));
  }

  static llvm::Expected<std::unique_ptr<llvm::Module>>
  parseIR(llvm::LLVMContext &Context, const llvm::MemoryBuffer &MB) {
    using namespace llvm;
    SMDiagnostic Diagnostic;
    std::unique_ptr<llvm::Module> MaybeModule = llvm::parseIR(MB,
                                                              Diagnostic,
                                                              Context);
    if (not MaybeModule) {
      std::string ErrMsg;
      {
        raw_string_ostream ErrStream(ErrMsg);
        Diagnostic.print("", ErrStream);
      }
      return make_error<StringError>(std::move(ErrMsg),
                                     inconvertibleErrorCode());
    }

    return MaybeModule;
  }

private:
  void updateModule() {
    auto *NamedMD = Module->getNamedMetadata(ModelMetadataName);
    if (NamedMD != nullptr)
      NamedMD->eraseFromParent();

    writeModel(*Model, *Module);
  }
};
