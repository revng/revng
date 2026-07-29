#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/ArrayRef.h"

#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/PipeboxCommon/RawContainer.h"

namespace revng::pypeline {

class ObjectFileContainer : public BytesContainer {
public:
  static constexpr llvm::StringRef Name = "ObjectFileContainer";
  static constexpr llvm::StringRef MimeType = "application/x-object";
  static constexpr llvm::StringRef Compression = "zstd;level=1";
};

namespace piperuns {

class CompileRootModule {
private:
  const model::Binary &Binary;
  LLVMRootContainer &Input;
  ObjectFileContainer &Output;

public:
  static constexpr llvm::StringRef Name = "compile-root-module";
  using Arguments = TypeList<PipeRunArgument<LLVMRootContainer,
                                             "Input",
                                             "The LLVM module that will be "
                                             "compiled",
                                             Access::Read>,
                             PipeRunArgument<ObjectFileContainer,
                                             "Output",
                                             "The compiled object file",
                                             Access::Write>>;

  CompileRootModule(const Model &TheModel,
                    llvm::StringRef StaticConfig,
                    llvm::StringRef DynamicConfig,
                    LLVMRootContainer &Input,
                    ObjectFileContainer &Output);

  void run();
};

} // namespace piperuns

} // namespace revng::pypeline
