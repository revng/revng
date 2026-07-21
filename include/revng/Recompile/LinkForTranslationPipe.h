#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/ArrayRef.h"

#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/PipeboxCommon/RawContainer.h"
#include "revng/Recompile/CompileModulePipe.h"

namespace revng::pypeline {

class TranslatedContainer : public BytesContainer {
public:
  static constexpr llvm::StringRef Name = "TranslatedContainer";
  static constexpr llvm::StringRef MimeType = "application/x-executable";
  static constexpr llvm::StringRef Compression = "zstd;level=1";
};

namespace piperuns {

class LinkForTranslation {
private:
  const model::Binary &Binary;
  const BinariesContainer &Binaries;
  const ObjectFileContainer &ObjectFile;
  TranslatedContainer &Output;

public:
  static constexpr llvm::StringRef Name = "link-for-translation";
  using Arguments = TypeList<
    PipeRunArgument<const BinariesContainer, "Binaries", "The input binaries">,
    PipeRunArgument<const ObjectFileContainer,
                    "ObjectFile",
                    "The complied object file">,
    PipeRunArgument<TranslatedContainer,
                    "Output",
                    "The output executable",
                    Access::Write>>;

  static llvm::Error checkPrecondition(const class Model &Model);

  LinkForTranslation(const Model &TheModel,
                     llvm::StringRef StaticConfig,
                     llvm::StringRef DynamicConfig,
                     const BinariesContainer &Binaries,
                     const ObjectFileContainer &ObjectFile,
                     TranslatedContainer &Output);

  void run();
};

} // namespace piperuns

} // namespace revng::pypeline
