#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/ObjectID.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/ZstdStream.h"

namespace revng::pypeline {

class LLVMRootContainer {
public:
  static constexpr llvm::StringRef Name = "LLVMRootContainer";
  static constexpr Kind Kind = Kinds::Binary;
  static constexpr llvm::StringRef MimeType = "application/x.llvm.bc+zstd";

private:
  llvm::LLVMContext Context;
  std::unique_ptr<llvm::Module> Module;

public:
  LLVMRootContainer() {
    Module = std::make_unique<llvm::Module>("revng.module", Context);
  }

public:
  std::set<ObjectID> objects() const {
    if (Module->empty())
      return std::set<ObjectID>{};
    else
      return std::set{ ObjectID() };
  }

  void
  deserialize(const std::map<const ObjectID *, llvm::ArrayRef<char>> Data) {
    if (Data.size() == 0)
      return;

    revng_assert(Data.size() == 1);
    for (const auto &[Key, Value] : Data) {
      revng_assert(Key->kind() == Kind);

      llvm::SmallVector<char> DecompressedData = zstdDecompress(Value);
      llvm::MemoryBufferRef Ref{
        { DecompressedData.data(), DecompressedData.size() }, "input"
      };

      Module = llvm::cantFail(llvm::parseBitcodeFile(Ref, Context));
    }
  }

  std::map<ObjectID, Buffer>
  serialize(const std::vector<const ObjectID *> Objects) const {
    if (Objects.size() == 0)
      return {};

    revng_assert(Objects.size() == 1 and Objects[0]->kind() == Kind);
    std::map<ObjectID, Buffer> Result;
    llvm::raw_svector_ostream OS(Result[*Objects[0]].data());
    ZstdCompressedOstream CompressedOS(OS, 3);
    llvm::WriteBitcodeToFile(*Module, CompressedOS);
    CompressedOS.flush();
    return Result;
  }

  bool verify() const {
    revng::forceVerify(&*Module);
    return true;
  }

public:
  const llvm::Module &getModule() const { return *Module; }
  llvm::Module &getModule() { return *Module; }
};

} // namespace revng::pypeline
