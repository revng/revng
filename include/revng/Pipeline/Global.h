#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <any>
#include <memory>
#include <type_traits>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/GlobalTupleTreeDiff.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTreeDiff.h"

namespace pipeline {

class Global {
private:
  const char *ID;

public:
  Global(const char *ID) : ID(ID) {}

  virtual ~Global() {}

public:
  const char *getID() const { return ID; }

public:
  virtual GlobalTupleTreeDiff diff(const Global &Other) const = 0;

  virtual llvm::Error applyDiff(const llvm::MemoryBuffer &Diff) = 0;
  virtual llvm::Error serialize(llvm::raw_ostream &OS) const = 0;
  virtual llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) = 0;
  virtual void verify(ErrorList &EL) const = 0;
  virtual void clear() = 0;
  virtual llvm::Expected<std::unique_ptr<Global>>
  createNew(const llvm::MemoryBuffer &Buffer) const = 0;
  virtual std::unique_ptr<Global> clone() const = 0;
  virtual llvm::Error storeToDisk(llvm::StringRef Path) const;
  virtual llvm::Error loadFromDisk(llvm::StringRef Path);
};

template<TupleTreeCompatibleWithVerify Object>
class TupleTreeGlobal : public Global {
private:
  TupleTree<Object> Value;

  static const char &getID() {
    static char ID;
    return ID;
  }

public:
  explicit TupleTreeGlobal(TupleTree<Object> Value) :
    Global(&getID()), Value(std::move(Value)) {}

  TupleTreeGlobal() : Global(&getID()) {}
  TupleTreeGlobal(const TupleTreeGlobal &Other) = default;
  TupleTreeGlobal(TupleTreeGlobal &&Other) = default;
  TupleTreeGlobal &operator=(const TupleTreeGlobal &Other) = default;
  TupleTreeGlobal &operator=(TupleTreeGlobal &&Other) = default;
  virtual ~TupleTreeGlobal() override = default;

  static bool classof(const Global *T) { return T->getID() == &getID(); }

public:
  llvm::Expected<std::unique_ptr<Global>>
  createNew(const llvm::MemoryBuffer &Buffer) const override {
    auto MaybeTree = TupleTree<Object>::deserialize(Buffer.getBuffer());
    if (!MaybeTree)
      return llvm::errorCodeToError(MaybeTree.getError());
    return std::make_unique<TupleTreeGlobal>(MaybeTree.get());
  }

  std::unique_ptr<Global> clone() const override {
    auto Ptr = new TupleTreeGlobal(*this);
    return std::unique_ptr<Global>(Ptr);
  }

  void clear() override { *Value = Object(); }

  llvm::Error serialize(llvm::raw_ostream &OS) const override {
    Value.serialize(OS);
    return llvm::Error::success();
  }

  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) override {
    auto MaybeDiff = TupleTree<Object>::deserialize(Buffer.getBuffer());
    if (!MaybeDiff)
      return llvm::errorCodeToError(MaybeDiff.getError());

    Value = *MaybeDiff;
    return llvm::Error::success();
  }

  void verify(ErrorList &EL) const override { Value->verify(EL); }

  GlobalTupleTreeDiff diff(const Global &Other) const override {
    const TupleTreeGlobal &Casted = llvm::cast<TupleTreeGlobal>(Other);
    auto Diff = ::diff(*Value, *Casted.Value);
    return GlobalTupleTreeDiff(std::move(Diff));
  }

  llvm::Error applyDiff(const llvm::MemoryBuffer &Diff) override {
    auto MaybeDiff = ::deserialize<TupleTreeDiff<Object>>(Diff.getBuffer());
    if (not MaybeDiff)
      return MaybeDiff.takeError();
    MaybeDiff->apply(Value);
    return llvm::Error::success();
  }

  const TupleTree<Object> &get() const { return Value; }
  TupleTree<Object> &get() { return Value; }
};

} // namespace pipeline
