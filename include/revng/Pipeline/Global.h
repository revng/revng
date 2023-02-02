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
  virtual Global &operator=(const Global &NewGlobal) = 0;

public:
  Global(const Global &) = default;
  Global(Global &&) = default;
  Global &operator=(Global &&) = default;

public:
  const char *getID() const { return ID; }

public:
  virtual GlobalTupleTreeDiff diff(const Global &Other) const = 0;
  virtual llvm::Error applyDiff(const llvm::MemoryBuffer &Diff) = 0;
  virtual llvm::Error applyDiff(const GlobalTupleTreeDiff &Diff) = 0;

  virtual llvm::Error serialize(llvm::raw_ostream &OS) const = 0;
  virtual llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) = 0;
  virtual llvm::Expected<GlobalTupleTreeDiff>
  deserializeDiff(const llvm::MemoryBuffer &Diff) = 0;

  virtual bool verify() const = 0;
  virtual void clear() = 0;

  virtual llvm::Expected<std::unique_ptr<Global>>
  createNew(const llvm::MemoryBuffer &Buffer) const = 0;
  virtual std::unique_ptr<Global> clone() const = 0;

  virtual llvm::Error storeToDisk(llvm::StringRef Path) const;
  virtual llvm::Error loadFromDisk(llvm::StringRef Path);
};

template<TupleTreeCompatibleAndVerifiable Object>
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

  void clear() override {
    Value.evictCachedReferences();
    *Value = Object();
  }

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

  llvm::Expected<GlobalTupleTreeDiff>
  deserializeDiff(const llvm::MemoryBuffer &Buffer) override {
    return ::deserialize<TupleTreeDiff<Object>>(Buffer.getBuffer());
  }

  bool verify() const override { return Value->verify(); }

  GlobalTupleTreeDiff diff(const Global &Other) const override {
    const TupleTreeGlobal &Casted = llvm::cast<TupleTreeGlobal>(Other);
    auto Diff = ::diff(*Value, *Casted.Value);
    return GlobalTupleTreeDiff(std::move(Diff));
  }

  llvm::Error applyDiff(const llvm::MemoryBuffer &Diff) override {
    auto MaybeDiff = TupleTreeDiff<Object>::deserialize(Diff.getBuffer());
    if (not MaybeDiff) {
      return MaybeDiff.takeError();
    }
    return MaybeDiff->apply(Value);
  }

  llvm::Error applyDiff(const TupleTreeDiff<Object> &Diff) {
    return Diff.apply(Value);
  }

  llvm::Error applyDiff(const GlobalTupleTreeDiff &Diff) override {
    return Diff.getAs<Object>()->apply(Value);
  }

  Global &operator=(const Global &Other) override {
    const TupleTreeGlobal &Casted = llvm::cast<TupleTreeGlobal>(Other);
    Value = Casted.Value;
    return *this;
  }

  const TupleTree<Object> &get() const { return Value; }
  TupleTree<Object> &get() { return Value; }
};

} // namespace pipeline
