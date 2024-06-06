#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <any>

#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTreeDiff.h"

namespace pipeline {
class GlobalTupleTreeDiffBase {
private:
  char *ID;
  llvm::StringRef GlobalName;

public:
  virtual ~GlobalTupleTreeDiffBase() = default;
  GlobalTupleTreeDiffBase(char *ID, llvm::StringRef GlobalName) :
    ID(ID), GlobalName(GlobalName) {}

public:
  virtual void serialize(llvm::raw_ostream &OS) const = 0;

public:
  virtual std::unique_ptr<GlobalTupleTreeDiffBase> clone() const = 0;

public:
  const char *getID() const { return ID; }

public:
  virtual bool isEmpty() const = 0;
  llvm::StringRef getGlobalName() const { return GlobalName; }
  virtual llvm::SmallVector<const TupleTreePath *, 4> getPaths() const = 0;
};

template<TupleTreeCompatible T>
class GlobalTupleTreeDiffImpl : public GlobalTupleTreeDiffBase {
private:
  TupleTreeDiff<T> Diff;

  static char *getID() {
    static char ID;
    return &ID;
  }

public:
  GlobalTupleTreeDiffImpl(TupleTreeDiff<T> Diff, llvm::StringRef GlobalName) :
    GlobalTupleTreeDiffBase(getID(), GlobalName), Diff(std::move(Diff)) {}

  ~GlobalTupleTreeDiffImpl() override = default;

  void serialize(llvm::raw_ostream &OS) const override {
    ::serialize(OS, Diff);
  }

  std::unique_ptr<GlobalTupleTreeDiffBase> clone() const override {
    auto Ptr = new GlobalTupleTreeDiffImpl(*this);
    return std::unique_ptr<GlobalTupleTreeDiffBase>(Ptr);
  }

  bool isEmpty() const override { return Diff.Changes.size() == 0; }

  static bool classof(const GlobalTupleTreeDiffBase *Base) {
    return Base->getID() == getID();
  }

  TupleTreeDiff<T> &getDiff() { return Diff; }
  const TupleTreeDiff<T> &getDiff() const { return Diff; }

  llvm::SmallVector<const TupleTreePath *, 4> getPaths() const override {
    llvm::SmallVector<const TupleTreePath *, 4> ToReturn;
    for (const Change<T> &Entry : Diff.Changes) {
      ToReturn.emplace_back(&Entry.Path);
    }
    return ToReturn;
  }
};

class GlobalTupleTreeDiff {
private:
  std::unique_ptr<GlobalTupleTreeDiffBase> Diff;

public:
  template<TupleTreeCompatible T>
  GlobalTupleTreeDiff(TupleTreeDiff<T> Diff, llvm::StringRef GlobalName) :
    Diff(new GlobalTupleTreeDiffImpl<T>(std::move(Diff), GlobalName)) {}

  GlobalTupleTreeDiff(GlobalTupleTreeDiff &&) = default;
  GlobalTupleTreeDiff &operator=(GlobalTupleTreeDiff &&) = default;

  GlobalTupleTreeDiff(const GlobalTupleTreeDiff &Other) :
    Diff(Other.Diff->clone()) {}
  GlobalTupleTreeDiff &operator=(const GlobalTupleTreeDiff &Other) {
    if (this == &Other)
      return *this;

    Diff = Other.Diff->clone();
    return *this;
  }

  ~GlobalTupleTreeDiff() = default;

  void serialize(llvm::raw_ostream &OS) const { Diff->serialize(OS); }

  template<TupleTreeCompatible T>
  const TupleTreeDiff<T> *getAs() const {
    const auto *Casted = llvm::dyn_cast<GlobalTupleTreeDiffImpl<T>>(Diff.get());
    if (not Casted)
      return nullptr;

    return &Casted->getDiff();
  }

  template<TupleTreeCompatible T>
  TupleTreeDiff<T> *getAs() {
    const auto *Casted = llvm::dyn_cast<GlobalTupleTreeDiffImpl<T>>(Diff.get());
    if (not Casted)
      return nullptr;

    return &Casted->getDiff();
  }

  llvm::SmallVector<const TupleTreePath *, 4> getPaths() const {
    return Diff->getPaths();
  }

  bool isEmpty() const { return Diff.get()->isEmpty(); }

  llvm::StringRef getGlobalName() const { return Diff->getGlobalName(); }
};

using DiffMap = llvm::StringMap<GlobalTupleTreeDiff>;
} // namespace pipeline
