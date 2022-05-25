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

public:
  virtual void serialize(llvm::raw_ostream &OS) const = 0;
  virtual ~GlobalTupleTreeDiffBase() = default;
  virtual std::unique_ptr<GlobalTupleTreeDiffBase> clone() const = 0;
  GlobalTupleTreeDiffBase(char *ID) : ID(ID) {}
  const char *getID() const { return ID; }
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
  GlobalTupleTreeDiffImpl(TupleTreeDiff<T> Diff) :
    GlobalTupleTreeDiffBase(getID()), Diff(std::move(Diff)) {}

  ~GlobalTupleTreeDiffImpl() override = default;

  void serialize(llvm::raw_ostream &OS) const override {
    ::serialize(OS, Diff);
  }

  std::unique_ptr<GlobalTupleTreeDiffBase> clone() const override {
    auto Ptr = new GlobalTupleTreeDiffImpl(*this);
    return std::unique_ptr<GlobalTupleTreeDiffBase>(Ptr);
  }

  static bool classof(const GlobalTupleTreeDiffBase *Base) {
    return Base->getID() == getID();
  }

  TupleTreeDiff<T> &getDiff() { return Diff; }
  const TupleTreeDiff<T> &getDiff() const { return Diff; }
};

class GlobalTupleTreeDiff {
private:
  std::unique_ptr<GlobalTupleTreeDiffBase> Diff;

public:
  template<typename T>
  GlobalTupleTreeDiff(TupleTreeDiff<T> Diff) :
    Diff(new GlobalTupleTreeDiffImpl<T>(std::move(Diff))) {}

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

  template<typename T>
  const TupleTreeDiff<T> *getAs() const {
    const auto *Casted = llvm::dyn_cast<GlobalTupleTreeDiffImpl<T>>(Diff.get());
    if (not Casted)
      return nullptr;

    return &Casted->getDiff();
  }

  template<typename T>
  TupleTreeDiff<T> *getAs() {
    const auto *Casted = llvm::dyn_cast<GlobalTupleTreeDiffImpl<T>>(Diff.get());
    if (not Casted)
      return nullptr;

    return &Casted->getDiff();
  }
};

using DiffMap = llvm::StringMap<GlobalTupleTreeDiff>;
} // namespace pipeline
