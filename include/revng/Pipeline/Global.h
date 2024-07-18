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
#include "revng/Pipeline/PathTargetBimap.h"
#include "revng/Storage/Path.h"
#include "revng/Support/Error.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/Tracking.h"
#include "revng/TupleTree/TupleTreeDiff.h"

namespace pipeline {

class Global {
private:
  const char *ID;
  std::string Name;

public:
  Global(const char *ID, llvm::StringRef Name) : ID(ID), Name(Name.str()) {}
  virtual ~Global() {}
  virtual Global &operator=(const Global &NewGlobal) = 0;

public:
  Global(const Global &) = default;
  Global(Global &&) = default;
  Global &operator=(Global &&) = default;

public:
  const char *getID() const { return ID; }
  llvm::StringRef getName() const { return Name; }

public:
  virtual GlobalTupleTreeDiff diff(const Global &Other) const = 0;
  virtual llvm::Error applyDiff(const llvm::MemoryBuffer &Diff) = 0;
  virtual llvm::Error applyDiff(const GlobalTupleTreeDiff &Diff) = 0;

  virtual llvm::Error serialize(llvm::raw_ostream &OS) const = 0;
  virtual llvm::Error fromString(llvm::StringRef String) = 0;
  virtual llvm::Expected<GlobalTupleTreeDiff>
  diffFromString(llvm::StringRef String) = 0;

  virtual bool verify() const = 0;
  virtual void clear() = 0;

  virtual llvm::Expected<std::unique_ptr<Global>>
  createNew(llvm::StringRef Name, const llvm::MemoryBuffer &Buffer) const = 0;

  virtual std::unique_ptr<Global> clone() const = 0;

  virtual llvm::Error store(const revng::FilePath &Path) const;
  virtual llvm::Error load(const revng::FilePath &Path);

  virtual std::optional<TupleTreePath>
  deserializePath(llvm::StringRef Serialized) const = 0;

  virtual std::optional<std::string>
  serializePath(const TupleTreePath &Path) const = 0;

  virtual void collectReadFields(const TargetInContainer &Target,
                                 PathTargetBimap &Out) = 0;
  virtual void clearAndResume() const = 0;
  virtual void pushReadFields() const = 0;
  virtual void popReadFields() const = 0;
  virtual void stopTracking() const = 0;
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
  explicit TupleTreeGlobal(llvm::StringRef Name, TupleTree<Object> Value) :
    Global(&getID(), Name), Value(std::move(Value)) {}

  explicit TupleTreeGlobal(llvm::StringRef Name) : Global(&getID(), Name) {}
  TupleTreeGlobal(const TupleTreeGlobal &Other) = default;
  TupleTreeGlobal(TupleTreeGlobal &&Other) = default;
  TupleTreeGlobal &operator=(const TupleTreeGlobal &Other) = default;
  TupleTreeGlobal &operator=(TupleTreeGlobal &&Other) = default;
  virtual ~TupleTreeGlobal() override = default;

  static bool classof(const Global *T) { return T->getID() == &getID(); }

public:
  llvm::Expected<std::unique_ptr<Global>>
  createNew(llvm::StringRef Name,
            const llvm::MemoryBuffer &Buffer) const override {
    auto MaybeTree = TupleTree<Object>::fromString(Buffer.getBuffer());
    if (!MaybeTree)
      return MaybeTree.takeError();
    return std::make_unique<TupleTreeGlobal>(Name, MaybeTree.get());
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

  llvm::Error fromString(llvm::StringRef String) override {
    auto MaybeTupleTree = TupleTree<Object>::fromString(String);
    if (!MaybeTupleTree)
      return MaybeTupleTree.takeError();

    if (not(*MaybeTupleTree)->verify()) {
      return revng::createError("Verify failed on " + getName());
    }

    Value = *MaybeTupleTree;
    return llvm::Error::success();
  }

  llvm::Expected<GlobalTupleTreeDiff>
  diffFromString(llvm::StringRef String) override {
    auto MaybeDiff = ::fromString<TupleTreeDiff<Object>>(String);
    if (not MaybeDiff)
      return MaybeDiff.takeError();
    return GlobalTupleTreeDiff(std::move(*MaybeDiff), getName());
  }

  bool verify() const override { return Value->verify(); }

  GlobalTupleTreeDiff diff(const Global &Other) const override {
    const TupleTreeGlobal &Casted = llvm::cast<TupleTreeGlobal>(Other);
    auto Diff = ::diff(*Value, *Casted.Value);
    return GlobalTupleTreeDiff(std::move(Diff), getName());
  }

  llvm::Error applyDiff(const llvm::MemoryBuffer &Diff) override {
    auto MaybeDiff = TupleTreeDiff<Object>::fromString(Diff.getBuffer());
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

  std::optional<TupleTreePath>
  deserializePath(llvm::StringRef Serialized) const override {
    return stringAsPath<Object>(Serialized);
  }

  std::optional<std::string>
  serializePath(const TupleTreePath &Path) const override {
    return pathAsString<Object>(Path);
  }

  void collectReadFields(const TargetInContainer &Target,
                         PathTargetBimap &Out) override {
    const TupleTree<Object> &AsConst = Value;
    ReadFields Results = revng::Tracking::collect(*AsConst);
    for (const TupleTreePath &Result : Results.Read)
      Out.insert(Target, Result);
    for (const TupleTreePath &Result : Results.ExactVectors)
      Out.insert(Target, Result);
  }

  void clearAndResume() const override {
    revng::Tracking::clearAndResume(*Value);
  }
  void pushReadFields() const override { revng::Tracking::push(*Value); }
  void popReadFields() const override { revng::Tracking::pop(*Value); }
  void stopTracking() const override { revng::Tracking::stop(*Value); }
};

} // namespace pipeline
