//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <set>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Model.h"
#include "revng/Pypeline/ObjectID.h"
#include "revng/Pypeline/Registrators/Analysis.h"
#include "revng/Pypeline/Registrators/Container.h"
#include "revng/Pypeline/Registrators/Pipe.h"

class StringContainer {
private:
  std::map<ObjectID, std::string> Storage;

public:
  static constexpr llvm::StringRef Name = "StringContainer";

  ObjectID::Kind kind() const { return ObjectID::Kind::Function; }

  std::set<ObjectID> objects() const {
    std::set<ObjectID> Result;
    for (auto &Entry : Storage)
      Result.insert(Entry.first);
    return Result;
  }

  void deserialize(const std::map<const ObjectID *, llvm::ArrayRef<const char>>
                     Data) {
    for (auto &Entry : Data) {
      revng_assert(Entry.first->kind() == kind());
      std::string EntryData(Entry.second.data(), Entry.second.size());
      Storage[*Entry.first] = std::move(EntryData);
    }
  }

  std::map<ObjectID, pypeline::Buffer>
  serialize(const std::vector<const ObjectID *> Objects) const {
    std::map<ObjectID, pypeline::Buffer> Result;
    for (auto &Element : Objects) {
      const std::string &StoredElement = Storage.at(*Element);
      llvm::ArrayRef<char> Ref = { StoredElement.data(), StoredElement.size() };
      Result[*Element] = llvm::SmallVector<char, 0>{ Ref };
    }
    return Result;
  }

  bool verify() const { return true; }

public:
  auto &getStorage() { return Storage; }
};

static RegisterContainer<StringContainer> X;

class AppendFooPipe {
public:
  static constexpr llvm::StringRef Name = "AppendFooPipe";

  AppendFooPipe(llvm::StringRef Config) {}

  pypeline::ObjectDependencies run(const Model *TheModel,
                                   pypeline::Request Incoming,
                                   pypeline::Request Outgoing,
                                   llvm::StringRef Configuration,
                                   StringContainer &Container) {
    revng_assert(Outgoing.size() == 1);

    auto &Storage = Container.getStorage();
    for (auto &Elem : Outgoing[0]) {
      if (Storage.contains(*Elem)) {
        Storage[*Elem] = Storage[*Elem] + "foo";
      } else {
        Storage[*Elem] = "foo";
      }
    }

    return {};
  }
};

static RegisterPipe<AppendFooPipe> Y;

class AppendFooLibAnalysis {
public:
  static constexpr llvm::StringRef Name = "AppendFooLibAnalysis";

  llvm::Error run(Model *TheModel,
                  pypeline::Request Incoming,
                  llvm::StringRef Configuration,
                  StringContainer &Container) {
    revng_assert(Incoming.size() == 1);
    TheModel->get()->ImportedLibraries().insert("foo.so");

    return llvm::Error::success();
  }
};

static RegisterAnalysis<AppendFooLibAnalysis> Z;
