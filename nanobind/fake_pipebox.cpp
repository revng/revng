//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iostream>
#include <map>
#include <set>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Analysis.h"
#include "revng/Pypeline/Container.h"
#include "revng/Pypeline/Model.h"
#include "revng/Pypeline/ObjectID.h"
#include "revng/Pypeline/Pipe.h"

static RegisterObjectID T;
static RegisterModel T2;

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

  void deserialize(std::map<ObjectID *, llvm::ArrayRef<char>> Data) {
    for (auto &Entry : Data) {
      revng_assert(Entry.first->kind() == kind());
      std::string EntryData(Entry.second.data(), Entry.second.size());
      Storage[*Entry.first] = std::move(EntryData);
    }
  }

  std::map<ObjectID, detail::Buffer>
  serialize(std::set<ObjectID *> Objects) const {
    std::map<ObjectID, detail::Buffer> Result;
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

  detail::ObjectDependencies run(const Model *TheModel,
                                 detail::RequestT Incoming,
                                 detail::RequestT Outgoing,
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

class PrintFooAnalysis {
public:
  static constexpr llvm::StringRef Name = "PrintFooAnalysis";

  bool run(Model *TheModel,
           detail::RequestT Incoming,
           llvm::StringRef Configuration,
           StringContainer &Container) {
    revng_assert(Incoming.size() == 1);

    auto &Storage = Container.getStorage();
    std::cout << "foo analysis running" << std::endl;
    return true;
  }
};

static RegisterAnalysis<PrintFooAnalysis> Z;
