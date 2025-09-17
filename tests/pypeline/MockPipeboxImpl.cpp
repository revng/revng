//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <set>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "revng/PipeboxCommon/Model.h"
#include "revng/PipeboxCommon/ObjectID.h"

#include "MockPipeboxImpl.h"

//
// StringContainer
//

std::set<ObjectID> StringContainer::objects() const {
  return std::views::keys(Storage) | revng::to<std::set<ObjectID>>();
}

void StringContainer::deserialize(const std::map<const ObjectID *,
                                                 llvm::ArrayRef<const char>>
                                    Data) {
  for (auto &Entry : Data) {
    revng_assert(Entry.first->kind() == Kind);
    std::string EntryData(Entry.second.data(), Entry.second.size());
    Storage[*Entry.first] = std::move(EntryData);
  }
}

std::map<ObjectID, revng::pypeline::Buffer>
StringContainer::serialize(const std::vector<const ObjectID *> Objects) const {
  std::map<ObjectID, revng::pypeline::Buffer> Result;
  for (auto &Element : Objects) {
    const std::string &StoredElement = Storage.at(*Element);
    llvm::ArrayRef<char> Ref = { StoredElement.data(), StoredElement.size() };
    Result[*Element] = llvm::SmallVector<char, 0>{ Ref };
  }
  return Result;
}

bool StringContainer::verify() const {
  return true;
}

//
// AppendFooPipe
//

revng::pypeline::ObjectDependencies
AppendFooPipe::run(const Model &TheModel,
                   const revng::pypeline::Request &Incoming,
                   const revng::pypeline::Request &Outgoing,
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

//
// AppendFooLibAnalysis
//

llvm::Error AppendFooLibAnalysis::run(Model &TheModel,
                                      const revng::pypeline::Request &Incoming,
                                      llvm::StringRef Configuration,
                                      StringContainer &Container) {
  revng_assert(Incoming.size() == 1);
  TheModel.get()->ImportedLibraries().insert("foo.so");

  return llvm::Error::success();
}
