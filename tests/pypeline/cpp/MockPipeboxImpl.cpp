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

using llvm::ArrayRef;

//
// StringContainer
//

std::set<ObjectID> StringContainer::objects() const {
  return std::views::keys(Storage) | revng::to<std::set<ObjectID>>();
}

void StringContainer::deserialize(const std::map<const ObjectID *,
                                                 llvm::ArrayRef<char>> Data) {
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

void StringContainer::setIsDisposable() {
  Disposable = true;
}

void StringContainer::disposeIfPossible() {
  if (not Disposable)
    return;

  Storage.clear();
  Disposable = false;
}

//
// RootStringContainer
//

std::set<ObjectID> RootStringContainer::objects() const {
  if (Storage.empty())
    return {};
  else
    return { ObjectID::root() };
}

void RootStringContainer::deserialize(const std::map<const ObjectID *,
                                                     ArrayRef<char>> Data) {
  for (auto &[Object, Entry] : Data)
    Storage = { Entry.data(), Entry.size() };
}

std::map<ObjectID, revng::pypeline::Buffer>
RootStringContainer::serialize(const std::vector<const ObjectID *> Objects)
  const {
  std::map<ObjectID, revng::pypeline::Buffer> Result;
  for (const ObjectID *Object : Objects)
    Result[*Object] = llvm::ArrayRef<char>{ Storage.data(), Storage.size() };
  return Result;
}

void RootStringContainer::setIsDisposable() {
  Disposable = true;
}

void RootStringContainer::disposeIfPossible() {
  if (not Disposable)
    return;

  Storage = {};
  Disposable = false;
}

//
// AppendFooPipe
//

revng::pypeline::PipeOutput
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

  return { {}, {} };
}

//
// CreateEntryPipe
//

revng::pypeline::PipeOutput
CreateEntryPipe::run(const Model &TheModel,
                     const revng::pypeline::Request &Incoming,
                     const revng::pypeline::Request &Outgoing,
                     llvm::StringRef Configuration,
                     RootStringContainer &Container) {
  revng_assert(Outgoing.size() == 1);
  revng_assert(Outgoing[0].size() == 1);
  Container.getStorage() = "foo";
  return { {}, {} };
}

//
// AppendFooPipe2
//

revng::pypeline::PipeOutput
AppendFooPipe2::run(const Model &TheModel,
                    const revng::pypeline::Request &Incoming,
                    const revng::pypeline::Request &Outgoing,
                    llvm::StringRef Configuration,
                    RootStringContainer &Container) {
  revng_assert(Outgoing.size() == 1);
  revng_assert(Outgoing[0].size() == 1);
  auto &Storage = Container.getStorage();
  Storage = Storage + "foo";
  return { {}, {} };
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

llvm::Error AppendFooLibAnalysis2::run(Model &TheModel,
                                       const revng::pypeline::Request &Incoming,
                                       llvm::StringRef Configuration,
                                       RootStringContainer &Container) {
  revng_assert(Incoming.size() == 1);
  TheModel.get()->ImportedLibraries().insert("foo.so");

  return llvm::Error::success();
}
