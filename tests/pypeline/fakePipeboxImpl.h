#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/Model.h"
#include "revng/Pypeline/ObjectID.h"

class StringContainer {
private:
  std::map<ObjectID, std::string> Storage;

public:
  static constexpr llvm::StringRef Name = "StringContainer";
  static constexpr ObjectID::Kind Kind = ObjectID::Kind::Function;
  std::set<ObjectID> objects() const;

  void deserialize(const std::map<const ObjectID *, llvm::ArrayRef<const char>>
                     Data);
  std::map<ObjectID, revng::pypeline::Buffer>
  serialize(const std::vector<const ObjectID *> Objects) const;
  bool verify() const;

public:
  auto &getStorage() { return Storage; }
};

class AppendFooPipe {
public:
  static constexpr llvm::StringRef Name = "AppendFooPipe";
  AppendFooPipe(llvm::StringRef Config) {}

  revng::pypeline::ObjectDependencies run(const Model *TheModel,
                                          revng::pypeline::Request Incoming,
                                          revng::pypeline::Request Outgoing,
                                          llvm::StringRef Configuration,
                                          StringContainer &Container);
};

class AppendFooLibAnalysis {
public:
  static constexpr llvm::StringRef Name = "AppendFooLibAnalysis";

  llvm::Error run(Model *TheModel,
                  revng::pypeline::Request Incoming,
                  llvm::StringRef Configuration,
                  StringContainer &Container);
};
