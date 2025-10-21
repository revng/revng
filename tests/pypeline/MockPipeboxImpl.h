#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/TypeList.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/PipeboxCommon/ObjectID.h"

class StringContainer {
private:
  std::map<ObjectID, std::string> Storage;

public:
  static constexpr llvm::StringRef Name = "StringContainer";
  static constexpr Kind Kind = Kinds::Function;
  static constexpr llvm::StringRef MimeType = "application/x-unknown";

  std::set<ObjectID> objects() const;

  void deserialize(const std::map<const ObjectID *, llvm::ArrayRef<char>> Data);
  std::map<ObjectID, revng::pypeline::Buffer>
  serialize(const std::vector<const ObjectID *> Objects) const;
  bool verify() const;

public:
  auto &getStorage() { return Storage; }
};

class AppendFooPipe {
private:
  using Access = revng::pypeline::Access;

public:
  static constexpr llvm::StringRef Name = "AppendFooPipe";
  using Arguments = TypeList<revng::pypeline::PipeArgument<"Container", "">>;

  const std::string StaticConfiguration;

  AppendFooPipe(llvm::StringRef Config) : StaticConfiguration(Config.str()) {}

  revng::pypeline::ObjectDependencies
  run(const Model &TheModel,
      const revng::pypeline::Request &Incoming,
      const revng::pypeline::Request &Outgoing,
      llvm::StringRef Configuration,
      StringContainer &Container);
};

class AppendFooLibAnalysis {
public:
  static constexpr llvm::StringRef Name = "AppendFooLibAnalysis";

  llvm::Error run(Model &TheModel,
                  const revng::pypeline::Request &Incoming,
                  llvm::StringRef Configuration,
                  StringContainer &Container);
};
