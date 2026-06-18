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
  bool Disposable = false;
  std::map<ObjectID, std::string> Storage;

public:
  static constexpr llvm::StringRef Name = "StringContainer";
  static constexpr Kind Kind = Kinds::Function;
  static constexpr llvm::StringRef MimeType = "application/x-unknown";
  static constexpr llvm::StringRef Compression = "none";

  std::set<ObjectID> objects() const;

  void deserialize(const std::map<const ObjectID *, llvm::ArrayRef<char>> Data);
  std::map<ObjectID, revng::pypeline::Buffer>
  serialize(const std::vector<const ObjectID *> Objects) const;
  bool verify() const;
  void setIsDisposable();
  void disposeIfPossible();

public:
  auto &getStorage() { return Storage; }
};

class RootStringContainer {
private:
  bool Disposable = false;
  std::string Storage;

public:
  static constexpr llvm::StringRef Name = "RootStringContainer";
  static constexpr Kind Kind = Kinds::Binary;
  static constexpr llvm::StringRef MimeType = "application/x-unknown";
  static constexpr llvm::StringRef Compression = "none";

  std::set<ObjectID> objects() const;

  void deserialize(const std::map<const ObjectID *, llvm::ArrayRef<char>> Data);
  std::map<ObjectID, revng::pypeline::Buffer>
  serialize(const std::vector<const ObjectID *> Objects) const;
  void setIsDisposable();
  void disposeIfPossible();

  bool verify() const { return true; }

public:
  auto &getStorage() { return Storage; }
};

class AppendFooPipe {
public:
  static constexpr llvm::StringRef Name = "AppendFooPipe";
  using Arguments = TypeList<revng::pypeline::PipeArgument<"Container", "">>;

  const std::string StaticConfiguration;

  AppendFooPipe(llvm::StringRef Config) : StaticConfiguration(Config.str()) {}

  revng::pypeline::PipeOutput run(const Model &TheModel,
                                  const revng::pypeline::Request &Incoming,
                                  const revng::pypeline::Request &Outgoing,
                                  llvm::StringRef Configuration,
                                  StringContainer &Container);
};

class CreateEntryPipe {
private:
  using Access = revng::pypeline::Access;

public:
  static constexpr llvm::StringRef Name = "CreateEntryPipe";
  using Arguments = TypeList<
    revng::pypeline::PipeArgument<"Container", "", Access::Write>>;

  const std::string StaticConfiguration;

  CreateEntryPipe(llvm::StringRef Config) : StaticConfiguration(Config.str()) {}

  revng::pypeline::PipeOutput run(const Model &TheModel,
                                  const revng::pypeline::Request &Incoming,
                                  const revng::pypeline::Request &Outgoing,
                                  llvm::StringRef Configuration,
                                  RootStringContainer &Container);
};

class AppendFooPipe2 {
public:
  static constexpr llvm::StringRef Name = "AppendFooPipe2";
  using Arguments = TypeList<revng::pypeline::PipeArgument<"Container", "">>;

  const std::string StaticConfiguration;

  AppendFooPipe2(llvm::StringRef Config) : StaticConfiguration(Config.str()) {}

  revng::pypeline::PipeOutput run(const Model &TheModel,
                                  const revng::pypeline::Request &Incoming,
                                  const revng::pypeline::Request &Outgoing,
                                  llvm::StringRef Configuration,
                                  RootStringContainer &Container);
};

class AppendFooLibAnalysis {
public:
  static constexpr llvm::StringRef Name = "AppendFooLibAnalysis";

  llvm::Error run(Model &TheModel,
                  const revng::pypeline::Request &Incoming,
                  llvm::StringRef Configuration,
                  StringContainer &Container);
};

class AppendFooLibAnalysis2 {
public:
  static constexpr llvm::StringRef Name = "AppendFooLibAnalysis2";

  llvm::Error run(Model &TheModel,
                  const revng::pypeline::Request &Incoming,
                  llvm::StringRef Configuration,
                  RootStringContainer &Container);
};
