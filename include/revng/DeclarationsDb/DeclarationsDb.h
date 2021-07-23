#pragma once

// standard include
#include <string>
#include <vector>

// llvm includes
#include "llvm/ADT/SmallString.h"

class sqlite3;

class Parameter {
private:
  std::string Name;
  std::string Loc;

public:
  Parameter(std::string Nm, std::string Location = "") :
    Name(std::move(Nm)), Loc(std::move(Location)) {}

  llvm::StringRef name() const { return Name; }
  llvm::StringRef location() const { return Loc; }

  void setLocation(std::string newLocation) { Loc = std::move(newLocation); }
};

class FunctionDecl {
private:
  llvm::SmallVector<Parameter, 10> Parameters;
  std::string Name;
  std::string LibName;

public:
  FunctionDecl(std::string Nm, std::string LibNm) :
    Name(std::move(Nm)), LibName(std::move(LibNm)) {}

  llvm::SmallVector<Parameter, 10> &getParameters() { return Parameters; }
  const llvm::SmallVector<Parameter, 10> &getParameters() const {
    return Parameters;
  }
  llvm::StringRef name() const { return Name; }
  llvm::StringRef libName() const { return LibName; }
};

class ParameterSaver {
public:
  ParameterSaver(llvm::StringRef Path);
  ~ParameterSaver();
  ParameterSaver(const ParameterSaver &other) = delete;
  ParameterSaver(ParameterSaver &&other) : Db(other.Db) { other.Db = nullptr; }
  ParameterSaver &operator=(const ParameterSaver &other) = delete;
  ParameterSaver &operator=(ParameterSaver &&other) {
    Db = other.Db;
    other.Db = nullptr;
    return *this;
  }

  FunctionDecl getFunction(const std::string &functionName,
                           const std::vector<std::string> &libName);
  bool save(const FunctionDecl &function);

private:
  sqlite3 *Db;
};
