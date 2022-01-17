#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <type_traits>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
namespace pipeline {

class SavableObjectBase {
private:
  const char *ID;

public:
  SavableObjectBase(const char *ID) : ID(ID) {}

  virtual ~SavableObjectBase() {}

public:
  const char *getID() const { return ID; }

public:
  virtual llvm::Error storeToDisk(llvm::StringRef Path) const = 0;
  virtual llvm::Error loadFromDisk(llvm::StringRef Path) = 0;
};

/// Crtp class to describe a object that can be saved and load from disk
template<typename MostDerived>
class SavableObject : public SavableObjectBase {
public:
  SavableObject() : SavableObjectBase(&MostDerived::ID) {}

  ~SavableObject() override {}

public:
  static bool classof(const SavableObjectBase *T) {
    return T->getID() == &MostDerived::ID;
  }
};

} // namespace pipeline
