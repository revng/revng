#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <type_traits>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
namespace Pipeline {

class SavableObjectBase {
private:
  const char *ID;

public:
  SavableObjectBase(const char *ID) : ID(ID) {}
  virtual llvm::Error storeToDisk(llvm::StringRef Path) const = 0;
  virtual llvm::Error loadFromDisk(llvm::StringRef Path) = 0;
  virtual ~SavableObjectBase() {}

  const char *getID() const { return ID; }
};

/// crpt class to describe a object that can be saved and load from disk
template<typename MostDerived>
class SavableObject : public SavableObjectBase {
public:
  SavableObject() : SavableObjectBase(&MostDerived::ID) {}

  ~SavableObject() override {}

  static bool classof(const SavableObjectBase *T) {
    return T->getID() == &MostDerived::ID;
  }
};

} // namespace Pipeline
