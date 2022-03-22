#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <type_traits>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
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
  virtual llvm::Error serialize(llvm::raw_ostream &OS) const = 0;
  virtual llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) = 0;
  virtual void clear() = 0;
  virtual llvm::Error storeToDisk(llvm::StringRef Path) const;
  virtual llvm::Error loadFromDisk(llvm::StringRef Path);
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
