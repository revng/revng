#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/Twine.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Support/Debug.h"

template<typename T>
char *typeID() {
  static char ID;
  return &ID;
};

class Any {
protected:
  void *Pointer;

protected:
  Any(void *Pointer) : Pointer(Pointer) {}

public:
  Any() : Pointer(nullptr) {}

  Any(const Any &Other) { Other.clone(this); }
  Any(Any &&Other) { Other.clone(this); }
  Any &operator=(const Any &Other) {
    Other.clone(this);
    return *this;
  }
  Any &operator=(Any &&Other) {
    Other.clone(this);
    return *this;
  }

  virtual ~Any(){};
  virtual bool operator==(const Any &) const {
    revng_assert(Pointer == nullptr);
    return true;
  }

  virtual char *id() const {
    revng_assert(Pointer == nullptr);
    return nullptr;
  }

  virtual void clone(Any *Target) const { revng_assert(Pointer == nullptr); }

  template<typename T>
  bool isa() const {
    return id() == typeID<T>();
  }

  template<typename T>
  T *tryGet() const {
    if (isa<T>())
      return reinterpret_cast<T *>(Pointer);
    else
      return nullptr;
  }

  template<typename T>
  T &get() const {
    if (T *Result = tryGet<T>())
      return *Result;
    else
      revng_abort();
  }
};

// TODO: optimize integral types
template<typename T>
class ConcreteAny : public Any {
private:
  static char ID;

private:
  T *get() const { return reinterpret_cast<T *>(Pointer); }

public:
  template<typename... Args>
  ConcreteAny(Args... A) : Any(new T(A...)) {}

  ~ConcreteAny() override { delete reinterpret_cast<T *>(Pointer); }

  bool operator==(const Any &Other) const override {
    return (id() == Other.id()
            and *get() == *static_cast<const ConcreteAny &>(Other).get());
  }

  char *id() const override { return typeID<T>(); }

  void clone(Any *Target) const override { new (Target) ConcreteAny(*get()); }
};

//
// KeyTraits
//
// WIP: rename this file
// WIP: rename the following
using KeyInt = Any;
// using KeyIntVector = std::vector<KeyInt>;

class KeyIntVector {
private:
  std::vector<KeyInt> Storage;

public:
  KeyIntVector() = default;
  KeyIntVector(KeyIntVector &&Other) = default;
  KeyIntVector &operator=(KeyIntVector &&Other) = default;

  KeyIntVector(const KeyIntVector &Other) { *this = Other; }

  KeyIntVector &operator=(const KeyIntVector &Other) {
    Storage.resize(Other.size());
    for (auto [ThisElement, OtherElement] : llvm::zip(Storage, Other.Storage)) {
      static_assert(std::is_reference_v<decltype(ThisElement)>);
      OtherElement.clone(&ThisElement);
    }

    return *this;
  }

public:
  template<typename T, typename... Args>
  void emplace_back(Args... A) {
    static_assert(sizeof(ConcreteAny<T>) == sizeof(Any));
    Storage.resize(Storage.size() + 1);
    new (&Storage.back()) ConcreteAny<T>(A...);
  }

  template<typename T>
  void push_back(const T &Obj) {
    emplace_back<T>(Obj);
  }

  void pop_back() { Storage.pop_back(); }

  void resize(size_t NewSize) { Storage.resize(NewSize); }

  KeyInt &operator[](size_t Index) { return Storage[Index]; }
  const KeyInt &operator[](size_t Index) const { return Storage[Index]; }
  bool operator==(const KeyIntVector &Other) const {
    return Storage == Other.Storage;
  }

  // TODO: should return ArrayRef<const Any>
  llvm::ArrayRef<Any> toArrayRef() const { return { Storage }; }

public:
  size_t size() const { return Storage.size(); }
};
