#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>

#include "llvm/ADT/Twine.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Support/Debug.h"

template<typename T>
char *typeID() {
  static char ID;
  return &ID;
};

class TupleTreeKeyWrapper {
protected:
  void *Pointer;

protected:
  TupleTreeKeyWrapper(void *Pointer) : Pointer(Pointer) {}

public:
  TupleTreeKeyWrapper() : Pointer(nullptr) {}

  TupleTreeKeyWrapper &operator=(const TupleTreeKeyWrapper &Other) {
    if (&Other != this) {
      Other.clone(this);
    }
    return *this;
  }

  TupleTreeKeyWrapper(const TupleTreeKeyWrapper &Other) { *this = Other; }

  TupleTreeKeyWrapper &operator=(TupleTreeKeyWrapper &&Other) {
    if (&Other != this) {
      Other.clone(this);
    }
    return *this;
  }

  TupleTreeKeyWrapper(TupleTreeKeyWrapper &&Other) { *this = Other; }

  virtual ~TupleTreeKeyWrapper(){};
  virtual bool operator==(const TupleTreeKeyWrapper &) const {
    revng_assert(Pointer == nullptr);
    return true;
  }

  virtual std::strong_ordering operator<=>(const TupleTreeKeyWrapper &) const {
    revng_assert(Pointer == nullptr);
    return std::strong_ordering::greater;
  }

  virtual bool matches(const TupleTreeKeyWrapper &) const {
    revng_assert(Pointer == nullptr);
    return true;
  }

  virtual char *id() const {
    revng_assert(Pointer == nullptr);
    return nullptr;
  }

  virtual void clone(TupleTreeKeyWrapper *Target) const {
    revng_assert(Pointer == nullptr);
  }

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
template<typename T, bool LastFieldIsKind = false>
class ConcreteTupleTreeKeyWrapper : public TupleTreeKeyWrapper {
private:
  static char ID;

public:
  T *get() const { return reinterpret_cast<T *>(Pointer); }

public:
  template<typename... Args>
  ConcreteTupleTreeKeyWrapper(Args... A) : TupleTreeKeyWrapper(new T(A...)) {}

  ~ConcreteTupleTreeKeyWrapper() override {
    delete reinterpret_cast<T *>(Pointer);
  }

  bool operator==(const TupleTreeKeyWrapper &Other) const override {
    if (id() == Other.id()) {
      using ThisType = const ConcreteTupleTreeKeyWrapper &;
      auto *OtherPointer = static_cast<ThisType>(Other).get();
      return *get() == *OtherPointer;
    } else {
      return false;
    }
  }

  std::strong_ordering
  operator<=>(const TupleTreeKeyWrapper &Other) const override {
    if (id() == Other.id()) {
      using ThisType = const ConcreteTupleTreeKeyWrapper &;
      const auto &OtherKey = *static_cast<ThisType>(Other).get();
      const auto &ThisKey = *get();
      if (ThisKey < OtherKey)
        return std::strong_ordering::less;
      if (OtherKey < ThisKey)
        return std::strong_ordering::greater;
      return std::strong_ordering::equal;
    } else {
      return id() <=> Other.id();
    }
  }

  bool matches(const TupleTreeKeyWrapper &Other) const override {
    if (id() != Other.id())
      return false;

    if constexpr (LastFieldIsKind) {
      // Compare kinds
      using ThisType = const ConcreteTupleTreeKeyWrapper &;
      const auto *OtherPointer = static_cast<ThisType>(Other).get();
      constexpr auto Index = std::tuple_size_v<T> - 1;
      return std::get<Index>(*get()) == std::get<Index>(*OtherPointer);
    } else {
      revng_assert(*get() == T());
      return true;
    }
  }

  char *id() const override { return typeID<T>(); }

  void clone(TupleTreeKeyWrapper *Target) const override {
    Target->~TupleTreeKeyWrapper();
    new (Target) ConcreteTupleTreeKeyWrapper(*get());
  }
};

class TupleTreePath {
private:
  std::vector<TupleTreeKeyWrapper> Storage;

public:
  TupleTreePath() = default;

  TupleTreePath &operator=(TupleTreePath &&) = default;
  TupleTreePath(TupleTreePath &&) = default;

  TupleTreePath &operator=(const TupleTreePath &Other) {
    if (&Other != this) {
      Storage.resize(Other.size());
      for (auto [ThisElement, OtherElement] :
           llvm::zip(Storage, Other.Storage)) {
        static_assert(std::is_reference_v<decltype(ThisElement)>);
        OtherElement.clone(&ThisElement);
      }
    }

    return *this;
  }

  TupleTreePath(const TupleTreePath &Other) { *this = Other; }

public:
  template<typename T, bool FirstIsKind = false, typename... Args>
  void emplace_back(Args... A) {
    using ConcreteWrapper = ConcreteTupleTreeKeyWrapper<T, FirstIsKind>;
    static_assert(sizeof(ConcreteWrapper) == sizeof(TupleTreeKeyWrapper));
    Storage.resize(Storage.size() + 1);
    new (&Storage.back()) ConcreteWrapper(A...);
  }

  template<typename T>
  void push_back(const T &Obj) {
    emplace_back<T>(Obj);
  }

  void pop_back() { Storage.pop_back(); }

  void resize(size_t NewSize) { Storage.resize(NewSize); }

  TupleTreeKeyWrapper &operator[](size_t Index) { return Storage[Index]; }
  const TupleTreeKeyWrapper &operator[](size_t Index) const {
    return Storage[Index];
  }
  bool operator==(const TupleTreePath &Other) const = default;
  std::strong_ordering operator<=>(const TupleTreePath &Other) const {
    if (Storage < Other.Storage)
      return std::strong_ordering::less;
    if (Other.Storage < Storage)
      return std::strong_ordering::greater;
    return std::strong_ordering::equal;
  }

  // TODO: should return ArrayRef<const TupleTreeKeyWrapper>
  llvm::ArrayRef<TupleTreeKeyWrapper> toArrayRef() const { return { Storage }; }

  bool isPrefixOf(const TupleTreePath &Other) const {
    if (size() > Other.size())
      return false;

    for (size_t I = 0; I < size(); I++)
      if (Storage[I] != Other.Storage[I])
        return false;

    return true;
  }

public:
  size_t size() const { return Storage.size(); }

  bool empty() const { return Storage.empty(); }
};
