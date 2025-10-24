#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>

#include "llvm/ADT/Twine.h"

#include "revng/ADT/STLExtras.h"
#include "revng/ADT/TraitfulAny.h"
#include "revng/Support/Debug.h"

template<typename T>
concept CheckLastFieldIsKind = requires {
  { T::LastFieldIsKind };
  requires T::LastFieldIsKind == true;
};

struct TupleTreeKeyAnyTrait {
  enum class TraitAction {
    Compare,
    Matches
  };

  template<typename T>
  static intptr_t handle(TraitAction Action,
                         revng::TraitfulAny<TupleTreeKeyAnyTrait> *First,
                         revng::TraitfulAny<TupleTreeKeyAnyTrait> *Second) {
    switch (Action) {
    case TraitAction::Compare: {
      intptr_t Result = 0;
      if (First->type_id() == Second->type_id()) {
        const auto &LHS = *revng::any_cast<T>(First);
        const auto &RHS = *revng::any_cast<T>(Second);
        auto ComparisonResult = LHS <=> RHS;
        *reinterpret_cast<std::strong_ordering *>(&Result) = ComparisonResult;
      } else {
        auto ComparisonResult = First->type_id() <=> Second->type_id();
        *reinterpret_cast<std::strong_ordering *>(&Result) = ComparisonResult;
      }
      return Result;
    }

    case TraitAction::Matches: {
      if (First->type_id() != Second->type_id())
        return false;

      if constexpr (CheckLastFieldIsKind<T>) {
        constexpr auto Index = std::tuple_size_v<T> - 1;
        const auto &LHS = *revng::any_cast<T>(First);
        const auto &RHS = *revng::any_cast<T>(Second);
        return std::get<Index>(LHS) == std::get<Index>(RHS);
      }

      revng_assert(*revng::any_cast<T>(First) == T());
      return true;
    }

    default:
      revng_abort();
    }
  }
};

static_assert(sizeof(std::strong_ordering) <= sizeof(intptr_t));

class TupleTreeKeyWrapper : public revng::TraitfulAny<TupleTreeKeyAnyTrait> {
public:
  std::strong_ordering operator<=>(const TupleTreeKeyWrapper &Other) const {
    void *Result = const_cast<TupleTreeKeyWrapper *>(this)
                     ->call(TupleTreeKeyAnyTrait::TraitAction::Compare,
                            const_cast<TupleTreeKeyWrapper *>(&Other));
    return *reinterpret_cast<std::strong_ordering *>(&Result);
  }

  bool matches(const TupleTreeKeyWrapper &Other) const {
    if (not this->has_value())
      return true;
    void *Result = const_cast<TupleTreeKeyWrapper *>(this)
                     ->call(TupleTreeKeyAnyTrait::TraitAction::Matches,
                            const_cast<TupleTreeKeyWrapper *>(&Other));
    return *reinterpret_cast<bool *>(&Result);
  }

  template<typename T>
  const T &get() const {
    return *revng::any_cast<const T>(this);
  }

  template<typename T>
  T &get() {
    return *revng::any_cast<T>(this);
  }

  template<typename T>
  const T *tryGet() const {
    return revng::any_cast<const T>(this);
  }

  template<typename T>
  T *tryGet() {
    return revng::any_cast<T>(this);
  }

  bool operator==(const TupleTreeKeyWrapper &Other) const {
    return (*this <=> Other) == std::strong_ordering::equal;
  }
};

class TupleTreePath {
private:
  // Some performance results:
  //   SmallVector<..., 0>
  //     elapsed_time_total_seconds 332.34s
  //     maximum_resident_set_size: 1466.68M
  //
  //   SmallVector<..., 1>
  //     elapsed_time_total_seconds 322.78s
  //     maximum_resident_set_size: 1471.01M
  //
  //   SmallVector<..., 2>
  //     elapsed_time_total_seconds 323.03s
  //     maximum_resident_set_size: 1476.1M
  //
  //   SmallVector<..., 4>
  //     elapsed_time_total_seconds 320.26s
  //     maximum_resident_set_size: 1483.88M
  llvm::SmallVector<TupleTreeKeyWrapper, 1> Storage;

public:
  TupleTreePath() = default;

  TupleTreePath &operator=(TupleTreePath &&) = default;
  TupleTreePath(TupleTreePath &&) = default;

  TupleTreePath &operator=(const TupleTreePath &Other) {
    if (&Other != this) {
      Storage.resize(Other.size());
      for (auto &&[ThisElement, OtherElement] :
           llvm::zip(Storage, Other.Storage)) {
        static_assert(std::is_reference_v<decltype(ThisElement)>);
        ThisElement = OtherElement;
      }
    }

    return *this;
  }

  TupleTreePath(const TupleTreePath &Other) { *this = Other; }

public:
  template<typename T, typename... Args>
  void emplace_back(Args... A) {
    Storage.emplace_back(std::forward<Args>(A)...);
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
