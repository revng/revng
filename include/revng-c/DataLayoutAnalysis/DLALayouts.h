#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <algorithm>
#include <compare>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <set>

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"

#include "revng/Support/Debug.h"

namespace llvm {
class Value;
} // end namespace llvm

namespace dla {

class Layout {
public:
  enum class LayoutKind { Padding, Base, Array, Struct, Union };
  using layout_size_t = uint64_t;

private:
  LayoutKind Kind;
  layout_size_t Size;

public:
  static LayoutKind getKind(const Layout *L) { return L->Kind; }
  LayoutKind getKind() const { return Kind; }

  static void deleteLayout(Layout *L);

  static void
  printText(llvm::raw_ostream &O, const Layout *L, unsigned Indent = 0);

  static void
  printGraphic(llvm::raw_ostream &O, const Layout *L, unsigned Indent = 0);

  static llvm::SmallVector<std::pair<const Layout *, unsigned>, 8>
  printGraphicElem(llvm::raw_ostream &O,
                   const Layout *L,
                   unsigned Indent = 0,
                   unsigned Offset = 0);

  static std::strong_ordering structuralOrder(const Layout *A, const Layout *B);

  static bool structuralLess(const Layout *A, const Layout *B) {
    return structuralOrder(A, B) < 0;
  }

  using structLessT = std::integral_constant<decltype(structuralLess) &,
                                             structuralLess>;

  Layout(const Layout &) = default;
  Layout(Layout &&) = default;

  Layout &operator=(const Layout &) = default;
  Layout &operator=(Layout &&) = default;

  ~Layout() = default;

  layout_size_t size() const { return Size; }

protected:
  Layout(LayoutKind K, layout_size_t S) : Kind(K), Size(S) {
    revng_assert(S != 0ULL);
  }
};

class PaddingLayout : public Layout {
public:
  static bool classof(const Layout *L) {
    return getKind(L) == LayoutKind::Padding;
  }

  PaddingLayout(layout_size_t S) : Layout(LayoutKind::Padding, S) {}
  PaddingLayout() = delete;

  PaddingLayout(const PaddingLayout &) = default;
  PaddingLayout(PaddingLayout &&) = default;

  PaddingLayout &operator=(const PaddingLayout &) = default;
  PaddingLayout &operator=(PaddingLayout &&) = default;

  ~PaddingLayout() = default;
};

class BaseLayout : public Layout {
public:
  static bool classof(const Layout *L) {
    return getKind(L) == LayoutKind::Base;
  }

  BaseLayout(layout_size_t S) : Layout(LayoutKind::Base, S) {}
  BaseLayout() = delete;

  BaseLayout(const BaseLayout &) = default;
  BaseLayout(BaseLayout &&) = default;

  BaseLayout &operator=(const BaseLayout &) = default;
  BaseLayout &operator=(BaseLayout &&) = default;

  ~BaseLayout() = default;
};

class UnionLayout : public Layout {

public:
  using elements_container_t = std::set<Layout *, structLessT>;
  using elements_num_t = elements_container_t::size_type;

private:
  elements_container_t Elems;

  static layout_size_t getMaxSize(const elements_container_t &Elements) {
    layout_size_t S = 0U;
    for (Layout *E : Elements) {
      S = std::max(S, E->size());
    }
    return S;
  }

public:
  static bool classof(const Layout *L) {
    return getKind(L) == LayoutKind::Union;
  }

  UnionLayout() = delete;

  UnionLayout(const UnionLayout &) = default;
  UnionLayout(UnionLayout &&) = default;

  UnionLayout &operator=(const UnionLayout &) = default;
  UnionLayout &operator=(UnionLayout &&) = default;

  ~UnionLayout() = default;

  UnionLayout(const elements_container_t &E) :
    Layout(LayoutKind::Union, getMaxSize(E)), Elems(E) {
    revng_assert(Elems.size() > 1);
  }

  UnionLayout(elements_container_t &&E) :
    Layout(LayoutKind::Union, getMaxSize(E)), Elems(std::move(E)) {
    revng_assert(Elems.size() > 1);
  }

  const elements_container_t &elements() const { return Elems; }
  elements_num_t numElements() const { return Elems.size(); }
};

class ArrayLayout : public Layout {
public:
  using length_t = uint64_t;

private:
  std::optional<length_t> NElems;
  Layout *ElemLayout;

public:
  static bool classof(const Layout *L) {
    return getKind(L) == LayoutKind::Array;
  }

  ArrayLayout(Layout *E, length_t N) :
    Layout(LayoutKind::Array, N * E->size()), NElems(N), ElemLayout(E) {}

  ArrayLayout(Layout *E, layout_size_t ElSize, std::optional<length_t> NEls) :
    Layout(LayoutKind::Array,
           NEls.has_value() ? (NEls.value() * ElSize) : ElSize),
    NElems(NEls),
    ElemLayout(E) {}

  ArrayLayout(const ArrayLayout &) = default;
  ArrayLayout(ArrayLayout &&) = default;

  ArrayLayout &operator=(const ArrayLayout &) = default;
  ArrayLayout &operator=(ArrayLayout &&) = default;

  ArrayLayout() = delete;
  ~ArrayLayout() = default;

  Layout *getElem() const { return ElemLayout; }

  bool hasKnownLength() const { return NElems.has_value(); }

  length_t length() const {
    revng_assert(hasKnownLength());
    return NElems.value();
  }
};

class StructLayout : public Layout {
public:
  using fields_container_t = llvm::SmallVector<Layout *, 8>;
  using fields_num_t = fields_container_t::size_type;

private:
  fields_container_t Fields;

  static layout_size_t getTotSize(const llvm::SmallVectorImpl<Layout *> &Flds) {
    const auto AccumulateSize = [](const auto &Flds) {
      return std::accumulate(Flds.begin(),
                             Flds.end(),
                             0ULL,
                             [](layout_size_t S, const Layout *L) {
                               return S + L->size();
                             });
    };
    return AccumulateSize(Flds);
  }

public:
  static bool classof(const Layout *L) {
    return getKind(L) == LayoutKind::Struct;
  }

  StructLayout(const llvm::SmallVectorImpl<Layout *> &Flds) :
    Layout(LayoutKind::Struct, getTotSize(Flds)),
    Fields(llvm::iterator_range(Flds.begin(), Flds.end())) {
    revng_assert(Fields.size() > 1U);
  }

  StructLayout(llvm::SmallVectorImpl<Layout *> &&Flds) :
    Layout(LayoutKind::Struct, getTotSize(Flds)), Fields(std::move(Flds)) {
    revng_assert(Fields.size() > 1U);
  }

  StructLayout() = delete;

  StructLayout(const StructLayout &) = default;
  StructLayout(StructLayout &&) = default;

  StructLayout &operator=(const StructLayout &) = default;
  StructLayout &operator=(StructLayout &&) = default;

  ~StructLayout() = default;

  const fields_container_t &fields() const { return Fields; }
  fields_num_t numFields() const { return Fields.size(); }
};

using DeleteLayout = std::integral_constant<decltype(Layout::deleteLayout) &,
                                            Layout::deleteLayout>;

using UniqueLayout = std::unique_ptr<Layout, DeleteLayout>;

template<typename T, typename... Args>
UniqueLayout makeUniqueLayout(Args &&...A) {
  return UniqueLayout(new T(std::forward<Args &&>(A)...), DeleteLayout());
}

using LayoutVector = std::vector<UniqueLayout>;

template<typename T, typename... Args>
Layout *createLayout(LayoutVector &S, Args &&...A) {
  auto U = makeUniqueLayout<T>(std::forward<Args &&>(A)...);
  return S.emplace_back(std::move(U)).get();
}

/// A representation of a pointer to a type.
class LayoutTypePtr {
  const llvm::Value *V;
  unsigned FieldIdx;

public:
  static constexpr unsigned fieldNumNone = std::numeric_limits<unsigned>::max();

  explicit LayoutTypePtr(const llvm::Value *Val, unsigned Idx = fieldNumNone) :
    V(Val), FieldIdx(Idx) {}

  LayoutTypePtr() = default;
  ~LayoutTypePtr() = default;
  LayoutTypePtr(const LayoutTypePtr &) = default;
  LayoutTypePtr(LayoutTypePtr &&) = default;
  LayoutTypePtr &operator=(const LayoutTypePtr &) = default;
  LayoutTypePtr &operator=(LayoutTypePtr &&) = default;

  std::strong_ordering operator<=>(const LayoutTypePtr &Other) const = default;

  unsigned fieldNum() const { return FieldIdx; }

  void print(llvm::raw_ostream &Out) const;

  const llvm::Value &getValue() const { return *V; }

  bool isEmpty() const { return (V == nullptr); }
}; // end class LayoutTypePtr

using ValueLayoutMap = std::map<LayoutTypePtr, Layout *>;
using LayoutTypePtrVect = std::vector<LayoutTypePtr>;

} // end namespace dla
