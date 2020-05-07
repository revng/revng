//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <algorithm>
#include <compare>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <type_traits>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/Support/Debug.h"

#include "DLAHelpers.h"
#include "DLAStep.h"
#include "DLATypeSystem.h"

using namespace llvm;

static Logger<> Log("dla-make-layouts");

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

void Layout::deleteLayout(Layout *L) {
  switch (getKind(L)) {
  case LayoutKind::Struct:
    delete static_cast<StructLayout *>(L);
    break;
  case LayoutKind::Union:
    delete static_cast<UnionLayout *>(L);
    break;
  case LayoutKind::Array:
    delete static_cast<ArrayLayout *>(L);
    break;
  case LayoutKind::Base:
    delete static_cast<BaseLayout *>(L);
    break;
  case LayoutKind::Padding:
    delete static_cast<PaddingLayout *>(L);
    break;
  default:
    revng_unreachable("Unexpected LayoutKind");
  }
}

std::strong_ordering Layout::structuralOrder(const Layout *A, const Layout *B) {
  revng_assert(nullptr != A and nullptr != B);

  if (auto Cmp = A->Kind <=> B->Kind; Cmp != 0)
    return Cmp;

  auto Kind = A->Kind;

  switch (Kind) {

  case LayoutKind::Struct: {

    auto *StructA = cast<StructLayout>(A);
    auto *StructB = cast<StructLayout>(B);

    if (std::lexicographical_compare(StructA->fields().begin(),
                                     StructA->fields().end(),
                                     StructB->fields().begin(),
                                     StructB->fields().end(),
                                     Layout::structuralLess))
      return std::strong_ordering::less;

    if (std::lexicographical_compare(StructB->fields().begin(),
                                     StructB->fields().end(),
                                     StructA->fields().begin(),
                                     StructA->fields().end(),
                                     Layout::structuralLess))
      return std::strong_ordering::greater;

    return std::strong_ordering::equal;

  } break;

  case LayoutKind::Union: {

    auto *UnionA = cast<UnionLayout>(A);
    auto *UnionB = cast<UnionLayout>(B);

    if (std::lexicographical_compare(UnionA->elements().begin(),
                                     UnionA->elements().end(),
                                     UnionB->elements().begin(),
                                     UnionB->elements().end(),
                                     Layout::structuralLess))
      return std::strong_ordering::less;

    if (std::lexicographical_compare(UnionB->elements().begin(),
                                     UnionB->elements().end(),
                                     UnionA->elements().begin(),
                                     UnionA->elements().end(),
                                     Layout::structuralLess))
      return std::strong_ordering::greater;

    return std::strong_ordering::equal;

  } break;

  case LayoutKind::Array: {
    auto *ArrayA = cast<ArrayLayout>(A);
    auto *ArrayB = cast<ArrayLayout>(B);
    bool hasKnownLength = ArrayA->hasKnownLength();
    auto Cmp = hasKnownLength <=> ArrayB->hasKnownLength();
    if (Cmp != 0)
      return Cmp;

    if (hasKnownLength) {
      Cmp = ArrayA->length() <=> ArrayB->length();
      if (Cmp != 0)
        return Cmp;
    }

    return structuralOrder(ArrayA->getElem(), ArrayB->getElem());
  } break;

  case LayoutKind::Padding:
  case LayoutKind::Base: {
    return A->size() <=> B->size();
  } break;

  default:
    revng_unreachable("Unexpected LayoutKind");
  }

  return std::strong_ordering::equal;
}

void Layout::printText(llvm::raw_ostream &O, const Layout *L, unsigned Indent) {
  llvm::SmallString<8> IndentStr;
  IndentStr.assign(Indent, ' ');
  revng_assert(L->size());
  switch (getKind(L)) {
  case LayoutKind::Padding: {
    auto *Padding = cast<PaddingLayout>(L);
    if (Padding->size() > 1) {
      O << IndentStr << "uint8_t padding [" << Padding->size() << ']';
    } else {
      O << "uint8_t padding";
    }
  } break;
  case LayoutKind::Struct: {
    auto *Struct = cast<StructLayout>(L);
    revng_assert(Struct->numFields() > 1);
    O << IndentStr << "struct {\n";
    for (const Layout *F : Struct->fields()) {
      printText(O, F, Indent + 2);
      O << ";\n";
    }
    O << IndentStr << "}";
  } break;
  case LayoutKind::Union: {
    auto *Union = cast<UnionLayout>(L);
    revng_assert(Union->numElements() > 1);
    O << IndentStr << "union {\n";
    for (const Layout *E : Union->elements()) {
      printText(O, E, Indent + 2);
      O << ";\n";
    }
    O << IndentStr << "}";
  } break;
  case LayoutKind::Array: {
    auto *Array = cast<ArrayLayout>(L);
    printText(O, Array->getElem(), Indent);
    O << '[';
    if (Array->hasKnownLength())
      O << Array->length();
    else
      O << ' ';
    O << ']';
  } break;
  case LayoutKind::Base: {
    auto *Base = cast<BaseLayout>(L);
    auto Size = Base->size();
    revng_assert(Size);
    bool IsPowerOf2 = (Size & (Size - 1)) == 0;
    revng_assert(IsPowerOf2);
    O << IndentStr << "uint" << (8 * Size) << "_t";
  } break;
  default:
    revng_unreachable("Unexpected LayoutKind");
  }
}

void Layout::printGraphic(llvm::raw_ostream &O,
                          const Layout *L,
                          unsigned Indent) {
  auto PendingUnionsWithOffsets = printGraphicElem(O, L, Indent);
  if (not PendingUnionsWithOffsets.empty()) {
    for (const auto &[L, Off] : PendingUnionsWithOffsets) {
      auto *U = cast<UnionLayout>(L);
      for (const Layout *Elem : U->elements()) {
        O << '\n';
        printGraphic(O, Elem, Indent + Off);
      }
    }
  }
}

llvm::SmallVector<std::pair<const Layout *, unsigned>, 8>
Layout::printGraphicElem(llvm::raw_ostream &O,
                         const Layout *L,
                         unsigned Indent,
                         unsigned Offset) {
  O << std::string(Indent, ' ');
  auto Size = L->size();
  revng_assert(Size);

  llvm::SmallVector<std::pair<const Layout *, unsigned>, 8> Res;
  switch (getKind(L)) {
  case LayoutKind::Padding: {
    O << std::string(Size, '-');
  } break;
  case LayoutKind::Base: {
    std::string N = std::to_string(Size);
    revng_assert(N.size() == 1);
    O << std::string(Size, N[0]);
  } break;
  case LayoutKind::Struct: {
    auto *Struct = cast<StructLayout>(L);
    revng_assert(Struct->numFields() > 1);
    Layout::layout_size_t TotSize = 0ULL;
    for (const Layout *F : Struct->fields()) {
      auto Tmp = printGraphicElem(O, F, 0, Offset + TotSize);
      Res.reserve(Res.size() + Tmp.size());
      Res.insert(Res.end(), Tmp.begin(), Tmp.end());
      TotSize += F->size();
    }
  } break;
  case LayoutKind::Union: {
    auto *Union = cast<UnionLayout>(L);
    revng_assert(Union->numElements() > 1);
    O << std::string(Size, 'U');
    Res.push_back(std::make_pair(L, Indent + Offset));
  } break;
  case LayoutKind::Array: {
    auto *Array = cast<ArrayLayout>(L);
    auto ElemSize = Array->getElem()->size();
    revng_assert(ElemSize);
    revng_assert(ElemSize <= Size);
    if (Array->hasKnownLength()) {
      auto Len = Array->length();
      for (decltype(Len) I = 0; I < Len; ++I) {
        auto Tmp = printGraphicElem(O,
                                    Array->getElem(),
                                    0,
                                    Offset + (ElemSize * I));
        Res.reserve(Res.size() + Tmp.size());
        Res.insert(Res.end(), Tmp.begin(), Tmp.end());
      }
    } else {
      auto Tmp = printGraphicElem(O, Array->getElem(), 0, Offset);
      Res.reserve(Res.size() + Tmp.size());
      Res.insert(Res.end(), Tmp.begin(), Tmp.end());
      O << std::string(Size - ElemSize, '|');
    }
  } break;
  default:
    revng_unreachable("Unexpected LayoutKind");
  }
  return Res;
}

using DeleteLayout = std::integral_constant<decltype(Layout::deleteLayout) &,
                                            Layout::deleteLayout>;

using UniqueLayout = std::unique_ptr<Layout, DeleteLayout>;

static bool uniqueStructLess(const UniqueLayout &A, const UniqueLayout &B) {
  auto *APtr = A.get();
  auto *BPtr = B.get();
  if (nullptr == APtr or nullptr == BPtr) {
    if (APtr == BPtr)
      return false;
    return nullptr == APtr;
  }
  return Layout::structuralLess(APtr, BPtr);
}

using uniqueStructLessT = std::integral_constant<decltype(uniqueStructLess) &,
                                                 uniqueStructLess>;

using LayoutSet = std::set<UniqueLayout, uniqueStructLessT>;

template<typename T, typename... Args>
UniqueLayout makeUniqueLayout(Args &&... A) {
  return UniqueLayout(new T(std::forward<Args &&>(A)...), DeleteLayout());
}

template<typename T, typename... Args>
Layout *createLayout(LayoutSet &S, Args &&... A) {
  auto U = makeUniqueLayout<T>(std::forward<Args &&>(A)...);
  return S.insert(std::move(U)).first->get();
}

using LTSN = LayoutTypeSystemNode;

static Layout *makeInstanceChildLayout(Layout *ChildType,
                                       const OffsetExpression &OE,
                                       LayoutSet &Layouts) {
  // We ignore all the layouts at negative offsets for now.
  if (OE.Offset < 0LL)
    return nullptr;

  // If we have trip counts we have an array of children of type ChildType,
  // otherwise ChildType already points to the right child type.
  revng_assert(OE.Strides.size() == OE.TripCounts.size());
  if (not OE.TripCounts.empty()) {
    Layout *Inner = ChildType;
    for (const auto &[TC, S] : llvm::zip(OE.TripCounts, OE.Strides)) {

      // Don't handle non-positive strides for now.
      if (S <= 0LL)
        return nullptr;

      Layout::layout_size_t StrideSize = (Layout::layout_size_t)(S);

      // For now, we don't handle stuff that for which the size of the element
      // is larger than the stride size
      if (StrideSize < Inner->size())
        return nullptr;

      // If the stride (StrideSize) is larger than the size of the inner
      // element, we need to reserve space after each element, using
      // padding.
      if (StrideSize > Inner->size()) {
        StructLayout::fields_container_t StructFields;
        StructFields.push_back(Inner);
        Layout::layout_size_t PadSize = StrideSize - Inner->size();
        Layout *Padding = createLayout<PaddingLayout>(Layouts, PadSize);
        StructFields.push_back(Padding);
        Inner = createLayout<StructLayout>(Layouts, std::move(StructFields));
      }
      // Create the real array of Inner elements.
      Inner = createLayout<ArrayLayout>(Layouts, Inner, S, TC);
    }
    ChildType = Inner;
  }

  revng_assert(OE.Offset >= 0LL);
  if (OE.Offset > 0LL) {
    // Create padding to insert before the field, according to the
    // offset.
    ArrayLayout::length_t Len = OE.Offset;
    // Create the struct with the padding prepended to the field.
    StructLayout::fields_container_t StructFields;
    StructFields.push_back(createLayout<PaddingLayout>(Layouts, Len));
    StructFields.push_back(ChildType);
    ChildType = createLayout<StructLayout>(Layouts, std::move(StructFields));
  }
  return ChildType;
}

static Layout *makeLayout(const LayoutTypeSystem &TS,
                          const LTSN *N,
                          std::map<const LTSN *, Layout *> &LayoutCTypes,
                          LayoutSet &Layouts) {

  revng_assert(not LayoutCTypes.count(N));

  UnionLayout::elements_container_t UFlds;
  for (const Use *U : N->L.Accesses) {
    const auto AccessSize = getLoadStoreSizeFromPtrOpUse(TS, U);
    revng_log(Log, "Access: " << AccessSize);
    UFlds.insert(createLayout<BaseLayout>(Layouts, AccessSize));
  }

  // Look at all the instance-of edges and inheritance edges all together
  bool InheritsFromOther = false;
  for (auto &[Child, EdgeTag] : children_edges<const LTSN *>(N)) {

    revng_log(Log, "Child ID: " << Child->ID);

    // Ignore children with size == 0
    auto ChildLayoutIt = LayoutCTypes.find(Child);
    if (ChildLayoutIt == LayoutCTypes.end())
      continue;

    Layout *ChildType = ChildLayoutIt->second;

    switch (EdgeTag->getKind()) {

    case TypeLinkTag::LK_Instance: {
      revng_log(Log, "Instance");
      const OffsetExpression &OE = EdgeTag->getOffsetExpr();
      revng_log(Log, "Has Offset: " << OE.Offset);
      ChildType = makeInstanceChildLayout(ChildType, OE, Layouts);
    } break;

    case TypeLinkTag::LK_Inheritance: {
      revng_log(Log, "Inheritance");
      // Treated as instance at offset 0, but can only have one
      revng_assert(not InheritsFromOther);
      InheritsFromOther = true;
    } break;

    default:
      revng_unreachable("unexpected edge");
    }

    // Bail out if we have not constructed a union field, because it means
    // that this is not a supported case yet.
    if (nullptr != ChildType)
      UFlds.insert(ChildType);
  }

  // This layout has no useful access or outgoing edges that can build the
  // type. Just skip it for now until we support handling richer edges and
  // emitting richer types
  if (UFlds.empty())
    return nullptr;

  Layout *CreatedLayout = (UFlds.size() > 1ULL) ?
                            createLayout<UnionLayout>(Layouts, UFlds) :
                            *UFlds.begin();

  LayoutCTypes[N] = CreatedLayout;
  return CreatedLayout;
}

static bool makeLayouts(const LayoutTypeSystem &TS) {
  revng_assert(TS.verifyDAG() and TS.verifyInheritanceTree());

  std::map<const LTSN *, Layout *> LayoutCTypes;
  LayoutSet Layouts;

  std::set<const LTSN *> Visited;
  for (LTSN *Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    if (not isRoot(Root))
      continue;

    for (const LTSN *N : post_order_ext(Root, Visited)) {
      // Leaves need to have ValidLayouts, otherwise they should have been
      // trimmed by PruneLayoutNodesWithoutLayout
      revng_assert(not isLeaf(N) or hasValidLayout(N));
      Layout *LN = makeLayout(TS, N, LayoutCTypes, Layouts);
      if (nullptr == LN) {
        llvm::dbgs() << "\nNode ID: " << N->ID << " Type: Empty\n";
        continue;
      }
      llvm::dbgs() << "\nNode ID: " << N->ID << " Type: ";
      Layout::printText(llvm::dbgs(), LN);
      llvm::dbgs() << ";\n";
      Layout::printGraphic(llvm::dbgs(), LN);
      llvm::dbgs() << '\n';
    }
  }
  return true;
};

bool MakeLayouts::runOnTypeSystem(LayoutTypeSystem &TS) {
  if (Log.isEnabled())
    TS.dumpDotOnFile("final.dot");
  return makeLayouts(TS);
}

} // end namespace dla
