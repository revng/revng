//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "DLALayouts.h"

namespace dla {
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

    auto *StructA = llvm::cast<StructLayout>(A);
    auto *StructB = llvm::cast<StructLayout>(B);

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

    auto *UnionA = llvm::cast<UnionLayout>(A);
    auto *UnionB = llvm::cast<UnionLayout>(B);

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
    auto *ArrayA = llvm::cast<ArrayLayout>(A);
    auto *ArrayB = llvm::cast<ArrayLayout>(B);
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
    auto *Padding = llvm::cast<PaddingLayout>(L);
    if (Padding->size() > 1) {
      O << IndentStr << "uint8_t padding [" << Padding->size() << ']';
    } else {
      O << "uint8_t padding";
    }
  } break;
  case LayoutKind::Struct: {
    auto *Struct = llvm::cast<StructLayout>(L);
    revng_assert(Struct->numFields() > 1);
    O << IndentStr << "struct {\n";
    for (const Layout *F : Struct->fields()) {
      printText(O, F, Indent + 2);
      O << ";\n";
    }
    O << IndentStr << "}";
  } break;
  case LayoutKind::Union: {
    auto *Union = llvm::cast<UnionLayout>(L);
    revng_assert(Union->numElements() > 1);
    O << IndentStr << "union {\n";
    for (const Layout *E : Union->elements()) {
      printText(O, E, Indent + 2);
      O << ";\n";
    }
    O << IndentStr << "}";
  } break;
  case LayoutKind::Array: {
    auto *Array = llvm::cast<ArrayLayout>(L);
    printText(O, Array->getElem(), Indent);
    O << '[';
    if (Array->hasKnownLength())
      O << Array->length();
    else
      O << ' ';
    O << ']';
  } break;
  case LayoutKind::Base: {
    auto *Base = llvm::cast<BaseLayout>(L);
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
      auto *U = llvm::cast<UnionLayout>(L);
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
    auto *Struct = llvm::cast<StructLayout>(L);
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
    auto *Union = llvm::cast<UnionLayout>(L);
    revng_assert(Union->numElements() > 1);
    O << std::string(Size, 'U');
    Res.push_back(std::make_pair(L, Indent + Offset));
  } break;
  case LayoutKind::Array: {
    auto *Array = llvm::cast<ArrayLayout>(L);
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

} // end namespace dla
