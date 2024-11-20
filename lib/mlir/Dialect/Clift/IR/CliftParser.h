#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/OpImplementation.h"

#include "revng/Support/Assert.h"
#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.h"

namespace mlir::clift {

std::map<uint64_t, void *> &getAsmRecursionMap();

template<typename ObjectT>
concept hasExplicitSize = requires(const typename ObjectT::ImplType::KeyTy
                                     &Key) { Key.Definition->Size; };

template<typename ObjectT>
void printCompositeType(AsmPrinter &Printer, ObjectT Object) {
  const uint64_t ID = Object.getImpl()->getID();

  Printer << Object.getMnemonic();
  Printer << "<id = ";
  Printer << ID;

  auto &RecursionMap = getAsmRecursionMap();
  const auto [Iterator, Inserted] = RecursionMap.try_emplace(ID);

  if (not Inserted) {
    Printer << ">";
    return;
  }

  const auto EraseGuard = llvm::make_scope_exit([&]() {
    RecursionMap.erase(Iterator);
  });

  Printer << ", name = ";
  Printer << "\"" << Object.getName() << "\"";

  if constexpr (hasExplicitSize<ObjectT>) {
    Printer << ", ";
    Printer.printKeywordOrString("size");
    Printer << " = ";
    Printer << Object.getByteSize();
  }

  Printer << ", fields = [";
  Printer.printStrippedAttrOrType(Object.getImpl()->getSubobjects());
  Printer << "]>";
}

template<typename ObjectT>
ObjectT parseCompositeType(AsmParser &Parser, const size_t MinSubobjects) {
  const auto OnUnexpectedToken = [&](const llvm::StringRef Name) -> ObjectT {
    Parser.emitError(Parser.getCurrentLocation(),
                     "Expected " + Name + " while parsing mlir "
                       + ObjectT::getMnemonic() + "type");
    return {};
  };

  if (Parser.parseLess())
    return OnUnexpectedToken("<");

  if (Parser.parseKeyword("id").failed())
    return OnUnexpectedToken("keyword 'id'");

  if (Parser.parseEqual().failed())
    return OnUnexpectedToken("=");

  uint64_t ID;
  if (Parser.parseInteger(ID).failed())
    return OnUnexpectedToken("<integer>");

  auto &RecursionMap = getAsmRecursionMap();
  const auto [Iterator, Inserted] = RecursionMap.try_emplace(ID);

  if (not Inserted) {
    revng_assert(Iterator->second != nullptr);

    if (Parser.parseGreater().failed())
      return OnUnexpectedToken(">");

    return ObjectT(static_cast<typename ObjectT::ImplType *>(Iterator->second));
  }

  const auto EraseGuard = llvm::make_scope_exit([&]() {
    RecursionMap.erase(Iterator);
  });

  if (Parser.parseComma().failed())
    return OnUnexpectedToken(",");

  if (Parser.parseKeyword("name").failed())
    return OnUnexpectedToken("keyword 'name'");

  if (Parser.parseEqual().failed())
    return OnUnexpectedToken("=");

  std::string OptionalName = "";
  if (Parser.parseOptionalString(&OptionalName).failed())
    return OnUnexpectedToken("<string>");

  uint64_t Size;
  if constexpr (hasExplicitSize<ObjectT>) {
    if (Parser.parseComma().failed())
      return OnUnexpectedToken(",");

    if (Parser.parseKeyword("size").failed())
      return OnUnexpectedToken("keyword 'size'");

    if (Parser.parseEqual().failed())
      return OnUnexpectedToken("=");

    if (Parser.parseInteger(Size).failed())
      return OnUnexpectedToken("<uint64_t>");
  }

  if (Parser.parseComma().failed())
    return OnUnexpectedToken(",");

  if (Parser.parseKeyword("fields").failed())
    return OnUnexpectedToken("keyword 'fields'");

  if (Parser.parseEqual().failed())
    return OnUnexpectedToken("=");

  if (Parser.parseLSquare().failed())
    return OnUnexpectedToken("[");

  revng_assert(Iterator->second == nullptr);
  ObjectT Object = ObjectT::get(Parser.getContext(), ID);
  Iterator->second = Object.getImpl();

  using SubobjectType = typename ObjectT::ImplType::SubobjectTy;
  using SubobjectVectorType = llvm::SmallVector<SubobjectType>;
  using SubobjectParserType = mlir::FieldParser<SubobjectVectorType>;
  mlir::FailureOr<SubobjectVectorType> Subobjects(SubobjectVectorType{});

  if (MinSubobjects > 0 or Parser.parseOptionalRSquare().failed()) {
    Subobjects = SubobjectParserType::parse(Parser);

    if (mlir::failed(Subobjects)) {
      Parser.emitError(Parser.getCurrentLocation(),
                       "in type with ID " + std::to_string(ID)
                         + ": failed to parse class type parameter 'fields' "
                           "which is to be a "
                           "`llvm::ArrayRef<mlir::clift::FieldAttr>`");
      return {};
    }

    if (Subobjects->size() < MinSubobjects) {
      Parser.emitError(Parser.getCurrentLocation(),
                       llvm::Twine(ObjectT::getMnemonic())
                         + " requires at least " + llvm::Twine(MinSubobjects)
                         + " fields");
      return {};
    }

    if (Parser.parseRSquare().failed())
      return OnUnexpectedToken("]");
  }

  if (Parser.parseGreater().failed())
    return OnUnexpectedToken(">");

  if constexpr (hasExplicitSize<ObjectT>)
    Object.define(OptionalName, Size, *Subobjects);
  else
    Object.define(OptionalName, *Subobjects);

  return Object;
}

} // namespace mlir::clift
