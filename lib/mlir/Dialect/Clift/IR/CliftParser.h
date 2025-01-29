#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

#include "mlir/IR/OpImplementation.h"

#include "revng/Support/Assert.h"
#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.h"

namespace mlir::clift {

llvm::SmallPtrSet<void *, 8> &getAsmRecursionMap();

template<typename ObjectT>
concept hasExplicitSize = requires(const typename ObjectT::ImplType::KeyTy
                                     &Key) { Key.Definition->Size; };

template<typename ObjectT>
void printCompositeType(AsmPrinter &Printer, ObjectT Object) {
  Printer << Object.getMnemonic();
  Printer << "<unique_handle = \"";
  llvm::printEscapedString(Object.getImpl()->getUniqueHandle(),
                           Printer.getStream());
  Printer << "\"";

  auto &RecursionMap = getAsmRecursionMap();
  const auto [Iterator, Inserted] = RecursionMap.insert(Object.getImpl());

  if (not Inserted) {
    Printer << ">";
    return;
  }

  const auto EraseGuard = llvm::make_scope_exit([&]() {
    RecursionMap.erase(Object.getImpl());
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
  auto ObjectLoc = Parser.getCurrentLocation();

  const auto OnUnexpectedToken = [&](const llvm::StringRef Name) -> ObjectT {
    Parser.emitError(Parser.getCurrentLocation(),
                     "Expected " + Name + " while parsing mlir "
                       + ObjectT::getMnemonic() + "type");
    return {};
  };

  if (Parser.parseLess())
    return OnUnexpectedToken("<");

  if (Parser.parseKeyword("unique_handle").failed())
    return OnUnexpectedToken("keyword 'unique_handle'");

  if (Parser.parseEqual().failed())
    return OnUnexpectedToken("=");

  std::string UniqueHandle;
  if (Parser.parseString(&UniqueHandle).failed())
    return OnUnexpectedToken("<integer>");

  ObjectT Object = ObjectT::get(Parser.getContext(), UniqueHandle);

  auto &RecursionMap = getAsmRecursionMap();
  const auto [Iterator, Inserted] = RecursionMap.insert(Object.getImpl());

  if (not Inserted) {
    if (Parser.parseGreater().failed())
      return OnUnexpectedToken(">");

    return Object;
  }

  const auto EraseGuard = llvm::make_scope_exit([&]() {
    RecursionMap.erase(Object.getImpl());
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

  using SubobjectType = typename ObjectT::ImplType::SubobjectTy;
  using SubobjectVectorType = llvm::SmallVector<SubobjectType>;
  using SubobjectParserType = mlir::FieldParser<SubobjectVectorType>;
  mlir::FailureOr<SubobjectVectorType> Subobjects(SubobjectVectorType{});

  if (MinSubobjects > 0 or Parser.parseOptionalRSquare().failed()) {
    Subobjects = SubobjectParserType::parse(Parser);

    if (mlir::failed(Subobjects)) {
      Parser.emitError(Parser.getCurrentLocation(),
                       "in type with unique handle '" + UniqueHandle
                         + "': failed to parse class type parameter 'fields' "
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

  auto DefineObject = [&](const auto &...Args) -> ObjectT {
    auto EmitError = [&] { return Parser.emitError(ObjectLoc); };
    if (ObjectT::verify(EmitError, UniqueHandle, Args...).failed())
      return {};

    Object.define(Args...);
    return Object;
  };

  if constexpr (hasExplicitSize<ObjectT>)
    return DefineObject(OptionalName, Size, *Subobjects);
  else
    return DefineObject(OptionalName, *Subobjects);
}

} // namespace mlir::clift
