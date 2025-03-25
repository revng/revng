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
  Printer << '<' << '"';
  llvm::printEscapedString(Object.getImpl()->getHandle(), Printer.getStream());
  Printer << '"';

  auto &RecursionMap = getAsmRecursionMap();
  const auto [Iterator, Inserted] = RecursionMap.insert(Object.getImpl());

  if (not Inserted) {
    Printer << ">";
    return;
  }

  const auto EraseGuard = llvm::make_scope_exit([&]() {
    RecursionMap.erase(Object.getImpl());
  });

  if (llvm::StringRef Name = Object.getName(); not Name.empty()) {
    Printer << " as \"";
    llvm::printEscapedString(Name, Printer.getStream());
    Printer << "\"";
  }

  Printer << " : ";
  if constexpr (hasExplicitSize<ObjectT>) {
    Printer << "size(" << Object.getByteSize() << ") ";
  }

  Printer << '{';
  for (auto [I, S] : llvm::enumerate(Object.getImpl()->getSubobjects())) {
    bool PrintColon = false;

    if (I != 0)
      Printer << ", ";

    if constexpr (hasExplicitSize<ObjectT>) {
      Printer << "offset(" << S.getOffset() << ") ";
      PrintColon = true;
    }

    if (llvm::StringRef Name = S.getName(); not Name.empty()) {
      if constexpr (hasExplicitSize<ObjectT>) {
        Printer << " as ";
      }

      Printer << "\"";
      llvm::printEscapedString(Name, Printer.getStream());
      Printer << "\" ";
      PrintColon = true;
    }

    if (PrintColon)
      Printer << ": ";

    Printer << S.getType();
  }
  Printer << '}' << '>';
}

template<typename ObjectT>
ObjectT parseCompositeType(AsmParser &Parser, const size_t MinSubobjects) {
  auto ObjectLoc = Parser.getCurrentLocation();

  if (Parser.parseLess().failed())
    return {};

  std::string Handle;
  if (Parser.parseString(&Handle).failed())
    return {};

  ObjectT Object = ObjectT::get(Parser.getContext(), Handle);

  auto &RecursionMap = getAsmRecursionMap();
  const auto [Iterator, Inserted] = RecursionMap.insert(Object.getImpl());

  if (not Inserted) {
    if (Parser.parseGreater().failed())
      return {};

    return Object;
  }

  const auto EraseGuard = llvm::make_scope_exit([&]() {
    RecursionMap.erase(Object.getImpl());
  });

  std::string Name;
  if (Parser.parseOptionalKeyword("as").succeeded()) {
    if (Parser.parseString(&Name).failed())
      return {};
  }

  if (Parser.parseColon().failed())
    return {};

  uint64_t Size;
  if constexpr (hasExplicitSize<ObjectT>) {
    if (Parser.parseKeyword("size").failed())
      return {};

    if (Parser.parseLParen().failed())
      return {};

    if (Parser.parseInteger(Size).failed())
      return {};

    if (Parser.parseRParen().failed())
      return {};
  }

  llvm::SmallVector<FieldAttr> Fields;
  auto ParseField = [&]() -> ParseResult {
    auto FieldLoc = Parser.getCurrentLocation();
    bool ParseColon = false;

    uint64_t Offset = 0;
    std::string Name;

    if constexpr (hasExplicitSize<ObjectT>) {
      if (Parser.parseKeyword("offset").failed())
        return {};

      if (Parser.parseLParen().failed())
        return {};

      if (Parser.parseInteger(Offset).failed())
        return mlir::failure();

      if (Parser.parseRParen().failed())
        return {};

      if (Parser.parseOptionalKeyword("as").succeeded()) {
        if (Parser.parseString(&Name).failed())
          return {};
      }

      ParseColon = true;
    } else {
      if (Parser.parseOptionalString(&Name).succeeded())
        ParseColon = true;
    }

    if (ParseColon and Parser.parseColon().failed())
      return mlir::failure();

    clift::ValueType Type;
    if (Parser.parseType(Type).failed())
      return mlir::failure();

    auto EmitError = [&] { return Parser.emitError(FieldLoc); };
    if (FieldAttr::verify(EmitError, Offset, Type, Name).failed())
      return mlir::failure();

    Fields.push_back(FieldAttr::get(Parser.getContext(),
                                    Offset,
                                    Type,
                                    std::move(Name)));

    return mlir::success();
  };

  if (Parser
        .parseCommaSeparatedList(OpAsmParser::Delimiter::Braces,
                                 ParseField,
                                 " in field list")
        .failed())
    return {};

  if (Parser.parseGreater().failed())
    return {};

  auto DefineObject = [&](auto &&...Args) -> ObjectT {
    auto EmitError = [&] { return Parser.emitError(ObjectLoc); };
    if (ObjectT::verify(EmitError, Handle, std::as_const(Args)...).failed())
      return {};

    Object.define(std::forward<decltype(Args)>(Args)...);
    return Object;
  };

  if constexpr (hasExplicitSize<ObjectT>)
    return DefineObject(std::move(Name), Size, std::move(Fields));
  else
    return DefineObject(std::move(Name), std::move(Fields));
}

} // namespace mlir::clift
