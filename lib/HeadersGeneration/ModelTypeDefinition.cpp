//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng-c/Backend/DecompiledCCodeIndentation.h"
#include "revng-c/HeadersGeneration/ModelToHeader.h"
#include "revng-c/HeadersGeneration/ModelTypeDefinition.h"
#include "revng-c/TypeNames/ModelToPTMLTypeHelpers.h"

static Logger<> Log{ "model-type-definition" };

std::string dumpModelTypeDefinition(const model::Binary &Model,
                                    model::Type::Key Key) {
  std::string Result;

  const UpcastablePointer<model::Type> &UpcastableType = Model.Types().at(Key);

  llvm::raw_string_ostream Out(Result);
  ptml::PTMLIndentedOstream PTMLOut(Out, DecompiledCCodeIndentation, true);
  ptml::PTMLCBuilder B(true);

  std::map<model::QualifiedType, std::string> AdditionalNames;
  const std::set<const model::Type *> TypesToInline;

  printDefinition(Log,
                  *UpcastableType,
                  PTMLOut,
                  B,
                  Model,
                  AdditionalNames,
                  TypesToInline,
                  {},
                  {},
                  true);

  return Result;
}
