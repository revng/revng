//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Helpers.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Pipeline/Location.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"

#include "revng-c/Backend/DecompiledCCodeIndentation.h"
#include "revng-c/HeadersGeneration/PTMLHeaderBuilder.h"
#include "revng-c/Pipes/Ranks.h"
#include "revng-c/Support/ModelHelpers.h"

static Logger<> Log{ "model-to-header" };

bool ptml::HeaderBuilder::printModelHeader(const model::Binary &Binary) {
  B.collectInlinableTypes(Binary);

  auto Scope = B.getIndentedTag(ptml::tags::Div);

  std::string Includes = B.getPragmaOnce() + "\n"
                         + B.getIncludeAngle("stdint.h")
                         + B.getIncludeAngle("stdbool.h")
                         + B.getIncludeQuote("primitive-types.h")
                         + B.getIncludeQuote("attributes.h") + "\n";
  B.append(std::move(Includes));

  if (not Configuration.PostIncludeSnippet.empty())
    B.append(Configuration.PostIncludeSnippet + "\n"s);

  std::string Defines = B.getDirective(CBuilder::Directive::IfNotDef) + " "
                        + B.getNullTag() + "\n"
                        + B.getDirective(CBuilder::Directive::Define) + " "
                        + B.getNullTag() + " (" + B.getZeroTag() + ")\n"
                        + B.getDirective(CBuilder::Directive::EndIf) + "\n";
  B.append(std::move(Defines));

  if (not Binary.TypeDefinitions().empty()) {
    auto Foldable = B.getIndentedScope(CBuilder::Scopes::TypeDeclarations,
                                       /* Newline = */ true);

    B.appendLineComment("===============");
    B.appendLineComment("==== Types ====");
    B.appendLineComment("===============");
    B.append("\n");

    B.printTypeDefinitions(Binary);
  }

  if (not Binary.Functions().empty()) {
    auto Foldable = B.getIndentedScope(CBuilder::Scopes::FunctionDeclarations,
                                       /* Newline = */ true);
    B.appendLineComment("===================");
    B.appendLineComment("==== Functions ====");
    B.appendLineComment("===================");
    B.append("\n");
    for (const model::Function &MF : Binary.Functions()) {
      if (Configuration.FunctionsToOmit.contains(MF.Entry()))
        continue;

      const auto &FT = *Binary.prototypeOrDefault(MF.prototype());
      if (B.Configuration.TypesToOmit.contains(FT.key()))
        continue;

      if (Log.isEnabled()) {
        helpers::BlockComment CommentScope = B.getBlockCommentScope();
        B.append("Emitting a model function '" + MF.name().str().str() + "':\n"
                 + MF.toString() + "Its prototype is:\n" + FT.toString());
      }

      B.printFunctionPrototype(FT, MF, /* SingleLine = */ false);
      B.append(";\n");
    }
  }

  if (not Binary.ImportedDynamicFunctions().empty()) {
    auto F = B.getIndentedScope(CBuilder::Scopes::DynamicFunctionDeclarations,
                                /* Newline = */ true);
    B.appendLineComment("==================================");
    B.appendLineComment("==== ImportedDynamicFunctions ====");
    B.appendLineComment("==================================");
    B.append("\n");
    for (const model::DynamicFunction &MF : Binary.ImportedDynamicFunctions()) {
      const auto &FT = *Binary.prototypeOrDefault(MF.prototype());
      if (B.Configuration.TypesToOmit.contains(FT.key()))
        continue;

      if (Log.isEnabled()) {
        helpers::BlockComment CommentScope = B.getBlockCommentScope();
        B.append("Emitting a dynamic function '" + MF.name().str().str()
                 + "':\n" + MF.toString() + "Its prototype is:\n"
                 + FT.toString());
      }

      B.printFunctionPrototype(FT, MF, /* SingleLine = */ false);
      B.append(";\n");
    }
  }

  if (not Binary.Segments().empty()) {
    auto Foldable = B.getIndentedScope(CBuilder::Scopes::SegmentDeclarations,
                                       /* Newline = */ true);
    B.appendLineComment("==================");
    B.appendLineComment("==== Segments ====");
    B.appendLineComment("==================");
    B.append("\n");
    for (const model::Segment &Segment : Binary.Segments())
      B.printSegmentType(Segment);
    B.append("\n");
  }

  return true;
}
