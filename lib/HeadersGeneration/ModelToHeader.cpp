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
#include "revng/Model/Type.h"
#include "revng/Pipeline/Location.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"

#include "revng-c/Backend/DecompiledCCodeIndentation.h"
#include "revng-c/HeadersGeneration/ModelToHeader.h"
#include "revng-c/Pipes/Ranks.h"
#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/Support/PTMLC.h"
#include "revng-c/TypeNames/ModelToPTMLTypeHelpers.h"
#include "revng-c/TypeNames/ModelTypeNames.h"

#include "DependencyGraph.h"

using llvm::isa;

using QualifiedTypeNameMap = std::map<model::QualifiedType, std::string>;
using TypeToNumOfRefsMap = std::unordered_map<const model::Type *, unsigned>;
using GraphInfo = TypeInlineHelper::GraphInfo;

static Logger<> Log{ "model-to-header" };

static void printSegmentsTypes(const model::Segment &Segment,
                               ptml::PTMLIndentedOstream &Header,
                               const ptml::PTMLCBuilder &B) {
  auto S = B.getLocationDefinition(Segment);

  model::QualifiedType SegmentType;
  if (Segment.Type().empty()) {
    // If the segment has not type, we emit it as an array of bytes.
    const model::Binary *Model = Segment.Type().getRoot();
    model::TypePath
      Byte = Model->getPrimitiveType(model::PrimitiveTypeKind::Generic, 1);
    model::Qualifier Array = model::Qualifier::createArray(Segment
                                                             .VirtualSize());
    SegmentType = model::QualifiedType{ Byte, { Array } };
  } else {
    SegmentType = model::QualifiedType{ Segment.Type(), {} };
  }

  Header << getNamedCInstance(SegmentType, S, B) << ";\n";
}

/// Print all type definitions for the types in the model
static void printTypeDefinitions(const model::Binary &Model,
                                 ptml::PTMLIndentedOstream &Header,
                                 ptml::PTMLCBuilder &B,
                                 QualifiedTypeNameMap &AdditionalTypeNames,
                                 const ModelToHeaderOptions &Options) {

  std::set<const model::Type *> TypesToInlineInStacks;
  std::set<const model::Type *> ToInline;
  if (not Options.DisableTypeInlining) {
    TypeInlineHelper TheTypeInlineHelper(Model);
    TypesToInlineInStacks = TheTypeInlineHelper.collectTypesInlinableInStacks();
    ToInline = TheTypeInlineHelper.getTypesToInline();
  }

  if (Log.isEnabled()) {
    revng_log(Log, "TypesToInlineInStacks: {");
    {
      LoggerIndent Indent{ Log };
      for (const model::Type *T : TypesToInlineInStacks)
        revng_log(Log, T->ID());
    }
    revng_log(Log, "}");
    revng_log(Log, "ToInline: {");
    {
      LoggerIndent Indent{ Log };
      for (const model::Type *T : ToInline)
        revng_log(Log, T->ID());
    }
    revng_log(Log, "}");
    revng_log(Log, "TypesToInlineInStacks should be a subset of ToInline");
  }

  DependencyGraph Dependencies = buildDependencyGraph(Model.Types());
  const auto &TypeNodes = Dependencies.TypeNodes();

  std::set<const TypeDependencyNode *> Defined;
  for (const auto *Root : Dependencies.nodes()) {
    revng_log(Log, "PostOrder from Root:" << getNodeLabel(Root));

    for (const auto *Node : llvm::post_order_ext(Root, Defined)) {

      LoggerIndent PostOrderIndent{ Log };
      revng_log(Log, "post_order visiting: " << getNodeLabel(Node));

      const model::Type *NodeT = Node->T;
      const auto DeclKind = Node->K;

      if (TypesToInlineInStacks.contains(NodeT)) {
        if (Options.DisableTypeInlining) {
          revng_log(Log, "Printed Stack Type");
        } else {
          revng_log(Log, "Ignored Stack Type");
          continue;
        }
      }

      if (Options.TypesToOmit.contains(NodeT)) {
        revng_log(Log, "Omitted");
        continue;
      }

      constexpr auto Declaration = TypeNode::Kind::Declaration;

      if (DeclKind == Declaration) {
        revng_log(Log, "Declaration");

        // Print the declaration. Notice that the forward declarations are
        // emitted even for inlined types, because it's only the full definition
        // that will be inlined.
        revng_log(Log, "printDeclaration");
        printDeclaration(Log,
                         *NodeT,
                         Header,
                         B,
                         Model,
                         AdditionalTypeNames,
                         ToInline);

      } else {
        revng_log(Log, "Definition");

        revng_assert(Defined.contains(TypeNodes.at({ NodeT, Declaration })));
        if (not declarationIsDefinition(NodeT)
            and not ToInline.contains(NodeT)) {
          revng_log(Log, "printDefinition");
          printDefinition(Log,
                          *NodeT,
                          Header,
                          B,
                          Model,
                          AdditionalTypeNames,
                          ToInline);
        }
      }
    }
    revng_log(Log, "PostOrder DONE");
  }
}

bool dumpModelToHeader(const model::Binary &Model,
                       llvm::raw_ostream &Out,
                       const ModelToHeaderOptions &Options) {
  using PTMLCBuilder = ptml::PTMLCBuilder;
  using Scopes = ptml::PTMLCBuilder::Scopes;

  PTMLCBuilder B(Options.GeneratePlainC);
  ptml::PTMLIndentedOstream Header(Out, DecompiledCCodeIndentation, true);
  {
    auto Scope = B.getTag(ptml::tags::Div).scope(Header);

    Header << B.getPragmaOnce();
    Header << "\n";
    Header << B.getIncludeAngle("stdint.h");
    Header << B.getIncludeAngle("stdbool.h");
    Header << B.getIncludeQuote("revng-primitive-types.h");
    Header << B.getIncludeQuote("revng-attributes.h");
    Header << "\n";

    if (Options.PostIncludes.size())
      Header << Options.PostIncludes << "\n";

    Header << B.getDirective(PTMLCBuilder::Directive::IfNotDef) << " "
           << B.getNullTag() << "\n"
           << B.getDirective(PTMLCBuilder::Directive::Define) << " "
           << B.getNullTag() << " (" << B.getZeroTag() << ")\n"
           << B.getDirective(PTMLCBuilder::Directive::EndIf) << "\n";

    if (not Model.Types().empty()) {
      auto Foldable = B.getScope(Scopes::TypeDeclarations)
                        .scope(Out,
                               /* Newline */ true);

      Header << B.getLineComment("===============");
      Header << B.getLineComment("==== Types ====");
      Header << B.getLineComment("===============");
      Header << '\n';
      QualifiedTypeNameMap AdditionalTypeNames;

      printTypeDefinitions(Model, Header, B, AdditionalTypeNames, Options);
    }

    if (not Model.Functions().empty()) {
      auto Foldable = B.getScope(Scopes::FunctionDeclarations)
                        .scope(Out,
                               /* Newline */ true);
      Header << B.getLineComment("===================");
      Header << B.getLineComment("==== Functions ====");
      Header << B.getLineComment("===================");
      Header << '\n';
      for (const model::Function &MF : Model.Functions()) {
        if (Options.FunctionsToOmit.contains(MF.Entry()))
          continue;

        const model::Type *FT = MF.prototype(Model).get();
        if (Options.TypesToOmit.contains(FT))
          continue;

        auto FName = MF.name();

        if (Log.isEnabled()) {
          helpers::BlockComment CommentScope(Header, B.isGenerateTagLessPTML());
          Header << "Analyzing Model function " << FName << "\n";
          serialize(Header, MF);
          Header << "Prototype\n";
          serialize(Header, *FT);
        }

        printFunctionPrototype(*FT, MF, Header, B, Model, false);
        Header << ";\n";
      }
    }

    if (not Model.ImportedDynamicFunctions().empty()) {
      auto Foldable = B.getScope(Scopes::DynamicFunctionDeclarations)
                        .scope(Out, /* Newline */ true);
      Header << B.getLineComment("==============================="
                                 "===");
      Header << B.getLineComment("==== ImportedDynamicFunctions "
                                 "====");
      Header << B.getLineComment("==============================="
                                 "===");
      Header << '\n';
      for (const model::DynamicFunction &MF :
           Model.ImportedDynamicFunctions()) {
        const model::Type *FT = MF.prototype(Model).get();
        revng_assert(FT != nullptr);
        if (Options.TypesToOmit.contains(FT))
          continue;

        auto FName = MF.name();

        if (Log.isEnabled()) {
          helpers::BlockComment CommentScope(Header, B.isGenerateTagLessPTML());
          Header << "Analyzing dynamic function " << FName << "\n";
          serialize(Header, MF);
          Header << "Prototype\n";
          serialize(Header, *FT);
        }
        printFunctionPrototype(*FT, MF, Header, B, Model, false);
        Header << ";\n";
      }
    }

    if (not Model.Segments().empty()) {
      auto Foldable = B.getScope(Scopes::SegmentDeclarations)
                        .scope(Out,
                               /* Newline */ true);
      Header << B.getLineComment("==================");
      Header << B.getLineComment("==== Segments ====");
      Header << B.getLineComment("==================");
      Header << '\n';
      for (const model::Segment &Segment : Model.Segments())
        printSegmentsTypes(Segment, Header, B);
      Header << '\n';
    }
  }
  return true;
}
