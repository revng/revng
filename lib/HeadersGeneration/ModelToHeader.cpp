//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
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
  Header << getNamedCInstance(model::QualifiedType{ Segment.Type(), {} }, S, B)
         << ";\n";
}

/// Print all type definitions for the types in the model
static void printTypeDefinitions(const model::Binary &Model,
                                 const TypeInlineHelper &TheTypeInlineHelper,
                                 ptml::PTMLIndentedOstream &Header,
                                 ptml::PTMLCBuilder &B,
                                 QualifiedTypeNameMap &AdditionalTypeNames,
                                 const ModelToHeaderOptions &Options) {
  std::set<const model::Type *> StackTypes, EmptyInlineTypes;
  if (not Options.DisableTypeInlining)
    StackTypes = TheTypeInlineHelper.collectStackTypes(Model);

  DependencyGraph Dependencies = buildDependencyGraph(Model.Types());
  const auto &TypeNodes = Dependencies.TypeNodes();
  auto &ToInline = Options.DisableTypeInlining ?
                     EmptyInlineTypes :
                     TheTypeInlineHelper.getTypesToInline();
  std::set<const TypeDependencyNode *> Defined;

  for (const auto *Root : Dependencies.nodes()) {
    revng_log(Log, "======== PostOrder " << getNodeLabel(Root));

    for (const auto *Node : llvm::post_order_ext(Root, Defined)) {
      revng_log(Log, "== visiting " << getNodeLabel(Node));
      for (const auto *Child :
           llvm::children<const TypeDependencyNode *>(Node)) {
        revng_log(Log, "= child " << getNodeLabel(Child));
        if (Defined.contains(Child))
          revng_log(Log, "      DEFINED");
        else
          revng_log(Log, "      NOT DEFINED");
      }

      if (StackTypes.contains(Node->T)) {
        if (Options.DisableTypeInlining) {
          revng_log(Log, "      PRINTED STACK TYPE");
        } else {
          revng_log(Log, "      IGNORED STACK TYPE");
          continue;
        }
      }

      const model::Type *NodeT = Node->T;
      const auto DeclKind = Node->K;
      constexpr auto TypeName = TypeNode::Kind::TypeName;
      constexpr auto FullType = TypeNode::Kind::FullType;

      if (Options.TypesToOmit.contains(NodeT))
        continue;

      if (DeclKind == FullType) {

        // When emitting a full definition we also want to emit a forward
        // declaration first, if it wasn't already emitted somewhere else.
        if (Defined.insert(TypeNodes.at({ NodeT, TypeName })).second
            and not ToInline.contains(NodeT)) {
          printDeclaration(Log,
                           *NodeT,
                           Header,
                           B,
                           Model,
                           AdditionalTypeNames,
                           ToInline);
        }

        if (not declarationIsDefinition(NodeT)
            and not ToInline.contains(NodeT)) {
          // For all inlinable types that we have seen them yet produce forward
          // declaration.
          if ((isa<model::UnionType>(NodeT) or isa<model::StructType>(NodeT))
              and not ToInline.contains(NodeT)) {
            auto TypesToInline = TheTypeInlineHelper
                                   .getTypesToInlineInTypeTy(Model, NodeT);
            for (auto *Type : TypesToInline) {
              revng_assert(isCandidateForInline(Type));
              printDeclaration(Log,
                               *Type,
                               Header,
                               B,
                               Model,
                               AdditionalTypeNames,
                               ToInline);
            }
          }

          printDefinition(Log,
                          *NodeT,
                          Header,
                          B,
                          Model,
                          AdditionalTypeNames,
                          ToInline);
        }

        // This is always a full type definition
        Defined.insert(TypeNodes.at({ NodeT, FullType }));
      } else {
        if (not ToInline.contains(NodeT)) {
          if (not Options.TypesToOmit.contains(Node->T)) {
            printDeclaration(Log,
                             *NodeT,
                             Header,
                             B,
                             Model,
                             AdditionalTypeNames,
                             ToInline);
          }
          Defined.insert(TypeNodes.at({ NodeT, TypeName }));
        }

        // For primitive types the forward declaration we emit is also a full
        // definition, so we need to keep track of this.
        if (isa<model::PrimitiveType>(NodeT))
          Defined.insert(TypeNodes.at({ NodeT, FullType }));

        // For struct, enums and unions the forward declaration is just a
        // forward declaration, without body.

        // TypedefType, RawFunctionType and CABIFunctionType are emitted in C
        // as typedefs, so they don't represent fully defined types, but just
        // names, unless all the types they depend from are also fully
        // defined, but that happens when DeclKind == FullType, not here.
      }
    }
    revng_log(Log, "====== PostOrder DONE");
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
      TypeInlineHelper TheTypeInlineHelper(Model);

      printTypeDefinitions(Model,
                           TheTypeInlineHelper,
                           Header,
                           B,
                           AdditionalTypeNames,
                           Options);
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
