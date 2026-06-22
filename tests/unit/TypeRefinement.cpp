//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE TypeRefinement
bool init_unit_test();
#include <list>
#include <unordered_set>

#include "boost/test/unit_test.hpp"

#include "revng/CliftTransforms/TypeRefinement.h"

using namespace clift;

namespace {

static MutableStringAttr makeMutableStringAttr(mlir::MLIRContext *Context,
                                               llvm::StringRef Key) {
  return MutableStringAttr::get(Context, mlir::StringAttr::get(Context, Key));
}

static StructType
makeStruct(mlir::MLIRContext *Context, std::string Handle, bool Opaque) {
  auto Attr = StructAttr::get(Context,
                              Handle,
                              makeMutableStringAttr(Context, Handle + "-n"),
                              makeMutableStringAttr(Context, Handle + "-c"),
                              /*Size=*/8,
                              Opaque,
                              /*Fields=*/{},
                              /*Attributes=*/{});

  return StructType::get(Attr);
}

static UnionType makeUnion(mlir::MLIRContext *Context) {
  std::string Handle = "union";
  std::string FieldHandle = Handle + "-field";

  FieldAttr Fields[1] = {
    FieldAttr::get(Context,
                   FieldHandle,
                   makeMutableStringAttr(Context, FieldHandle + "-n"),
                   makeMutableStringAttr(Context, FieldHandle + "-c"),
                   /*Offset=*/0,
                   IntegerType::get(Context, IntegerKind::Generic, 8))
  };

  auto Attr = UnionAttr::get(Context,
                             Handle,
                             makeMutableStringAttr(Context, Handle + "-n"),
                             makeMutableStringAttr(Context, Handle + "-c"),
                             Fields,
                             /*Attributes=*/{});

  return UnionType::get(Attr);
}

static EnumType makeEnum(mlir::MLIRContext *Context,
                         std::string Handle,
                         mlir::Type Underlying) {
  std::string EnumeratorHandle = Handle + "-enumerator";

  EnumFieldAttr Fields[1] = {
    EnumFieldAttr::get(Context,
                       EnumeratorHandle,
                       makeMutableStringAttr(Context, EnumeratorHandle + "-n"),
                       makeMutableStringAttr(Context, EnumeratorHandle + "-c"),
                       /*RawValue=*/0)
  };

  auto Attr = EnumAttr::get(Context,
                            Handle,
                            makeMutableStringAttr(Context, Handle + "-n"),
                            makeMutableStringAttr(Context, Handle + "-c"),
                            Underlying,
                            Fields);

  return EnumType::get(Attr);
}

static PointerType makePointer(mlir::Type PointeeType) {
  return PointerType::get(PointeeType, /*PointerSize=*/8);
}

static TypedefType makeTypedef(mlir::Type Type) {
  mlir::MLIRContext *Context = Type.getContext();
  const void *Ptr = Type.getAsOpaquePointer();

  std::string Handle;
  {
    llvm::raw_string_ostream Out(Handle);
    Out << "typedef-" << Ptr;
  }

  auto Attr = TypedefAttr::get(Context,
                               Handle,
                               makeMutableStringAttr(Context, Handle + "-n"),
                               makeMutableStringAttr(Context, Handle + "-c"),
                               Type);

  return TypedefType::get(Attr);
}

struct Node {
  mlir::Type Archetype;
  mlir::Type InnerTypedef;
  mlir::Type OuterTypedef;

  std::list<Node> Children;

  explicit Node(mlir::Type Archetype) :
    Archetype(Archetype),
    InnerTypedef(makeTypedef(Archetype)),
    OuterTypedef(makeTypedef(InnerTypedef)) {}

  Node &put(mlir::Type Archetype) { return Children.emplace_back(Archetype); }

  bool contains(const Node &Other) const {
    bool IsContained = false;
    if (&Other != this) {
      visit([&](const Node &Child) {
        if (&Child == &Other)
          IsContained = true;
      });
    }
    return IsContained;
  }

  template<typename InvocableT>
  void visit(InvocableT &&Invocable) const {
    std::invoke(Invocable, *this);

    for (const Node &Child : Children)
      Child.visit(std::forward<InvocableT>(Invocable));
  }
};

} // namespace

BOOST_AUTO_TEST_CASE(TypeRefinementOrdering) {
  mlir::MLIRContext ContextObject;
  ContextObject.loadDialect<CliftDialect>();

  mlir::MLIRContext *Context = &ContextObject;

  auto const makeInteger = [&](IntegerKind Kind) {
    return IntegerType::get(Context, Kind, /*Size=*/8);
  };

  Node Opaque(makeStruct(Context, "opaque", /*Opaque=*/true));

  Node &Struct = Opaque.put(makeStruct(Context, "struct", /*Opaque=*/false));
  Node &Union = Opaque.put(makeUnion(Context));

  Node &Generic = Opaque.put(makeInteger(IntegerKind::Generic));
  Node &PtrOrNum = Generic.put(makeInteger(IntegerKind::PointerOrNumber));

  Node &Pointer1 = PtrOrNum.put(makePointer(Generic.Archetype));
  Node &Pointer2 = PtrOrNum.put(makePointer(Pointer1.Archetype));

  Node &FloatingPoint = Generic.put(FloatType::get(Context, 8));

  Node &Number = PtrOrNum.put(makeInteger(IntegerKind::Number));
  Node &Signed = Number.put(makeInteger(IntegerKind::Signed));
  Node &Unsigned = Number.put(makeInteger(IntegerKind::Unsigned));

  Node &SignedEnum = Signed.put(makeEnum(Context,
                                         "signed-enum",
                                         Signed.Archetype));

  Node &UnsignedEnum = Unsigned.put(makeEnum(Context,
                                             "unsigned-enum",
                                             Unsigned.Archetype));

  Opaque.visit([&](const Node &A) {
    Opaque.visit([&](const Node &B) {
      auto Ordering = compareTypeRefinement(A.Archetype, B.Archetype);
      if (Ordering == 0) {
        revng_assert(&A == &B);
        revng_assert(compareTypeRefinement(A.Archetype, B.InnerTypedef) < 0);
        revng_assert(compareTypeRefinement(A.Archetype, B.OuterTypedef) < 0);
        revng_assert(compareTypeRefinement(A.InnerTypedef, B.OuterTypedef) < 0);
      } else if (Ordering > 0) {
        revng_assert(B.contains(A));
        revng_assert(compareTypeRefinement(A.OuterTypedef, B.Archetype) > 0);
      } else if (Ordering < 0) {
        revng_assert(A.contains(B));
        revng_assert(compareTypeRefinement(A.Archetype, B.OuterTypedef) < 0);
      } else {
        revng_assert(not A.contains(B));
        revng_assert(not B.contains(A));

        auto O = compareTypeRefinement(A.OuterTypedef, B.OuterTypedef);
        revng_assert(not(O < 0 or O == 0 or O > 0));
      }
    });
  });
}
