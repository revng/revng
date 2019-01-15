#ifndef REVNGC_IRASTTYPETRANSLATION_H
#define REVNGC_IRASTTYPETRANSLATION_H

#include <llvm/ADT/Optional.h>

#include <clang/AST/Type.h>

namespace llvm {
class GlobalVariable;
class Value;
} // end namespace llvm

namespace IRASTTypeTranslation {

clang::QualType getQualType(const llvm::Value *I, clang::ASTContext &C);

clang::QualType
getQualType(const llvm::GlobalVariable *G, clang::ASTContext &C);

clang::QualType getQualType(const llvm::Type *T, clang::ASTContext &C);

} // end namespace IRASTTypeTranslation

#endif // REVNGC_IRASTTYPETRANSLATION_H
