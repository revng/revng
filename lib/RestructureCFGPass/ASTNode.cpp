/// \file ASTNode.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>
#include <fstream>
#include <iostream>

// Local libraries includes
#include "revng-c/RestructureCFGPass/ASTNode.h"

void CodeNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"box\",color=\"red\"];\n";
}

void IfNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"invhouse\",color=\"blue\"];\n";

  if (this->getThen() != nullptr) {
    ASTFile << "\"" << this->getName() << "\""
        << " -> \"" << this->getThen()->getName() << "\""
        << " [color=green,label=\"then\"];\n";
    this->getThen()->dump(ASTFile);
  }

  if (this->getElse() != nullptr) {
    ASTFile << "\"" << this->getName() << "\""
        << " -> \"" << this->getElse()->getName() << "\""
        << " [color=green,label=\"else\"];\n";
    this->getElse()->dump(ASTFile);
  }
}

void ScsNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"circle\",color=\"black\"];\n";

  revng_assert(this->getBody() != nullptr);
  ASTFile << "\"" << this->getName() << "\""
      << " -> \"" << this->getBody()->getName() << "\""
      << " [color=green,label=\"body\"];\n";
  this->getBody()->dump(ASTFile);
}

void SequenceNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"box\",color=\"black\"];\n";

  for (ASTNode *Successor : this->nodes()) {
    ASTFile << "\"" << this->getName() << "\""
        << " -> \"" << Successor->getName() << "\""
        << " [color=green,label=\"elem\"];\n";
    Successor->dump(ASTFile);
  }
}
