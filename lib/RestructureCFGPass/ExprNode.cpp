//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

// revng includes
#include <revng/Support/Debug.h>

// Local libraries includes
#include "revng-c/RestructureCFGPass/ExprNode.h"

void ExprNode::deleteExprNode(ExprNode *E) {
  switch (E->getKind()) {
  case NodeKind::NK_Atomic:
    delete static_cast<AtomicNode *>(E);
    break;
  case NodeKind::NK_Not:
    delete static_cast<NotNode *>(E);
    break;
  case NodeKind::NK_And:
    delete static_cast<AndNode *>(E);
    break;
  case NodeKind::NK_Or:
    delete static_cast<OrNode *>(E);
    break;
  }
}
