#include "tree_sitter/parser.h"
#include <stdbool.h>

enum TokenType {
  BLOCK_COMMENT_CONTENT,
  ERROR_SENTINEL,
};

void *tree_sitter_shimlang_external_scanner_create(void) {
  return NULL;
}

void tree_sitter_shimlang_external_scanner_destroy(void *payload) {
}

unsigned tree_sitter_shimlang_external_scanner_serialize(void *payload, char *buffer) {
  return 0;
}

void tree_sitter_shimlang_external_scanner_deserialize(void *payload, const char *buffer, unsigned length) {
}

static void advance(TSLexer *lexer) {
  lexer->advance(lexer, false);
}

bool tree_sitter_shimlang_external_scanner_scan(void *payload, TSLexer *lexer, const bool *valid_symbols) {
  if (valid_symbols[ERROR_SENTINEL]) {
    return false;
  }

  if (!valid_symbols[BLOCK_COMMENT_CONTENT]) {
    return false;
  }

  // We are inside a block comment (after the opening /*).
  // We need to scan the content, handling nested /* ... */ pairs,
  // and stop just before the final closing */.
  int depth = 1;
  bool has_content = false;

  while (!lexer->eof(lexer)) {
    if (lexer->lookahead == '/') {
      // Mark the end before potentially consuming into nested comment
      lexer->mark_end(lexer);
      advance(lexer);
      if (!lexer->eof(lexer) && lexer->lookahead == '*') {
        advance(lexer);
        depth++;
        has_content = true;
        continue;
      }
      has_content = true;
      continue;
    }
    if (lexer->lookahead == '*') {
      lexer->mark_end(lexer);
      advance(lexer);
      if (!lexer->eof(lexer) && lexer->lookahead == '/') {
        depth--;
        if (depth == 0) {
          // Don't consume the closing */, let the grammar handle it
          // mark_end was called before consuming *, so we stop before it
          lexer->result_symbol = BLOCK_COMMENT_CONTENT;
          return has_content;
        }
        advance(lexer);
        has_content = true;
        continue;
      }
      has_content = true;
      continue;
    }
    advance(lexer);
    has_content = true;
  }

  return false;
}
