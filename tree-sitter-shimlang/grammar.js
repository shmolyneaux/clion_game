/// <reference types="tree-sitter-cli/dsl" />

module.exports = grammar({
  name: 'shimlang',

  extras: $ => [
    /\s/,
    $.line_comment,
    $.block_comment,
  ],

  word: $ => $.identifier,

  externals: $ => [
    $._block_comment_content,
    $._error_sentinel,
  ],

  conflicts: $ => [
  ],

  rules: {
    source_file: $ => optional($._block_inner),

    _block_inner: $ => choice(
      // Statements only
      repeat1($._statement),
      // Statements followed by a trailing expression
      seq(repeat1($._statement), $._expression),
      // Just a trailing expression
      $._expression,
    ),

    block: $ => seq('{', optional($._block_inner), '}'),

    // Statements
    _statement: $ => choice(
      $.let_statement,
      $.assignment_statement,
      $.attribute_assignment_statement,
      $.index_assignment_statement,
      $.if_statement,
      $.for_statement,
      $.while_statement,
      $.break_statement,
      $.continue_statement,
      $.function_definition,
      $.struct_definition,
      $.return_statement,
      $.expression_statement,
    ),

    let_statement: $ => seq('let', $.identifier, '=', $._expression, ';'),

    assignment_statement: $ => seq($.identifier, '=', $._expression, ';'),

    attribute_assignment_statement: $ => seq(
      $.attribute_expression, '=', $._expression, ';',
    ),

    index_assignment_statement: $ => seq(
      $.index_expression, '=', $._expression, ';',
    ),

    // if as a statement: always appears in statement context
    // We parse if uniformly and distinguish statement vs expression in AST conversion
    if_statement: $ => prec.right(seq(
      'if', field('condition', $._expression), field('consequence', $.block),
      optional(seq('else', field('alternative', $.block))),
    )),

    for_statement: $ => seq('for', $.identifier, 'in', $._expression, $.block),

    while_statement: $ => seq('while', $._expression, $.block),

    break_statement: $ => seq('break', ';'),

    continue_statement: $ => seq('continue', ';'),

    return_statement: $ => choice(
      seq('return', ';'),
      seq('return', $._expression, ';'),
    ),

    expression_statement: $ => seq($._expression, ';'),

    function_definition: $ => seq(
      'fn', field('name', $.identifier), '(', optional($.parameter_list), ')', $.block,
    ),

    struct_definition: $ => seq(
      'struct', field('name', $.identifier), '{',
      optional($.member_list),
      repeat($.method_definition),
      '}',
    ),

    method_definition: $ => seq(
      'fn', field('name', $.identifier), '(', optional($.parameter_list), ')', $.block,
    ),

    member_list: $ => seq(
      $._member, repeat(seq(',', $._member)), optional(','),
    ),

    _member: $ => choice(
      $.required_member,
      $.optional_member,
    ),

    required_member: $ => $.identifier,

    optional_member: $ => seq($.identifier, '=', $._expression),

    parameter_list: $ => seq(
      $._parameter, repeat(seq(',', $._parameter)),
    ),

    _parameter: $ => choice(
      $.required_parameter,
      $.optional_parameter,
    ),

    required_parameter: $ => $.identifier,

    optional_parameter: $ => seq($.identifier, '=', $._expression),

    // Expressions - ordered by precedence (lowest to highest)
    _expression: $ => choice(
      $.if_expression,
      $._logical_or_expression,
    ),

    // if as an expression (used in expression contexts like let x = if ... { ... } else { ... })
    if_expression: $ => prec.right(-1, seq(
      'if', field('condition', $._expression), field('consequence', $.block),
      optional(seq('else', field('alternative', $.block))),
    )),

    _logical_or_expression: $ => choice(
      $.or_expression,
      $._logical_and_expression,
    ),

    or_expression: $ => prec.left(1, seq($._expression, 'or', $._expression)),

    _logical_and_expression: $ => choice(
      $.and_expression,
      $._range_expression,
    ),

    and_expression: $ => prec.left(2, seq($._expression, 'and', $._expression)),

    _range_expression: $ => choice(
      $.range_expression,
      $._equality_expression,
    ),

    range_expression: $ => prec.left(3, seq($._expression, '..', $._expression)),

    _equality_expression: $ => choice(
      $.equal_expression,
      $.not_equal_expression,
      $._comparison_expression,
    ),

    equal_expression: $ => prec.left(4, seq($._expression, '==', $._expression)),
    not_equal_expression: $ => prec.left(4, seq($._expression, '!=', $._expression)),

    _comparison_expression: $ => choice(
      $.gt_expression,
      $.gte_expression,
      $.lt_expression,
      $.lte_expression,
      $.in_expression,
      $._term_expression,
    ),

    gt_expression: $ => prec.left(5, seq($._expression, '>', $._expression)),
    gte_expression: $ => prec.left(5, seq($._expression, '>=', $._expression)),
    lt_expression: $ => prec.left(5, seq($._expression, '<', $._expression)),
    lte_expression: $ => prec.left(5, seq($._expression, '<=', $._expression)),
    in_expression: $ => prec.left(5, seq($._expression, 'in', $._expression)),

    _term_expression: $ => choice(
      $.add_expression,
      $.subtract_expression,
      $._factor_expression,
    ),

    add_expression: $ => prec.left(6, seq($._expression, '+', $._expression)),
    subtract_expression: $ => prec.left(6, seq($._expression, '-', $._expression)),

    _factor_expression: $ => choice(
      $.multiply_expression,
      $.divide_expression,
      $.modulus_expression,
      $._unary_expression,
    ),

    multiply_expression: $ => prec.left(7, seq($._expression, '*', $._expression)),
    divide_expression: $ => prec.left(7, seq($._expression, '/', $._expression)),
    modulus_expression: $ => prec.left(7, seq($._expression, '%', $._expression)),

    _unary_expression: $ => choice(
      $.not_expression,
      $.negate_expression,
      $._call_expression,
    ),

    not_expression: $ => prec(8, seq('!', $._expression)),
    negate_expression: $ => prec(8, seq('-', $._expression)),

    _call_expression: $ => choice(
      $.call_expression,
      $.index_expression,
      $.attribute_expression,
      $._primary,
    ),

    call_expression: $ => prec.left(9, seq(
      $._expression, '(', optional($.argument_list), ')',
    )),

    index_expression: $ => prec.left(9, seq(
      $._expression, '[', $._expression, ']',
    )),

    attribute_expression: $ => prec.left(9, seq(
      $._expression, '.', $.identifier,
    )),

    argument_list: $ => seq(
      $._argument, repeat(seq(',', $._argument)), optional(','),
    ),

    _argument: $ => choice(
      $.keyword_argument,
      $._expression,
    ),

    keyword_argument: $ => seq($.identifier, '=', $._expression),

    // Primary expressions
    _primary: $ => choice(
      $.none,
      $.integer,
      $.float,
      $.string,
      $.boolean,
      $.identifier,
      $.list,
      $.parenthesized_expression,
      $.block,
      $.anonymous_function,
    ),

    none: $ => 'None',
    boolean: $ => choice('true', 'false'),

    integer: $ => /[0-9]+/,

    float: $ => /[0-9]+\.[0-9]+/,

    string: $ => seq(
      '"',
      repeat(choice(
        $.string_content,
        $.escape_sequence,
        $.string_interpolation,
      )),
      '"',
    ),

    string_content: $ => prec.right(repeat1(
      token.immediate(/[^"\\]+/)
    )),

    escape_sequence: $ => token.immediate(seq(
      '\\',
      /[ntr'"\\]/,
    )),

    string_interpolation: $ => seq(
      token.immediate('\\('),
      $._expression,
      ')',
    ),

    list: $ => seq('[', optional(seq($._expression, repeat(seq(',', $._expression)), optional(','))), ']'),

    parenthesized_expression: $ => seq('(', $._expression, ')'),

    anonymous_function: $ => seq(
      'fn', '(', optional($.parameter_list), ')', $.block,
    ),

    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,

    line_comment: $ => token(seq('//', /.*/)),

    block_comment: $ => seq('/*', optional($._block_comment_content), '*/'),
  }
});
