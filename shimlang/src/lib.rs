#![allow(dead_code)]

use std::collections::HashMap;

#[cfg(feature = "facet")]
use facet::Facet;

use std::ops::Range;
use std::ops::{Add, Sub};
use std::ops::{AddAssign, SubAssign};
use std::any::{Any, type_name, type_name_of_val};
use std::mem::size_of;
use std::rc::Rc;

use shm_tracy::*;
use shm_tracy::zone_scoped;

#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}


impl Add<Span> for Span {
    type Output = Span;

    fn add(self, other: Self) -> Self {
        Self {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

#[derive(Debug)]
pub struct Node<T> {
    pub data: T,
    pub span: Span,
}

impl<T> Node<T> {
}

// Now redefine your types using the wrapper
pub type ExprNode = Node<Expression>;
pub type StatementNode = Node<Statement>;
pub type Ident = Vec<u8>;

#[derive(Debug)]
pub enum Primary {
    None,
    Integer(i32),
    Float(f32),
    Identifier(Vec<u8>),
    Bool(bool),
    String(Vec<u8>),
    List(Vec<ExprNode>),
    Expression(Box<ExprNode>),
}

#[derive(Debug)]
pub enum UnaryOp {
    Not(Box<ExprNode>),
    Negate(Box<ExprNode>),
}

#[derive(Debug)]
pub enum BinaryOp {
    Add(Box<ExprNode>, Box<ExprNode>),
    Subtract(Box<ExprNode>, Box<ExprNode>),
    Multiply(Box<ExprNode>, Box<ExprNode>),
    Divide(Box<ExprNode>, Box<ExprNode>),
    Equal(Box<ExprNode>, Box<ExprNode>),
    NotEqual(Box<ExprNode>, Box<ExprNode>),
    GT(Box<ExprNode>, Box<ExprNode>),
    GTE(Box<ExprNode>, Box<ExprNode>),
    LT(Box<ExprNode>, Box<ExprNode>),
    LTE(Box<ExprNode>, Box<ExprNode>),
    Modulus(Box<ExprNode>, Box<ExprNode>),
    In(Box<ExprNode>, Box<ExprNode>),
    Range(Box<ExprNode>, Box<ExprNode>),
}

#[derive(Debug)]
pub enum BooleanOp {
    And(Box<ExprNode>, Box<ExprNode>),
    Or(Box<ExprNode>, Box<ExprNode>),
}

#[derive(Debug)]
pub struct Block {
    stmts: Vec<StatementNode>,
    last_expr: Option<Box<ExprNode>>,
}

#[derive(Debug)]
pub enum Expression {
    Primary(Primary),
    BooleanOp(BooleanOp),
    BinaryOp(BinaryOp),
    UnaryOp(UnaryOp),
    Stringify(Box<ExprNode>),
    Call(Box<ExprNode>, Vec<ExprNode>, Vec<(Ident, ExprNode)>),
    Index(Box<ExprNode>, Box<ExprNode>),
    Attribute(Box<ExprNode>, Vec<u8>),
    Block(Block),
    If(Box<ExprNode>, Block, Block),
    Fn(Fn),
}

#[derive(Debug)]
pub struct Fn {
    ident: Option<Vec<u8>>,
    pos_args_required: Vec<Vec<u8>>,
    pos_args_optional: Vec<(Vec<u8>, ExprNode)>,
    body: Block,
}

#[derive(Debug)]
pub struct Struct {
    ident: Vec<u8>,
    members_required: Vec<Vec<u8>>,
    members_optional: Vec<(Vec<u8>, ExprNode)>,
    methods: Vec<Fn>,
}

#[derive(Debug)]
pub enum Statement {
    Let(Vec<u8>, ExprNode),
    Assignment(Vec<u8>, ExprNode),
    AttributeAssignment(ExprNode, Vec<u8>, ExprNode),
    IndexAssignment(ExprNode, ExprNode, ExprNode),
    If(ExprNode, Block, Block),
    For(Vec<u8>, ExprNode, Block),
    While(ExprNode, Block),
    Break,
    Continue,
    Fn(Fn),
    Struct(Struct),
    Expression(ExprNode),
    Return(Option<ExprNode>),
}

#[derive(Debug)]
pub struct Ast {
    block: Block,
    script: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Dot,
    DotDot,
    Bang,
    Comma,
    Colon,
    LBracket,
    RBracket,
    Plus,
    Minus,
    Slash,
    Star,
    Let,
    Fn,
    If,
    Else,
    While,
    For,
    In,
    And,
    Or,
    Break,
    Continue,
    Struct,
    Return,
    Equal,
    DEqual,
    BangEqual,
    GT,
    GTE,
    LT,
    LTE,
    Percent,
    Semicolon,
    LSquare,
    RSquare,
    LAngle,
    RAngle,
    LCurly,
    RCurly,
    None,
    Integer(i32),
    Float(f32),
    Bool(bool),
    Identifier(Vec<u8>),
    String(Vec<u8>),
    StringInterpolationStart,
    StringInterpolationEnd,
    EOF,
}

impl Token {
    fn to_string(&self) -> String {
        match self {
            Token::LSquare => "[".to_string(),
            Token::LCurly => "{".to_string(),
            Token::LBracket => "(".to_string(),
            _ => format!("{self:?}")
        }
    }
}

pub struct TokenStream {
    idx: usize,
    tokens: Vec<Token>,
    // Technically we don't need to store this since we could recompute this
    // from the script when we need to show an error. This just keeps things
    // simple for now. The upper index is not inclusive, so (10,11) is 1 character.
    token_spans: Vec<Span>,
    script: Vec<u8>,
}

#[derive(Debug)]
struct LineInfo {
    start_idx: u32,
    end_idx: u32,
}

fn script_lines(script: &[u8]) -> Vec<LineInfo> {
    let mut line_info = Vec::new();
    let mut line_start_idx: u32 = 0;
    for (idx, c) in script.iter().enumerate() {
        if *c == b'\n' {
            line_info.push(LineInfo {
                start_idx: line_start_idx,
                end_idx: idx as u32,
            });
            line_start_idx = idx as u32 + 1;
        }
    }

    // If the last character was not a newline
    if line_start_idx != script.len() as u32 {
        line_info.push(LineInfo {
            start_idx: line_start_idx,
            end_idx: script.len() as u32 - 1,
        });
    }

    line_info
}

fn debug_u8s(data: &[u8]) -> &str {
    unsafe { std::str::from_utf8_unchecked(data) }
}

fn format_script_err(span: Span, script: &[u8], msg: &str) -> String {
    let script_lines = script_lines(script);
    let mut out = "".to_string();

    // Find which lines the span covers
    let mut first_line: usize = 0;
    let mut last_line: usize = 0;
    for (lineno, line_info) in script_lines.iter().enumerate() {
        if line_info.start_idx <= span.start && span.start <= line_info.end_idx {
            first_line = lineno;
        }
        // The `+ 1` accounts for span.end being one past the last character
        // (e.g. at a newline or EOF position just after the line ends)
        if line_info.start_idx <= span.end && span.end <= line_info.end_idx + 1 {
            last_line = lineno;
        }
    }

    // The `gutter_size` includes everything up to the first character of the line
    let gutter_size = script_lines.len().to_string().len() + 4;
    let is_multiline = first_line != last_line;

    // For single-line errors, show 2 lines of context before and after
    let context_before = if !is_multiline { 2 } else { 0 };
    let context_after = if !is_multiline { 2 } else { 0 };
    let display_start = first_line.saturating_sub(context_before);
    let display_end = (last_line + context_after).min(script_lines.len() - 1);

    for (lineno_0, line_info) in script_lines.iter().enumerate() {
        let lineno = lineno_0 + 1;

        // Only show lines in the display range
        if lineno_0 < display_start || lineno_0 > display_end {
            continue;
        }

        let line: String = unsafe {
            std::str::from_utf8_unchecked(
                &script[line_info.start_idx as usize..=line_info.end_idx as usize],
            )
            .to_string()
        };

        out.push_str(&format!(
            " {:lineno_size$} | {}",
            lineno,
            line,
            lineno_size = gutter_size - 4
        ));
        if !line.ends_with("\n") {
            out.push_str("\n");
        }

        if !is_multiline {
            if lineno_0 == first_line {
                let line_span_start = span.start - line_info.start_idx;
                let line_span_end = span.end - line_info.start_idx;

                out.push_str(&" ".repeat(gutter_size));
                out.push_str(&" ".repeat(line_span_start as usize));
                out.push_str(&"^".repeat((line_span_end - line_span_start) as usize));
                out.push('\n');
            }
        } else {
            if lineno_0 == first_line {
                // First line: show carets from span start to end of line content
                let line_span_start = span.start - line_info.start_idx;
                let line_len = line_info.end_idx - line_info.start_idx + 1;
                let caret_len = line_len - line_span_start;

                out.push_str(&" ".repeat(gutter_size));
                out.push_str(&" ".repeat(line_span_start as usize));
                out.push_str(&"^".repeat(caret_len as usize));
                out.push('\n');
            } else if lineno_0 == last_line {
                // Last line: show carets from start of line to span end
                let line_span_end = span.end - line_info.start_idx;

                out.push_str(&" ".repeat(gutter_size));
                out.push_str(&"^".repeat(line_span_end as usize));
                out.push('\n');
            } else {
                // Middle lines: show carets for full line
                let line_len = line_info.end_idx - line_info.start_idx + 1;
                out.push_str(&" ".repeat(gutter_size));
                out.push_str(&"^".repeat(line_len as usize));
                out.push('\n');
            }
        }
    }

    out.push_str(&format!("Error: {msg}"));

    out
}

impl TokenStream {
    /**
     * Return the next token (if there are tokens remaining) without advancing the stream
     */
    fn peek(&self) -> Result<&Token, String> {
        if self.is_empty() {
            Ok(&Token::EOF)
        } else {
            Ok(&self.tokens[self.idx])
        }
    }

    fn peek_span(&self) -> Result<Span, String> {
        if self.is_empty() {
            Ok(self.token_spans[self.token_spans.len()-1])
        } else {
            Ok(self.token_spans[self.idx])
        }
    }

    /// Return the span of the most recently consumed token
    fn previous_span(&self) -> Result<Span, String> {
        if self.idx > 0 {
            Ok(self.token_spans[self.idx - 1])
        } else {
            Err("No previous token".to_string())
        }
    }

    /**
     * Return the next token (if there are tokens remaining) and advance the stream
     */
    fn pop(&mut self) -> Result<Token, String> {
        if !self.is_empty() {
            let result = self.tokens[self.idx].clone();
            self.idx += 1;
            Ok(result)
        } else {
            Ok(Token::EOF)
        }
    }

    fn consume(&mut self, expected: Token) -> Result<(), String> {
        let value = self.pop()?;
        if value == expected {
            Ok(())
        } else {
            self.unadvance()?;
            Err(self.format_peek_err(&format!(
                "Expected token {:?} but found {:?}",
                expected, value
            )))
        }
    }

    fn advance(&mut self) -> Result<(), String> {
        if self.pop()? == Token::EOF {
            return Err(self.format_peek_err("End of token stream"));
        }
        Ok(())
    }

    fn unadvance(&mut self) -> Result<(), String> {
        if self.idx != 0 {
            self.idx -= 1;
            Ok(())
        } else {
            Err("Can't unadvance past beginning of token stream".to_string())
        }
    }

    fn is_empty(&self) -> bool {
        self.idx >= self.tokens.len()
    }

    fn format_peek_err(&self, msg: &str) -> String {
        let span = if !self.is_empty() {
            self.token_spans[self.idx]
        } else {
            Span {
                start: self.script.len() as u32 - 1,
                end: self.script.len() as u32,
            }
        };
        format_script_err(span, &self.script, msg)
    }

    pub fn spans(&self) -> Vec<Span> {
        self.token_spans.clone()
    }
}

pub fn parse_block(tokens: &mut TokenStream) -> Result<Block, String> {
    tokens.consume(Token::LCurly)?;
    let block = parse_block_inner(tokens)?;
    tokens.consume(Token::RCurly)?;

    Ok(block)
}

pub fn parse_primary(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let span = tokens.peek_span()?;
    let expr: Expression = match tokens.pop()? {
        Token::None => Expression::Primary(Primary::None),
        Token::Integer(i) => Expression::Primary(Primary::Integer(i)),
        Token::Float(f) => Expression::Primary(Primary::Float(f)),
        Token::String(s) => {
            let mut expr = Expression::Primary(Primary::String(s));

            while !tokens.is_empty() {
                match *tokens.peek()? {
                    Token::StringInterpolationStart => {
                        tokens.advance()?;
                        let interp_expr = parse_expression(tokens)?;
                        tokens.consume(Token::StringInterpolationEnd)?;

                        let token = tokens.pop()?;
                        match token {
                            Token::String(s) => {
                                expr = Expression::BinaryOp(
                                    BinaryOp::Add(
                                        Box::new(
                                            Node { data: expr, span }
                                        ),
                                        Box::new(
                                            Node {
                                                data: Expression::Stringify(Box::new(interp_expr)),
                                                span,
                                            }
                                        ),
                                    )
                                );
                                expr = Expression::BinaryOp(
                                    BinaryOp::Add(
                                        Box::new(Node { data: expr, span }),
                                        Box::new(Node { data: Expression::Primary(Primary::String(s)), span }),
                                    )
                                );
                            },
                            token => {
                                tokens.unadvance()?;
                                return Err(tokens.format_peek_err(&format!(
                                    "Unexpected `{:?}` after string interpolation",
                                    token
                                )));
                            }
                        }
                    },
                    _ => break,
                }
            }

            expr
        },
        Token::Bool(b) => Expression::Primary(Primary::Bool(b)),
        Token::Identifier(s) => Expression::Primary(Primary::Identifier(s)),
        Token::LCurly => {
            tokens.unadvance()?;
            let block = parse_block(tokens)?;
            Expression::Block(block)
        }
        Token::LBracket => {
            let expr = parse_expression(tokens)?;
            tokens.consume(Token::RBracket)?;
            return Ok(expr);
        }
        Token::LSquare => {
            let items = parse_arguments(tokens, Token::RSquare)?;
            // TODO: fix span here
            Expression::Primary(Primary::List(items))
        }
        Token::Fn => {
            tokens.unadvance()?;
            let f = parse_function(tokens)?;
            Expression::Fn(f)
        }
        token => {
            tokens.unadvance()?;
            return Err(tokens.format_peek_err(&format!(
                "Unexpected `{:?}` in parse_primary",
                // TODO: should display the exact character like `[` rather than name line `SemiColon`
                token
            )));
        }
    };
    Ok(Node {
        data: expr,
        span: span,
    })
}

pub fn parse_arguments(
    tokens: &mut TokenStream,
    closing_token: Token,
) -> Result<Vec<ExprNode>, String> {
    let mut args = Vec::new();
    if !tokens.is_empty() && *tokens.peek()? == closing_token {
        tokens.advance()?;
        return Ok(args);
    }
    while !tokens.is_empty() {
        let expr = parse_expression(tokens)?;
        args.push(expr);

        match tokens.peek()? {
            token if *token == closing_token => {
                tokens.advance()?;
                break;
            }
            Token::Comma => {
                tokens.advance()?;
                if !tokens.is_empty() && *tokens.peek()? == closing_token {
                    // Exit when there's a trailing comma
                    tokens.advance()?;
                    break;
                }
                continue;
            }
            token => {
                return Err(tokens.format_peek_err(&format!(
                    "Expected comma or closing bracket, found {:?}",
                    token
                )));
            }
        }
    }
    Ok(args)
}

pub fn parse_fn_arguments(
    tokens: &mut TokenStream,
    closing_token: Token,
) -> Result<(Vec<ExprNode>, Vec<(Ident, ExprNode)>), String> {
    let mut args = Vec::new();
    let mut kwargs = Vec::new();
    if !tokens.is_empty() && *tokens.peek()? == closing_token {
        tokens.advance()?;
        return Ok((args, kwargs));
    }
    while !tokens.is_empty() {
        let expr = parse_expression(tokens)?;

        if *tokens.peek()? == Token::Equal {
            if let Expression::Primary(Primary::Identifier(ident)) = expr.data {
                tokens.advance()?;
                kwargs.push((ident, parse_expression(tokens)?));
            } else {
                return Err(
                    tokens.format_peek_err(&format!("Expected ident before `=` in fn args"))
                );
            }
        } else {
            if kwargs.len() > 0 {
                return Err(tokens.format_peek_err(&format!(
                    "Positional arguments can't appear after keyword arguments"
                )));
            }
            args.push(expr);
        }

        match tokens.peek()? {
            token if *token == closing_token => {
                tokens.advance()?;
                break;
            }
            Token::Comma => {
                tokens.advance()?;
                if !tokens.is_empty() && *tokens.peek()? == closing_token {
                    // Exit when there's a trailing comma
                    tokens.advance()?;
                    break;
                }
                continue;
            }
            token => {
                return Err(tokens.format_peek_err(&format!(
                    "Expected comma or closing bracket, found {:?}",
                    token
                )));
            }
        }
    }
    Ok((args, kwargs))
}

pub fn parse_call(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_primary(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match *tokens.peek()? {
            Token::LBracket => {
                tokens.advance()?;
                let args = parse_fn_arguments(tokens, Token::RBracket)?;
                let end_span = tokens.previous_span()?;
                expr = Node {
                    span: expr.span + end_span,
                    data: Expression::Call(Box::new(expr), args.0, args.1),
                };
            }
            Token::LSquare => {
                tokens.advance()?;
                let index_expr = parse_expression(tokens)?;
                tokens.consume(Token::RSquare)?;
                expr = Node {
                    data: Expression::Index(Box::new(expr), Box::new(index_expr)),
                    span: span,
                };
            }
            Token::Dot => {
                tokens.advance()?;
                let ident_span = tokens.peek_span()?;
                let ident = match tokens.pop()? {
                    Token::Identifier(ident) => ident.clone(),
                    token => {
                        return Err(tokens.format_peek_err(&format!(
                            "Expected ident after dot, found {:?}",
                            token
                        )));
                    }
                };
                expr = Node {
                    data: Expression::Attribute(Box::new(expr), ident),
                    span: span + ident_span,
                };
            }
            _ => return Ok(expr),
        }
    }

    Ok(expr)
}

pub fn parse_logical_or(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_logical_and(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match tokens.peek()? {
            Token::Or => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BooleanOp(BooleanOp::Or(
                        Box::new(expr),
                        Box::new(parse_logical_and(tokens)?),
                    )),
                    span: span,
                };
            }
            _ => return Ok(expr),
        }
    }
    Ok(expr)
}

pub fn parse_logical_and(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_range(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match tokens.peek()? {
            Token::And => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BooleanOp(BooleanOp::And(
                        Box::new(expr),
                        Box::new(parse_range(tokens)?),
                    )),
                    span: span,
                };
            }
            _ => return Ok(expr),
        }
    }
    Ok(expr)
}

pub fn parse_range(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_equality(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match tokens.peek()? {
            Token::DotDot => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::Range(
                        Box::new(expr),
                        Box::new(parse_equality(tokens)?),
                    )),
                    span: span,
                };
            }
            _ => return Ok(expr),
        }
    }
    Ok(expr)
}

pub fn parse_comparison(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_term(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match tokens.peek()? {
            Token::GT => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::GT(
                        Box::new(expr),
                        Box::new(parse_term(tokens)?),
                    )),
                    span: span,
                };
            }
            Token::GTE => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::GTE(
                        Box::new(expr),
                        Box::new(parse_term(tokens)?),
                    )),
                    span: span,
                };
            }
            Token::LT => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::LT(
                        Box::new(expr),
                        Box::new(parse_term(tokens)?),
                    )),
                    span: span,
                };
            }
            Token::LTE => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::LTE(
                        Box::new(expr),
                        Box::new(parse_term(tokens)?),
                    )),
                    span: span,
                };
            }
            Token::In => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::In(
                        Box::new(expr),
                        Box::new(parse_term(tokens)?),
                    )),
                    span: span,
                };
            }
            _ => return Ok(expr),
        }
    }
    Ok(expr)
}

pub fn parse_factor(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_unary(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match tokens.peek()? {
            Token::Star => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::Multiply(
                        Box::new(expr),
                        Box::new(parse_unary(tokens)?),
                    )),
                    span: span,
                };
            }
            Token::Slash => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::Divide(
                        Box::new(expr),
                        Box::new(parse_unary(tokens)?),
                    )),
                    span: span,
                };
            }
            Token::Percent => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::Modulus(
                        Box::new(expr),
                        Box::new(parse_unary(tokens)?),
                    )),
                    span: span,
                };
            }
            _ => return Ok(expr),
        }
    }
    Ok(expr)
}

pub fn parse_equality(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_comparison(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match tokens.peek()? {
            Token::DEqual => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::Equal(
                        Box::new(expr),
                        Box::new(parse_comparison(tokens)?),
                    )),
                    span: span,
                };
            }
            Token::BangEqual => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::NotEqual(
                        Box::new(expr),
                        Box::new(parse_comparison(tokens)?),
                    )),
                    span: span,
                };
            }
            _ => return Ok(expr),
        }
    }
    Ok(expr)
}

pub fn parse_term(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_factor(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match tokens.peek()? {
            Token::Plus => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::Add(
                        Box::new(expr),
                        Box::new(parse_factor(tokens)?),
                    )),
                    span: span,
                };
            }
            Token::Minus => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::Subtract(
                        Box::new(expr),
                        Box::new(parse_factor(tokens)?),
                    )),
                    span: span,
                };
            }
            _ => return Ok(expr),
        }
    }
    Ok(expr)
}

pub fn parse_unary(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let span = tokens.peek_span()?;
    match tokens.peek()? {
        Token::Bang => {
            tokens.advance()?;
            let expr = parse_unary(tokens)?;
            Ok(Node {
                data: Expression::UnaryOp(UnaryOp::Not(Box::new(expr))),
                span: span,
            })
        }
        Token::Minus => {
            tokens.advance()?;
            let expr = parse_unary(tokens)?;
            Ok(Node {
                data: Expression::UnaryOp(UnaryOp::Negate(Box::new(expr))),
                span: span,
            })
        }
        _ => parse_call(tokens),
    }
}

pub struct Conditional {
    conditional: ExprNode,
    if_body: Block,
    else_body: Block,
}

impl Conditional {
    fn new(conditional: ExprNode, if_body: Block, else_body: Block) -> Self {
        Conditional {
            conditional,
            if_body,
            else_body,
        }
    }
}

pub fn parse_conditional(tokens: &mut TokenStream) -> Result<(Conditional, Span), String> {
    let start_span = tokens.peek_span()?;
    tokens.consume(Token::If)?;
    let conditional = parse_expression(tokens)?;

    let if_body = parse_block(tokens)?;

    let else_body = if !tokens.is_empty() && *tokens.peek()? == Token::Else {
        tokens.advance()?;
        if *tokens.peek()? == Token::If {
            let (elseif, elseif_span) = parse_conditional(tokens)?;
            Block {
                stmts: Vec::new(),
                last_expr: Some(
                    Box::new(ExprNode {
                        data: Expression::If(Box::new(elseif.conditional), elseif.if_body, elseif.else_body),
                        span: elseif_span,
                    })
                )
            }
        } else {
            parse_block(tokens)?
        }
    } else {
        Block {
            stmts: Vec::new(),
            last_expr: None,
        }
    };

    // Get the end span (previous token, which is the closing curly)
    let end_span = tokens.previous_span()?;

    Ok((Conditional::new(conditional, if_body, else_body), start_span + end_span))
}

pub fn parse_expression(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    match *tokens.peek()? {
        Token::If => {
            let (cond, cond_span) = parse_conditional(tokens)?;
            Ok(ExprNode {
                data: Expression::If(Box::new(cond.conditional), cond.if_body, cond.else_body),
                span: cond_span,
            })
        },
        _ => parse_logical_or(tokens),
    }
}

pub fn parse_ast(tokens: &mut TokenStream) -> Result<Ast, String> {
    let block = parse_block_inner(tokens)?;
    Ok(Ast {
        block: block,
        script: tokens.script.clone(),
    })
}

pub fn parse_function(tokens: &mut TokenStream) -> Result<Fn, String> {
    tokens.consume(Token::Fn)?;
    let ident = match tokens.peek()? {
        Token::Identifier(ident) => {
            let ident = ident.clone();
            tokens.advance()?;
            Some(ident)
        },
        Token::LBracket => None,
        token => {
            return Err(
                tokens.format_peek_err(&format!("Expected ident after fn, found {:?}", token))
            );
        }
    };
    tokens.consume(Token::LBracket)?;

    let mut params = Vec::new();
    let mut optional_params = Vec::new();
    while !tokens.is_empty() {
        let ident = match tokens.peek()? {
            Token::Identifier(ident) => ident.clone(),
            Token::RBracket => break,
            token => {
                return Err(tokens
                    .format_peek_err(&format!("Expected ident for fn params, found {:?}", token)));
            }
        };
        tokens.advance()?;

        let expr = if *tokens.peek()? == Token::Equal {
            tokens.advance()?;
            Some(parse_expression(tokens)?)
        } else {
            if optional_params.len() > 0 {
                return Err(tokens.format_peek_err(&format!("No required arguments after optional")));
            }
            None
        };

        if let Some(expr) = expr {
            optional_params.push((ident, expr));
        } else {
            params.push(ident);
        }

        if *tokens.peek()? != Token::Comma {
            break;
        }
        tokens.advance()?;
    }
    tokens.consume(Token::RBracket)?;

    let body = parse_block(tokens)?;

    Ok(Fn {
        ident: ident,
        pos_args_required: params,
        pos_args_optional: optional_params,
        body: body,
    })
}

pub fn parse_block_inner(tokens: &mut TokenStream) -> Result<Block, String> {
    let mut stmts = Vec::new();
    let mut last_expr: Option<Box<ExprNode>> = None;

    while !tokens.is_empty() && *tokens.peek()? != Token::RCurly {
        let start_span = tokens.peek_span()?;
        let stmt = if *tokens.peek()? == Token::Let {
            tokens.advance()?;
            if tokens.is_empty() {
                return Err("No token found after let".to_string());
            }
            let ident = match tokens.pop()? {
                Token::Identifier(ident) => ident.clone(),
                token => return Err(format!("Expected ident after let, found {:?}", token)),
            };

            match tokens.pop()? {
                Token::Equal => (),
                token => return Err(format!("Expected = after `let ident`, found {:?}", token)),
            }

            let expr = parse_expression(tokens)?;
            let end_span = tokens.peek_span()?;
            match tokens.pop()? {
                Token::Semicolon => (),
                token => {
                    tokens.unadvance()?;
                    return Err(tokens.format_peek_err(&format!(
                        "Expected semicolon after `let <ident> = <expr>`, found {:?}",
                        token
                    )));
                }
            }

            StatementNode {
                data: Statement::Let(ident, expr),
                span: start_span + end_span,
            }
        } else if *tokens.peek()? == Token::Fn {
            let fn_result = parse_function(tokens)?;
            let end_span = tokens.previous_span()?;
            StatementNode {
                data: Statement::Fn(fn_result),
                span: start_span + end_span,
            }
        } else if *tokens.peek()? == Token::If {
            let (cond, cond_span) = parse_conditional(tokens)?;

            // Do we treat this as an expression or statement?
            if *tokens.peek()? == Token::RCurly {
                // If the next token is an RCurly, it means that this is the closing curly of the block
                let expr = Expression::If(Box::new(cond.conditional), cond.if_body, cond.else_body);
                last_expr = Some(Box::new(ExprNode {
                    data: expr,
                    span: cond_span,
                }));
                break;
            } else {
                StatementNode {
                    data: Statement::If(cond.conditional, cond.if_body, cond.else_body),
                    span: cond_span,
                }
            }
        } else if *tokens.peek()? == Token::For {
            tokens.advance()?;
            let ident = match tokens.pop()? {
                Token::Identifier(ident) => ident,
                token => {
                    tokens.unadvance()?;
                    return Err(tokens
                        .format_peek_err(&format!("Expected ident after for, found {:?}", token)));
                }
            };
            tokens.consume(Token::In)?;
            let expr = parse_expression(tokens)?;
            let body = parse_block(tokens)?;

            let end_span = tokens.previous_span()?;
            StatementNode {
                data: Statement::For(ident, expr, body),
                span: start_span + end_span,
            }
        } else if *tokens.peek()? == Token::While {
            tokens.advance()?;
            let conditional = parse_expression(tokens)?;
            let loop_body = parse_block(tokens)?;

            let end_span = tokens.previous_span()?;
            StatementNode {
                data: Statement::While(conditional, loop_body),
                span: start_span + end_span,
            }
        } else if *tokens.peek()? == Token::Break {
            tokens.advance()?;
            let end_span = tokens.peek_span()?;
            tokens.consume(Token::Semicolon)?;
            StatementNode {
                data: Statement::Break,
                span: start_span + end_span,
            }
        } else if *tokens.peek()? == Token::Continue {
            tokens.advance()?;
            let end_span = tokens.peek_span()?;
            tokens.consume(Token::Semicolon)?;
            StatementNode {
                data: Statement::Continue,
                span: start_span + end_span,
            }
        } else if *tokens.peek()? == Token::Struct {
            tokens.advance()?;
            let ident = match tokens.pop()? {
                Token::Identifier(ident) => ident.clone(),
                token => {
                    tokens.unadvance()?;
                    return Err(tokens.format_peek_err(&format!(
                        "Expected ident after struct, found {:?}",
                        token
                    )));
                }
            };
            tokens.consume(Token::LCurly)?;

            let mut members = Vec::new();
            let mut optional_members = Vec::new();
            while !tokens.is_empty() {
                match tokens.pop()? {
                    Token::Identifier(ident) => {
                        if *tokens.peek()? == Token::Equal {
                            tokens.advance()?;
                            let expr = parse_expression(tokens)?;
                            optional_members.push((ident.clone(), expr));
                        } else {
                            if !optional_members.is_empty() {
                                return Err(tokens.format_peek_err(&format!(
                                    "Required members not allowed after optional members"
                                )));
                            }
                            members.push(ident.clone());
                        }
                        if *tokens.peek()? != Token::Comma {
                            break;
                        }
                        // Consume the comma
                        tokens.advance()?;
                    }
                    Token::RCurly | Token::Fn => {
                        tokens.unadvance()?;
                        break;
                    }
                    token => {
                        tokens.unadvance()?;
                        return Err(tokens.format_peek_err(&format!(
                            "Expected member list after struct, found {:?}",
                            token
                        )));
                    }
                };
            }

            let mut methods = Vec::new();
            while !tokens.is_empty() {
                match tokens.peek()? {
                    Token::Fn => {
                        methods.push(parse_function(tokens)?);
                    }
                    Token::RCurly => {
                        break;
                    }
                    token => {
                        return Err(tokens.format_peek_err(&format!(
                            "Unexpected token during method parsing {:?}",
                            token
                        )));
                    }
                }
            }
            let end_span = tokens.peek_span()?;
            tokens.consume(Token::RCurly)?;

            StatementNode {
                data: Statement::Struct(Struct {
                    ident,
                    members_required: members,
                    members_optional: optional_members,
                    methods,
                }),
                span: start_span + end_span,
            }
        } else if *tokens.peek()? == Token::Return {
            tokens.advance()?;
            if *tokens.peek()? == Token::Semicolon {
                let end_span = tokens.peek_span()?;
                tokens.advance()?;
                StatementNode {
                    data: Statement::Return(None),
                    span: start_span + end_span,
                }
            } else {
                let expr = parse_expression(tokens)?;
                let end_span = tokens.peek_span()?;
                match tokens.pop()? {
                    Token::Semicolon => (),
                    token => {
                        tokens.unadvance()?;
                        return Err(format_script_err(
                            start_span + expr.span,
                            &tokens.script,
                            &format!(
                                "Expected semicolon after `return <expr>`, found {:?}",
                                token
                            ),
                        ));
                    }
                }
                StatementNode {
                    data: Statement::Return(Some(expr)),
                    span: start_span + end_span,
                }
            }
        } else {
            let expr = parse_expression(tokens)?;
            if tokens.is_empty() {
                last_expr = Some(Box::new(expr));
                break;
            }

            match tokens.peek()? {
                Token::RCurly => {
                    last_expr = Some(Box::new(expr));
                    break;
                }
                Token::Semicolon => {
                    let end_span = tokens.peek_span()?;
                    tokens.pop()?;
                    StatementNode {
                        data: Statement::Expression(expr),
                        span: start_span + end_span,
                    }
                }
                Token::Equal => {
                    tokens.pop()?;
                    match expr.data {
                        Expression::Primary(Primary::Identifier(ident)) => {
                            let expr_to_assign = parse_expression(tokens)?;
                            let end_span = tokens.peek_span()?;
                            tokens.consume(Token::Semicolon)?;
                            StatementNode {
                                data: Statement::Assignment(ident.clone(), expr_to_assign),
                                span: start_span + end_span,
                            }
                        }
                        Expression::Attribute(expr, ident) => {
                            let expr_to_assign = parse_expression(tokens)?;
                            let end_span = tokens.peek_span()?;
                            tokens.consume(Token::Semicolon)?;
                            StatementNode {
                                data: Statement::AttributeAssignment(*expr, ident.clone(), expr_to_assign),
                                span: start_span + end_span,
                            }
                        }
                        Expression::Index(expr, index_expr) => {
                            let expr_to_assign = parse_expression(tokens)?;
                            let end_span = tokens.peek_span()?;
                            tokens.consume(Token::Semicolon)?;
                            StatementNode {
                                data: Statement::IndexAssignment(*expr, *index_expr, expr_to_assign),
                                span: start_span + end_span,
                            }
                        }
                        expr_data => {
                            return Err(format_script_err(
                                expr.span,
                                &tokens.script,
                                &format!("Can't assign to {:?}", expr_data),
                            ));
                        }
                    }
                }
                token => {
                    return Err(tokens.format_peek_err(&format!(
                        "Expected semicolon after expression statement, found {:?}",
                        token
                    )));
                }
            }
        };

        stmts.push(stmt);
    }

    Ok(Block { stmts, last_expr })
}

pub fn printable_byte(b: u8) -> String {
    match char::from_u32(b as u32) {
        Some(c) if !c.is_control() => c.to_string(),
        _ => format!("\\x{:02X}", b),
    }
}

pub fn lex_identifier(text: &mut &[u8]) -> Result<Vec<u8>, String> {
    for (idx, c) in text.iter().enumerate() {
        match c {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' => continue,
            _ => {
                let ident = text[0..idx].to_vec();
                *text = &text[(idx - 1)..];
                return Ok(ident);
            }
        }
    }
    // End of string - consume all of text
    Ok(text.to_vec())
}

pub enum StringLexResult {
    Literal(Vec<u8>),
    Interpolation(Vec<u8>),
}

pub fn lex_string(text: &mut &[u8]) -> Result<StringLexResult, String> {
    let enclosing_char = b'"';
    *text = &text[1..];
    let mut out: Vec<u8> = Vec::new();
    let mut escape_next = false;
    for (idx, c) in text.iter().enumerate() {
        if escape_next {
            match c {
                b'n' => out.push(b'\n'),
                b't' => out.push(b'\t'),
                b'\'' => out.push(b'\''),
                b'\\' => out.push(b'\\'),
                b'"' => out.push(b'"'),
                b'(' => {
                    *text = &text[idx..];
                    return Ok(StringLexResult::Interpolation(out));
                },
                b => return Err(format!("Could not escape {:?}", printable_byte(*b))),
            }
            escape_next = false;
            continue;
        }
        if *c == enclosing_char {
            *text = &text[idx..];
            return Ok(StringLexResult::Literal(out));
        }
        match c {
            b'\\' => {
                escape_next = true;
                continue;
            }
            _ => {
                out.push(*c);
            }
        }
    }
    Err(format!(
        "No closing quote {:?} found",
        printable_byte(enclosing_char)
    ))
}

pub fn lex_number(text: &mut &[u8]) -> Result<Token, String> {
    let mut found_decimal = false;
    for (idx, c) in text.iter().enumerate() {
        match c {
            b'0'..=b'9' => continue,
            b'.' => {
                // Check if this is a range operator (..)
                if idx + 1 < text.len() && text[idx + 1] == b'.' {
                    // This is a range operator, stop here
                    let token = if found_decimal {
                        Token::Float(unsafe {
                            let slice = &text[..idx];
                            std::str::from_utf8_unchecked(slice).parse().map_err(|e| {
                                let string_slice = match std::str::from_utf8(slice) {
                                    Ok(s) => s,
                                    Err(e) => return format!("Not utf-8 {:?}", e),
                                };
                                format!("Could not tokenize number '{}' {:?}", string_slice, e)
                            })?
                        })
                    } else {
                        Token::Integer(unsafe {
                            let slice = &text[..idx];
                            std::str::from_utf8_unchecked(slice).parse().map_err(|e| {
                                let string_slice = match std::str::from_utf8(slice) {
                                    Ok(s) => s,
                                    Err(e) => return format!("Not utf-8 {:?}", e),
                                };
                                format!("Could not tokenize number '{}' {:?}", string_slice, e)
                            })?
                        })
                    };
                    *text = &text[(idx - 1)..];
                    return Ok(token);
                }
                if found_decimal {
                    return Err(format!("Found multiple decimals in number"));
                }
                found_decimal = true;
            }
            _ => {
                let token = if found_decimal {
                    Token::Float(unsafe {
                        let slice = &text[..idx];
                        std::str::from_utf8_unchecked(slice).parse().map_err(|e| {
                            let string_slice = match std::str::from_utf8(slice) {
                                Ok(s) => s,
                                Err(e) => return format!("Not utf-8 {:?}", e),
                            };
                            format!("Could not tokenize number '{}' {:?}", string_slice, e)
                        })?
                    })
                } else {
                    Token::Integer(unsafe {
                        let slice = &text[..idx];
                        std::str::from_utf8_unchecked(slice).parse().map_err(|e| {
                            let string_slice = match std::str::from_utf8(slice) {
                                Ok(s) => s,
                                Err(e) => return format!("Not utf-8 {:?}", e),
                            };
                            format!("Could not tokenize number '{}' {:?}", string_slice, e)
                        })?
                    })
                };
                *text = &text[(idx - 1)..];
                return Ok(token);
            }
        }
    }
    // End of string - consume all of text
    let token = Token::Integer(unsafe {
        std::str::from_utf8_unchecked(text)
            .parse()
            .map_err(|e| format!("{:?}", e))?
    });
    Ok(token)
}

pub fn lex_multiline_comment_end_idx(text: &[u8]) -> Result<usize, String> {
    if text.len() < 4 {
        return Err(format!("Text not long enough to finish multiline comment"));
    }

    if text[..2] != *b"/*" {
        return Err(format!("Multiline comment does not start with `/*`"));
    }

    let mut depth = 1;
    let mut idx = 2;

    while text.len() - idx > (depth*2) {
        if text[idx] == b'/' && text[idx+1] == b'*' {
            depth += 1;
            idx += 2;
            continue
        }

        if text[idx] == b'*' && text[idx+1] == b'/' {
            depth -= 1;
            idx += 2;

            if depth == 0 {
                return Ok(idx);
            }
            continue
        }
        idx += 1;
    }
    Err(format!("Not enough text remaining to close multiline comment"))
}

pub fn lex(text: &[u8]) -> Result<TokenStream, String> {
    let starting_len = text.len();
    let starting_text = text;
    let original_text = text;
    let mut text = text;
    let mut spans = Vec::new();
    let mut tokens = Vec::new();

    let mut braces: Vec<Token> = Vec::new();

    while !text.is_empty() {
        let c = text[0];
        let token_start_len = text.len();
        match c {
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let ident = lex_identifier(&mut text)?;
                if ident == b"let" {
                    tokens.push(Token::Let);
                } else if ident == b"fn" {
                    tokens.push(Token::Fn);
                } else if ident == b"if" {
                    tokens.push(Token::If);
                } else if ident == b"else" {
                    tokens.push(Token::Else);
                } else if ident == b"in" {
                    tokens.push(Token::In);
                } else if ident == b"for" {
                    tokens.push(Token::For);
                } else if ident == b"while" {
                    tokens.push(Token::While);
                } else if ident == b"break" {
                    tokens.push(Token::Break);
                } else if ident == b"continue" {
                    tokens.push(Token::Continue);
                } else if ident == b"in" {
                    tokens.push(Token::In);
                } else if ident == b"struct" {
                    tokens.push(Token::Struct);
                } else if ident == b"return" {
                    tokens.push(Token::Return);
                } else if ident == b"and" {
                    tokens.push(Token::And);
                } else if ident == b"or" {
                    tokens.push(Token::Or);
                } else if ident == b"true" {
                    tokens.push(Token::Bool(true));
                } else if ident == b"false" {
                    tokens.push(Token::Bool(false));
                } else if ident == b"None" {
                    tokens.push(Token::None);
                } else {
                    tokens.push(Token::Identifier(ident))
                }
            }
            b'0'..=b'9' => tokens.push(lex_number(&mut text)?),
            b'"' => {
                match lex_string(&mut text)? {
                    StringLexResult::Literal(s) => tokens.push(Token::String(s)),
                    StringLexResult::Interpolation(s) => {
                        tokens.push(Token::String(s));
                        tokens.push(Token::StringInterpolationStart);
                        braces.push(Token::StringInterpolationStart);
                    },
                }
            },
            b'{' => {
                tokens.push(Token::LCurly);
                braces.push(Token::LCurly);
            },
            b'[' => {
                tokens.push(Token::LSquare);
                braces.push(Token::LSquare);
            },
            b'(' => {
                tokens.push(Token::LBracket);
                braces.push(Token::LBracket);
            },
            b'}' => {
                match braces.pop() {
                    Some(Token::LCurly) => (),
                    Some(b) => {
                        return Err(
                            format_script_err(
                                Span {
                                    start: (original_text.len() - text.len()) as u32,
                                    end: (original_text.len() - text.len() + 1) as u32,
                                },
                                original_text,
                                &format!("Brace {} does not match {}", b.to_string(), c as char),
                            )
                        );
                    },
                    None => {
                        return Err(
                            format_script_err(
                                Span {
                                    start: (original_text.len() - text.len()) as u32,
                                    end: (original_text.len() - text.len() + 1) as u32,
                                },
                                original_text,
                                &format!("No braces remaining on stack!"),
                            )
                        );
                    }
                }
                tokens.push(Token::RCurly)
            },
            b']' => {
                match braces.pop() {
                    Some(Token::LSquare) => (),
                    Some(b) => {
                        return Err(
                            format_script_err(
                                Span {
                                    start: (original_text.len() - text.len()) as u32,
                                    end: (original_text.len() - text.len() + 1) as u32,
                                },
                                original_text,
                                &format!("Brace {} does not match {}", b.to_string(), c as char),
                            )
                        );
                    },
                    None => {
                        return Err(
                            format_script_err(
                                Span {
                                    start: (original_text.len() - text.len()) as u32,
                                    end: (original_text.len() - text.len() + 1) as u32,
                                },
                                original_text,
                                &format!("No braces remaining on stack!"),
                            )
                        );
                    }
                }
                tokens.push(Token::RSquare);
            },
            b')' => {
                match braces.pop() {
                    Some(Token::LBracket) => tokens.push(Token::RBracket),
                    Some(Token::StringInterpolationStart) => {
                        tokens.push(Token::StringInterpolationEnd);
                        match lex_string(&mut text)? {
                            StringLexResult::Literal(s) => tokens.push(Token::String(s)),
                            StringLexResult::Interpolation(s) => {
                                tokens.push(Token::String(s));
                                tokens.push(Token::StringInterpolationStart);
                                braces.push(Token::StringInterpolationStart);
                            },
                        }
                    },
                    Some(b) => {
                        return Err(
                            format_script_err(
                                Span {
                                    start: (original_text.len() - text.len()) as u32,
                                    end: (original_text.len() - text.len() + 1) as u32,
                                },
                                original_text,
                                &format!("Brace {} does not match {}", b.to_string(), c as char),
                            )
                        );
                    },
                    None => {
                        return Err(
                            format_script_err(
                                Span {
                                    start: (original_text.len() - text.len()) as u32,
                                    end: (original_text.len() - text.len() + 1) as u32,
                                },
                                original_text,
                                &format!("No braces remaining on stack!"),
                            )
                        );
                    }
                }
            },
            b',' => tokens.push(Token::Comma),
            b'+' => tokens.push(Token::Plus),
            b'*' => tokens.push(Token::Star),
            b'%' => tokens.push(Token::Percent),
            b'/' => match text[1] {
                b'/' => {
                    loop {
                        text = &text[1..];
                        if text.is_empty() {
                            break;
                        }
                        match text[0] {
                            b'\n' => break,
                            _ => (),
                        }
                    }
                    // NOTE: no token to push since this is a comment
                },
                b'*' => {
                    let idx = lex_multiline_comment_end_idx(&text)?;
                    text = &text[(idx-1)..];
                }
                _ => tokens.push(Token::Slash),
            },
            b'-' => tokens.push(Token::Minus),
            b'=' => match text[1] {
                b'=' => {
                    text = &text[1..];
                    tokens.push(Token::DEqual);
                }
                _ => tokens.push(Token::Equal),
            },
            b'>' => match text[1] {
                b'=' => {
                    text = &text[1..];
                    tokens.push(Token::GTE);
                }
                _ => tokens.push(Token::GT),
            },
            b'<' => match text[1] {
                b'=' => {
                    text = &text[1..];
                    tokens.push(Token::LTE);
                }
                _ => tokens.push(Token::LT),
            },
            b';' => tokens.push(Token::Semicolon),
            b'\n' => (),
            b'\r' => (),
            b':' => tokens.push(Token::Colon),
            b'!' => {
                if text.len() > 1 && text[1] == b'=' {
                    text = &text[1..];
                    tokens.push(Token::BangEqual);
                } else {
                    tokens.push(Token::Bang);
                }
            },
            b'.' => {
                if text.len() > 1 && text[1] == b'.' {
                    text = &text[1..];
                    tokens.push(Token::DotDot);
                } else {
                    tokens.push(Token::Dot);
                }
            },
            b' ' => (),
            _ => return Err(format!("Unknown character '{}'", printable_byte(c))),
        }
        text = &text[1..];
        let token_end_len = text.len();
        while tokens.len() > spans.len() {
            spans.push(Span {
                start: (starting_len - token_start_len) as u32,
                end: (starting_len - token_end_len) as u32,
            });
        }
    }
    assert_eq!(
        tokens.len(),
        spans.len(),
    );
    Ok(TokenStream {
        idx: 0,
        tokens: tokens,
        token_spans: spans,
        script: starting_text.to_vec(),
    })
}

pub fn ast_from_text(text: &[u8]) -> Result<Ast, String> {
    let mut tokens = lex(text)?;
    parse_ast(&mut tokens)
}

#[derive(Debug)]
pub struct Config {
    // There are max 2^24 addressable values, each 8 bytes large
    // This value can be up to 2^27-1.
    memory_space_bytes: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            memory_space_bytes: MAX_U24 * 8,
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Hash, Eq, PartialOrd, Ord, Copy, Clone, Debug, PartialEq)]
#[repr(packed)]
pub struct u24([u8; 3]);
const MAX_U24: u32 = 0xFFFFFF;

impl From<Word> for usize {
    fn from(val: Word) -> Self {
        val.0.into()
    }
}

impl From<usize> for u24 {
    fn from(val: usize) -> Self {
        (val as u32).into()
    }
}

impl From<i32> for u24 {
    fn from(val: i32) -> Self {
        (val as u32).into()
    }
}

impl From<u32> for u24 {
    fn from(val: u32) -> Self {
        let b = val.to_be_bytes();
        u24([b[1], b[2], b[3]])
    }
}

impl From<u24> for u32 {
    fn from(val: u24) -> u32 {
        u32::from_be_bytes([0, val.0[0], val.0[1], val.0[2]])
    }
}

impl From<u24> for usize {
    fn from(val: u24) -> usize {
        u32::from(val) as usize
    }
}

impl From<u24> for u64 {
    fn from(val: u24) -> u64 {
        u32::from(val) as u64
    }
}

/**
 * The interpreter stores memory in 8-byte words. Each `Word` is
 * an index into the interpreter memory.
 */
#[cfg_attr(feature = "facet", derive(Facet))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Word(pub u24);

impl Add<u8> for Word {
    type Output = Word;

    fn add(self, rhs: u8) -> Word {
        self + rhs as u32
    }
}

impl Add<i32> for Word {
    type Output = Word;

    fn add(self, rhs: i32) -> Word {
        let val = (u32::from(self.0) as i32 + rhs) as u32;
        Word(val.into())
    }
}

impl Add<u32> for Word {
    type Output = Word;

    fn add(self, rhs: u32) -> Word {
        Word((u32::from(self.0) + rhs).into())
    }
}

impl Sub<u32> for Word {
    type Output = Word;

    fn sub(self, rhs: u32) -> Word {
        Word((u32::from(self.0) - rhs).into())
    }
}

impl Add<Word> for Word {
    type Output = Word;

    fn add(self, rhs: Word) -> Word {
        Word((u32::from(self.0) + u32::from(rhs.0)).into())
    }
}

impl Sub<Word> for Word {
    type Output = Word;

    fn sub(self, rhs: Word) -> Word {
        Word((u32::from(self.0) - u32::from(rhs.0)).into())
    }
}

impl AddAssign<u32> for Word {
    fn add_assign(&mut self, rhs: u32) {
        self.0 = (u32::from(self.0) + rhs).into()
    }
}

impl SubAssign<u32> for Word {
    fn sub_assign(&mut self, rhs: u32) {
        self.0 = (u32::from(self.0) - rhs).into()
    }
}

impl AddAssign<Word> for Word {
    fn add_assign(&mut self, rhs: Word) {
        self.0 = (u32::from(self.0) + u32::from(rhs.0)).into()
    }
}

impl SubAssign<Word> for Word {
    fn sub_assign(&mut self, rhs: Word) {
        self.0 = (u32::from(self.0) - u32::from(rhs.0)).into()
    }
}

impl From<usize> for Word {
    fn from(val: usize) -> Word {
        Word(val.into())
    }
}

#[cfg_attr(feature = "facet", derive(Facet))]
#[derive(Debug, Copy, Clone)]
pub struct FreeBlock {
    #[cfg(feature = "dev")]
    pub pos: Word,
    #[cfg(feature = "dev")]
    pub size: Word,

    #[cfg(not(feature = "dev"))]
    pos: Word,
    #[cfg(not(feature = "dev"))]
    size: Word,
}

impl FreeBlock {
    fn new(pos: Word, size: Word) -> Self {
        Self { pos, size }
    }

    pub fn end(&self) -> Word {
        self.pos + self.size
    }
}

#[cfg_attr(feature = "facet", derive(Facet))]
#[derive(Debug)]
pub struct MMU {
    // This is the raw memory managed by the MMU
    #[cfg(feature = "dev")]
    pub mem: Vec<u64>,
    #[cfg(not(feature = "dev"))]
    mem: Vec<u64>,

    // This is a list of chunks of free memory
    // The first value is the position in words
    // The second value is the number of words
    // Sorted for sanity's sake, though I'm not
    // sure if necessary?
    pub free_list: Vec<FreeBlock>,
    // We don't store metadata about any allocations
    // It's up to the caller to know how much memory
    // should be freed.
}

use std::any::TypeId;

macro_rules! alloc {
    ($mmu:expr, $count:expr, $msg:expr) => {
        {
            #[cfg(debug_assertions)]
            {
                //$mmu.alloc_debug($count, $msg)
                $mmu.alloc_no_debug($count)
            }

            #[cfg(not(debug_assertions))]
            {
                $mmu.alloc_no_debug($count)
            }
        }
    };
}

impl MMU {
    fn eprint_free_list(&self) {
        eprintln!("Free list:");
        for block in self.free_list.iter() {
            eprintln!("    {block:?}");
        }
    }

    fn with_capacity(word_count: Word) -> Self {
        let mem = vec![0; usize::from(word_count.0)];
        // Start the free list at word 1, reserving word 0 as a sentinel.
        // This ensures no allocation ever returns position 0, which is used
        // as a "null" / "no scope" sentinel by consumers.
        let free_list = vec![FreeBlock::new(Word(1.into()), word_count - Word(1.into()))];
        Self {
            mem: mem,
            free_list: free_list,
        }
    }

    /*
    fn compact_free_list() {
        todo!("compact_free_list not implemented");
    }
    */

    unsafe fn get<T: 'static>(&self, word: Word) -> &T {
        if TypeId::of::<T>() == TypeId::of::<Word>() {
            panic!("Can't MMU::get<Word>");
        }

        unsafe {
            let ptr: *const T = std::mem::transmute(&self.mem[usize::from(word.0)]);
            &*ptr
        }
    }

    unsafe fn get_mut<T>(&mut self, word: Word) -> &mut T {
        unsafe {
            let ptr: *mut T = std::mem::transmute(&mut self.mem[usize::from(word.0)]);
            &mut *ptr
        }
    }

    fn alloc_and_set<T>(&mut self, value: T, _debug_name: &str) -> Word {
        let word_count = Word((std::mem::size_of::<T>() as u32).div_ceil(8).into());
        let position = alloc!(self, word_count, _debug_name);
        unsafe {
            let ptr: *mut T = std::mem::transmute(&mut self.mem[usize::from(position.0)]);
            ptr.write(value);
        }
        position
    }

    fn alloc_str_raw(&mut self, contents: &[u8]) -> Word {
        let total_len = contents.len().div_ceil(8);
        let word_count = Word(total_len.into());
        let position = alloc!(self, word_count, &format!("str `{}`", debug_u8s(contents)));

        let bytes: &mut [u8] = unsafe {
            let u64_slice = &mut self.mem[
                usize::from(position.0)..
                (usize::from(position.0)+total_len)
            ];
            std::slice::from_raw_parts_mut(
                u64_slice.as_mut_ptr() as *mut u8,
                contents.len(),
            )
        };

        for (idx, b) in contents.iter().enumerate() {
            bytes[idx] = *b;
        }

        position
    }

    fn alloc_str(&mut self, contents: &[u8]) -> ShimValue {
        assert!(contents.len() <= u16::MAX as usize, "String length exceeds u16::MAX");
        let pos = self.alloc_str_raw(contents);
        ShimValue::String(contents.len() as u16, 0, pos.0)
    }

    fn alloc_dict_raw(&mut self) -> Word {
        let word_count = Word((std::mem::size_of::<ShimDict>() as u32).div_ceil(8).into());
        let position = alloc!(self, word_count, "Dict");
        unsafe {
            let ptr: *mut ShimDict =
                std::mem::transmute(&mut self.mem[usize::from(position.0)]);
            ptr.write(ShimDict::new());
        }
        position
    }

    fn alloc_dict(&mut self) -> ShimValue {
        ShimValue::Dict(self.alloc_dict_raw())
    }

    fn alloc_list(&mut self) -> ShimValue {
        let word_count = Word((std::mem::size_of::<ShimList>() as u32).div_ceil(8).into());
        let position = alloc!(self, word_count, "List");
        unsafe {
            let ptr: *mut ShimList =
                std::mem::transmute(&mut self.mem[usize::from(position.0)]);
            ptr.write(ShimList::new());
        }
        ShimValue::List(position)
    }

    fn alloc_fn(&mut self, pc: u32, name: &[u8], captured_scope: u32) -> ShimValue {
        let word_count = Word((std::mem::size_of::<ShimFn>() as u32).div_ceil(8).into());
        let position = alloc!(self, word_count, &format!("Fn `{}`", debug_u8s(name)));
        
        // Allocate the name string
        let name_pos = self.alloc_str_raw(name);
        
        unsafe {
            let ptr: *mut ShimFn =
                std::mem::transmute(&mut self.mem[usize::from(position.0)]);
            ptr.write(ShimFn { pc, name_len: name.len() as u16, name: name_pos, captured_scope });
        }
        ShimValue::Fn(position)
    }

    fn alloc_native<T: ShimNative>(&mut self, val: T) -> ShimValue {
        assert!(std::mem::size_of::<Box<dyn ShimNative>>() == 16);
        let word_count = Word(2.into());
        let position = alloc!(self, word_count, "Native");
        unsafe {
            let ptr: *mut Box<dyn ShimNative> =
                std::mem::transmute(&mut self.mem[usize::from(position.0)]);
            ptr.write(Box::new(val));
        }
        ShimValue::Native(position)
    }

    fn alloc_bound_native_fn(&mut self, obj: &ShimValue, func: NativeFn) -> ShimValue {
        let position = alloc!(self, Word(2.into()), "Bound Native Fn");
        unsafe {
            let obj_ptr: *mut ShimValue =
                std::mem::transmute(&mut self.mem[usize::from(position.0)]);
            obj_ptr.write(*obj);
            let fn_ptr: *mut NativeFn = std::mem::transmute(
                &mut self.mem[usize::from(position.0) + 1],
            );
            fn_ptr.write(func);

            ShimValue::BoundNativeMethod(position)
        }
    }

    fn alloc_debug(&mut self, words: Word, msg: &str) -> Word {
        let result = self.alloc_no_debug(words);
        eprintln!("Alloc {} {}: {}", usize::from(words.0), msg, usize::from(result));
        result
    }

    fn alloc_no_debug(&mut self, words: Word) -> Word {
        if u32::from(words.0) == 0u32 {
            return Word(0.into());
        }
        for idx in 0..self.free_list.len() {
            if self.free_list[idx].size >= words {
                let returned_pos: Word = self.free_list[idx].pos;

                if self.free_list[idx].size == words {
                    self.free_list.remove(idx);
                } else {
                    self.free_list[idx].pos += words;
                    self.free_list[idx].size -= words;
                }

                // Compaction is handled when it's convenient.
                // Some people might tend towards using a linked list to have
                // constant time insert/deletion without needing a separate
                // compaction step, but I'm guessing that iterating through
                // linear memory is going to be pretty fast.
                //
                // Another option is to allocate from the end of the Vec so
                // that we can at least pop off chunks as they're depleted.
                //
                // Or we could keep track of how many empty elements there are
                // in `free_list` so that we can skip them until the next compaction.
                //
                // There are further enhancements if we split things into buckets,
                // but we can keep things simple for now.

                return returned_pos;
            }
        }
        panic!(
            "Could not allocate {:?} words from free list {:#?} (total: {})",
            words, self.free_list, self.mem.len()
        );
    }

    /**
     * Returns the position in `self.mem` of the block allocted
     */
    fn alloc(&mut self, size: Word) -> Word {
        self.alloc_debug(size, "Unspecified alloc")
    }

    fn free(&mut self, pos: Word, size: Word) {
        if u32::from(size.0) == 0 || u32::from(size.0) == 0 {
            return;
        }

        // eprintln!("Free {}: {}", usize::from(size.0), usize::from(pos));

        // This is the idx of the frst free block containing addresses greater than the
        // position we need to free
        let idx = {
            let mut ret = None;
            for idx in 0..self.free_list.len() {
                if pos < self.free_list[idx].end() {
                    ret = Some(idx);
                    break;
                }
            }
                // Technically we could get here if there was no free block at the end
                // of the memory, but we basically don't expect that to happen, so it's
                // not worth addressing.
            ret.expect("Could not find free list position to insert free mem")
        };

        // The data we're freeing is in one of the four categories:
        //   1. needs to be joined to the end of the previous idx
        //   2. joins the previous idx and this idx
        //   3. sits between the previous idx and this idx
        //   4. needs to be joined to the start of this idx
        if idx != 0 {
            if pos == self.free_list[idx-1].end() {
                // Case 1 or 2
                // Since the position matches the end of the previous
                // block we need to join with it
                if pos + size < self.free_list[idx].pos {
                    // Case 1
                    // It's not long enough to reach the idx block, just
                    // add the sizes
                    self.free_list[idx-1].size += size;
                    return;
                } else if pos + size == self.free_list[idx].pos {
                    // Case 2
                    self.free_list[idx-1].size = self.free_list[idx].end() - self.free_list[idx-1].pos;
                    self.free_list.remove(idx);
                    return;
                } else {
                    panic!("Mis-sized free does not fit in gap!");
                }
            }
        }
        if pos + size < self.free_list[idx].pos {
            // Case 3
            self.free_list.insert(idx, FreeBlock::new(pos, size));
            return;
        } else if pos + size == self.free_list[idx].pos {
            // Case 4
            self.free_list[idx].pos = pos;
            self.free_list[idx].size += size;
            return;
        } else {
            panic!("Mis-sized free overlaps with idx block!");
        }
    }
}

// Wrapper structure that chains scopes for the environment.
// Variables are stored in a contiguous block of [len: u8][ident_bytes: [u8; len]][value: ShimValue]
// entries stored inline in a single MMU allocation. Lookups scan raw &[u8] bytes directly —
// no allocations, no hashing, no probing.
//
// Each entry occupies 1 + name_len + 8 bytes. For a typical variable name of ~6 bytes, that's
// 15 bytes per entry. A scope starts with capacity 0 and lazily allocates on first insert.
#[derive(Debug)]
struct EnvScope {
    // Pointer to the contiguous data block in MMU (Word(0) when capacity is 0)
    data: Word,
    // Allocated size of the data block in u64 words
    capacity: u32,
    // Used size of the data block in bytes
    used: u32,
    // Pointer to the parent scope in MMU (0 means no parent)
    parent: u24,
    // Depth of this scope in the chain (root is 1)
    depth: u32,
}

// Default capacity when a scope's data block is first allocated (in u64 words).
// 16 words = 128 bytes, enough for ~8 variables with 6-byte names before needing to grow.
const ENV_SCOPE_DEFAULT_CAPACITY: u32 = 16;

impl EnvScope {
    fn new() -> Self {
        Self {
            data: Word(0.into()),
            capacity: 0,
            used: 0,
            parent: 0.into(),
            depth: 1,
        }
    }

    fn new_with_parent(parent_pos: u24, parent_depth: u32) -> Self {
        Self {
            data: Word(0.into()),
            capacity: 0,
            used: 0,
            parent: parent_pos,
            depth: parent_depth + 1,
        }
    }

    /// Get a byte slice view of the used portion of this scope's data block.
    /// Safety: `self.data` must be a valid MMU word pointing to at least `self.capacity`
    /// words, and `self.used` must be <= `self.capacity * 8`.
    unsafe fn raw_bytes<'a>(&self, mem: &'a MMU) -> &'a [u8] {
        if self.used == 0 {
            return &[];
        }
        let start = usize::from(self.data.0);
        let word_count = (self.used as usize).div_ceil(8);
        let u64_slice = &mem.mem[start..start + word_count];
        let ptr = u64_slice.as_ptr() as *const u8;
        unsafe { std::slice::from_raw_parts(ptr, self.used as usize) }
    }

    /// Get a mutable byte slice view of the full capacity of a scope's data block.
    /// Takes explicit data/capacity to avoid borrow conflicts when the EnvScope
    /// reference is obtained via raw pointer.
    unsafe fn raw_bytes_mut_from(mem: &mut MMU, data: Word, capacity: u32) -> &mut [u8] {
        let start = usize::from(data.0);
        let u64_slice = &mut mem.mem[start..start + capacity as usize];
        let ptr = u64_slice.as_mut_ptr() as *mut u8;
        unsafe { std::slice::from_raw_parts_mut(ptr, capacity as usize * 8) }
    }

    /// Scan this scope's data block for `key`, returning the byte offset of
    /// the value (ShimValue) within the block, or None if not found.
    /// Layout per entry: [len: u8][ident_bytes: [u8; len]][value: ShimValue (8 bytes)]
    fn scan_for_key(&self, mem: &MMU, key: &[u8]) -> Option<usize> {
        let bytes = unsafe { self.raw_bytes(mem) };
        scan_for_key(bytes, key)
    }

    /// Write a ShimValue at the given byte offset within this scope's data block.
    /// Safety: `value_offset + 8` must be within capacity.
    unsafe fn write_value_at(mem: &mut MMU, data: Word, capacity: u32, value_offset: usize, val: ShimValue) {
        unsafe {
            let buf = EnvScope::raw_bytes_mut_from(mem, data, capacity);
            let val_bytes: [u8; 8] = std::mem::transmute(val);
            std::ptr::copy_nonoverlapping(val_bytes.as_ptr(), buf[value_offset..].as_mut_ptr(), 8);
        }
    }

    /// Reallocate the data block to `new_capacity` words, copying `used` bytes of
    /// existing data. Frees the old block if `capacity > 0`. Returns the new data pointer.
    fn realloc(mem: &mut MMU, data: Word, capacity: u32, used: u32, new_capacity: u32) -> Word {
        let new_data = alloc!(mem, Word(new_capacity.into()), "EnvScope data grow");
        // Copy old data
        if used > 0 {
            let old_start = usize::from(data.0);
            let new_start = usize::from(new_data.0);
            let old_word_count = (used as usize).div_ceil(8);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mem.mem.as_ptr().add(old_start),
                    mem.mem.as_mut_ptr().add(new_start),
                    old_word_count,
                );
            }
        }
        // Free old block (only if there was one)
        if capacity > 0 {
            mem.free(data, Word(capacity.into()));
        }
        new_data
    }
}

/// Scan a contiguous scope data block (as raw bytes) for `key`, returning the byte
/// offset of the value (ShimValue) within the block, or None if not found.
fn scan_for_key(bytes: &[u8], key: &[u8]) -> Option<usize> {
    let mut offset = 0usize;
    while offset < bytes.len() {
        let entry_key_len = bytes[offset] as usize;
        let entry_key_start = offset + 1;
        let entry_key_end = entry_key_start + entry_key_len;
        let value_offset = entry_key_end;
        // Each entry is 1 + key_len + 8 bytes
        let entry_end = value_offset + 8;
        if entry_end > bytes.len() {
            break;
        }
        if entry_key_len == key.len() && &bytes[entry_key_start..entry_key_end] == key {
            return Some(value_offset);
        }
        offset = entry_end;
    }
    None
}

#[derive(Debug)]
pub struct Environment {
    // Points to the current EnvScope in MMU
    // u32 is used as u24 converted to u32, 0 means no scope (empty environment)
    current_scope: u32,
}

impl Environment {
    pub fn new(mem: &mut MMU) -> Self {
        // Allocate an EnvScope wrapper (data block allocated lazily on first insert)
        let scope_pos = mem.alloc_and_set(EnvScope::new(), "EnvScope");

        Self {
            current_scope: scope_pos.0.into(),
        }
    }

    pub fn with_scope(captured_scope: u32) -> Self {
        Self {
            current_scope: captured_scope,
        }
    }

    pub fn new_with_builtins(interpreter: &mut Interpreter) -> Self {
        let mut env = Self::new(&mut interpreter.mem);
        let builtins: &[(&[u8], Box<NativeFn>)] = &[
            (b"print", Box::new(shim_print)),
            (b"panic", Box::new(shim_panic)),
            (b"dict", Box::new(shim_dict)),
            (b"Range", Box::new(shim_range)),
            (b"assert", Box::new(shim_assert)),
            (b"str", Box::new(shim_str)),
            (b"int", Box::new(shim_int)),
            (b"float", Box::new(shim_float)),
            (b"try_int", Box::new(shim_try_int)),
            (b"try_float", Box::new(shim_try_float)),
        ];

        for (name, func) in builtins {
            let position = interpreter.mem.alloc_and_set(**func, &format!("builtin func {}", debug_u8s(name)));
            env.insert_new(interpreter, name.to_vec(), ShimValue::NativeFn(position));
        }

        env
    }

    fn insert_new(&mut self, interpreter: &mut Interpreter, key: Vec<u8>, val: ShimValue) {
        assert!(key.len() <= u8::MAX as usize, "Key length {} exceeds maximum {}", key.len(), u8::MAX);

        // Check if key already exists in the current scope — update in place (upsert)
        let scope: &EnvScope = unsafe { interpreter.mem.get(Word(self.current_scope.into())) };
        if let Some(value_offset) = scope.scan_for_key(&interpreter.mem, &key) {
            let (data, capacity) = (scope.data, scope.capacity);
            unsafe { EnvScope::write_value_at(&mut interpreter.mem, data, capacity, value_offset, val); }
            return;
        }

        // Read current scope header via raw pointer to avoid borrow issues
        let (data, capacity, used) = unsafe {
            let scope_ptr: *mut EnvScope =
                interpreter.mem.mem[usize::from(u24::from(self.current_scope))..].as_mut_ptr() as *mut EnvScope;
            ((*scope_ptr).data, (*scope_ptr).capacity, (*scope_ptr).used)
        };

        // Key not found — append new entry
        let entry_size = 1 + key.len() + 8; // len byte + ident bytes + ShimValue
        let new_used = used as usize + entry_size;

        // Grow if needed (also handles initial allocation when capacity == 0)
        let (data, capacity) = if new_used > capacity as usize * 8 {
            let mut new_capacity = if capacity == 0 { ENV_SCOPE_DEFAULT_CAPACITY } else { capacity * 2 };
            while new_used > new_capacity as usize * 8 {
                new_capacity *= 2;
            }
            let new_data = EnvScope::realloc(&mut interpreter.mem, data, capacity, used, new_capacity);
            (new_data, new_capacity)
        } else {
            (data, capacity)
        };

        // Update scope header (data/capacity may have changed)
        unsafe {
            let scope_ptr: *mut EnvScope =
                interpreter.mem.mem[usize::from(u24::from(self.current_scope))..].as_mut_ptr() as *mut EnvScope;
            (*scope_ptr).data = data;
            (*scope_ptr).capacity = capacity;
        }

        // Append entry: [len: u8][ident_bytes][value: ShimValue (8 bytes)]
        unsafe {
            let buf = EnvScope::raw_bytes_mut_from(&mut interpreter.mem, data, capacity);
            let off = used as usize;
            buf[off] = key.len() as u8;
            buf[off + 1..off + 1 + key.len()].copy_from_slice(&key);
        }
        unsafe { EnvScope::write_value_at(&mut interpreter.mem, data, capacity, used as usize + 1 + key.len(), val); }

        // Update used in scope header
        unsafe {
            let scope_ptr: *mut EnvScope =
                interpreter.mem.mem[usize::from(u24::from(self.current_scope))..].as_mut_ptr() as *mut EnvScope;
            (*scope_ptr).used = new_used as u32;
        }
    }

    fn update(&mut self, interpreter: &mut Interpreter, key: &[u8], val: ShimValue) -> Result<(), String> {
        // Walk the scope chain to find the key
        let mut current_scope_pos = self.current_scope;

        loop {
            if current_scope_pos == 0 {
                break;
            }

            let (parent, data, capacity, value_offset) = unsafe {
                let scope: &EnvScope = interpreter.mem.get(Word(current_scope_pos.into()));
                (scope.parent, scope.data, scope.capacity, scope.scan_for_key(&interpreter.mem, key))
            };

            if let Some(value_offset) = value_offset {
                unsafe { EnvScope::write_value_at(&mut interpreter.mem, data, capacity, value_offset, val); }
                return Ok(());
            }

            current_scope_pos = parent.into();
        }

        Err(format!("Key {:?} not found in environment", key))
    }

    fn get(&self, interpreter: &mut Interpreter, key: &[u8]) -> Option<ShimValue> {
        let mut current_scope_pos = self.current_scope;

        loop {
            if current_scope_pos == 0 {
                break;
            }

            let (parent, value_offset) = unsafe {
                let scope: &EnvScope = interpreter.mem.get(Word(current_scope_pos.into()));
                (scope.parent, scope.scan_for_key(&interpreter.mem, key))
            };

            if let Some(value_offset) = value_offset {
                // Read the ShimValue from the byte offset
                let val: ShimValue = unsafe {
                    let scope: &EnvScope = interpreter.mem.get(Word(current_scope_pos.into()));
                    let bytes = scope.raw_bytes(&interpreter.mem);
                    let mut val_bytes = [0u8; 8];
                    std::ptr::copy_nonoverlapping(bytes[value_offset..].as_ptr(), val_bytes.as_mut_ptr(), 8);
                    std::mem::transmute(val_bytes)
                };
                return Some(val);
            }

            current_scope_pos = parent.into();
        }

        None
    }

    fn contains_key(&self, interpreter: &mut Interpreter, key: &[u8]) -> bool {
        self.get(interpreter, key).is_some()
    }

    fn push_scope(&mut self, mem: &mut MMU) {
        // Get current scope depth
        let current_depth = if self.current_scope == 0 {
            0
        } else {
            let current: &EnvScope = unsafe {
                mem.get(Word(self.current_scope.into()))
            };
            current.depth
        };
        
        // Allocate a new EnvScope with parent pointing to current scope
        // (data block allocated lazily on first insert)
        let scope_pos = mem.alloc_and_set(
            EnvScope::new_with_parent(self.current_scope.into(), current_depth),
            "EnvScope"
        );
        
        // Update current scope to the new one
        self.current_scope = scope_pos.0.into();
    }

    fn pop_scope(&mut self, mem: &MMU) -> Result<(), String> {
        if self.current_scope == 0 {
            return Err(format!("Ran out of scopes to pop!"));
        }
        
        // Get the current EnvScope
        let scope: &EnvScope = unsafe {
            mem.get(Word(self.current_scope.into()))
        };
        
        // Move to parent scope
        let parent: u32 = scope.parent.into();
        if parent == 0 {
            return Err(format!("Cannot pop root scope!"));
        }
        
        self.current_scope = parent;
        Ok(())
    }
    
    // Helper to get the depth of the current scope
    fn scope_depth(&self, mem: &MMU) -> usize {
        if self.current_scope == 0 {
            return 0;
        }
        
        let scope: &EnvScope = unsafe {
            mem.get(Word(self.current_scope.into()))
        };
        scope.depth as usize
    }
}

// TODO: If we do NaN-boxing we could have f64 (rather than f32) for "free"
#[derive(Copy, Clone, Debug)]
pub enum ShimValue {
    Uninitialized,
    Unit,
    None,
    Integer(i32),
    Float(f32),
    Bool(bool),
    // Memory position pointing to ShimFn structure
    Fn(Word),
    BoundMethod(
        // Object
        Word,
        // Fn memory position pointing to ShimFn structure
        Word,
    ),
    BoundNativeMethod(
        // ShimValue followed by NativeFn
        Word,
    ),
    // A function pointer doesn't fit in the ShimValue, so we need to store the
    // function pointer in interpreter memory
    NativeFn(Word),
    // TODO: it seems like this should point to a more generic reference-counted
    // object type that all non-value types share
    String(
        // len
        u16,
        // byte offset within the 8-byte aligned word
        u8,
        // position (word index into memory)
        u24,
    ),
    List(Word),
    Dict(Word),
    StructDef(Word),
    Struct(Word),
    Native(Word),
    // For now this is really only used for GC purposes
    Environment(Word),
}
const _: () = {
    assert!(std::mem::size_of::<ShimValue>() == 8);
};

trait ShimNative: Any {
    fn to_string(&self, _interpreter: &mut Interpreter) -> String {
        format!("{}", type_name::<Self>())
    }

    fn get_attr(&self, _self_as_val: &ShimValue, _interpreter: &mut Interpreter, _ident: &[u8]) -> Result<ShimValue, String> {
        Err(format!("Can't get_attr on {}", type_name::<Self>() ))
    }

    fn set_attr(
        &self,
        _interpreter: &mut Interpreter,
        _ident: &[u8],
        _val: ShimValue,
    ) -> Result<(), String> {
        Err(format!("Can't set_attr on {}", type_name::<Self>() ))
    }

    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn gc_vals(&self) -> Vec<ShimValue>;
}

struct ListIterator {
    lst: ShimValue,
    idx: usize,
}
impl ShimNative for ListIterator {
    fn get_attr(&self, self_as_val: &ShimValue, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_list_iter_next(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err(format!("Can't provide positional args to ListIterator.next()"));
                }

                let itr: &mut ListIterator = args.args[0].as_native(interpreter)?;
                let lst = itr.lst.list(interpreter)?;
                if itr.idx >= lst.len() {
                    Ok(ShimValue::None)
                } else {
                    let result = lst.get(&mut interpreter.mem, itr.idx as isize)?;
                    itr.idx += 1;

                    Ok(result)
                }
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_list_iter_next))
        } else {
            Err(format!("Can't get_attr {} on {}", debug_u8s(ident), type_name::<Self>() ))
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.lst]
    }
}

struct DictKeysIterator {
    dict: ShimValue,
    idx: usize,
}
impl ShimNative for DictKeysIterator {
    fn get_attr(&self, self_as_val: &ShimValue, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_dict_keys_iter_next(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err(format!("Can't provide positional args to DictKeysIterator.next()"));
                }

                let itr: &mut DictKeysIterator = args.args[0].as_native(interpreter)?;
                let dict = itr.dict.dict(interpreter)?;
                let entries = dict.entries_array(interpreter);
                
                // Skip invalid entries (tombstones)
                while itr.idx < entries.len() {
                    if entries[itr.idx].is_valid() {
                        let result = entries[itr.idx].key;
                        itr.idx += 1;
                        return Ok(result);
                    }
                    itr.idx += 1;
                }
                
                Ok(ShimValue::None)
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_dict_keys_iter_next))
        } else if ident == b"iter" {
            fn shim_dict_keys_iter_iter(_interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_dict_keys_iter_iter))
        } else {
            Err(format!("Can't get_attr {} on {}", debug_u8s(ident), type_name::<Self>() ))
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.dict]
    }
}

struct DictValuesIterator {
    dict: ShimValue,
    idx: usize,
}
impl ShimNative for DictValuesIterator {
    fn get_attr(&self, self_as_val: &ShimValue, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_dict_values_iter_next(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err(format!("Can't provide positional args to DictValuesIterator.next()"));
                }

                let itr: &mut DictValuesIterator = args.args[0].as_native(interpreter)?;
                let dict = itr.dict.dict(interpreter)?;
                let entries = dict.entries_array(interpreter);
                
                // Skip invalid entries (tombstones)
                while itr.idx < entries.len() {
                    if entries[itr.idx].is_valid() {
                        let result = entries[itr.idx].value;
                        itr.idx += 1;
                        return Ok(result);
                    }
                    itr.idx += 1;
                }
                
                Ok(ShimValue::None)
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_dict_values_iter_next))
        } else if ident == b"iter" {
            fn shim_dict_values_iter_iter(_interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_dict_values_iter_iter))
        } else {
            Err(format!("Can't get_attr {} on {}", debug_u8s(ident), type_name::<Self>() ))
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.dict]
    }
}

struct DictEntryNative {
    key: ShimValue,
    value: ShimValue,
}
impl ShimNative for DictEntryNative {
    fn get_attr(&self, _self_as_val: &ShimValue, _interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"key" {
            Ok(self.key)
        } else if ident == b"value" {
            Ok(self.value)
        } else {
            Err(format!("Can't get_attr {} on {}", debug_u8s(ident), type_name::<Self>() ))
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.key, self.value]
    }
}

struct DictItemsIterator {
    dict: ShimValue,
    idx: usize,
}
impl ShimNative for DictItemsIterator {
    fn get_attr(&self, self_as_val: &ShimValue, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_dict_items_iter_next(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err(format!("Can't provide positional args to DictItemsIterator.next()"));
                }

                let itr: &mut DictItemsIterator = args.args[0].as_native(interpreter)?;
                let dict = itr.dict.dict(interpreter)?;
                let entries = dict.entries_array(interpreter);
                
                // Skip invalid entries (tombstones)
                while itr.idx < entries.len() {
                    if entries[itr.idx].is_valid() {
                        let entry = &entries[itr.idx];
                        let result = interpreter.mem.alloc_native(DictEntryNative {
                            key: entry.key,
                            value: entry.value,
                        });
                        itr.idx += 1;
                        return Ok(result);
                    }
                    itr.idx += 1;
                }
                
                Ok(ShimValue::None)
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_dict_items_iter_next))
        } else if ident == b"iter" {
            fn shim_dict_items_iter_iter(_interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_dict_items_iter_iter))
        } else {
            Err(format!("Can't get_attr {} on {}", debug_u8s(ident), type_name::<Self>() ))
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.dict]
    }
}

struct RangeNative {
    start: ShimValue,
    end: ShimValue,
}

impl ShimNative for RangeNative {
    fn to_string(&self, interpreter: &mut Interpreter) -> String {
        format!("Range({}, {})", self.start.to_string(interpreter), self.end.to_string(interpreter))
    }

    fn get_attr(&self, self_as_val: &ShimValue, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"step" {
            fn shim_range_step(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                let step = unpacker.required(b"step")?;
                unpacker.end()?;

                let range: &RangeNative = obj.as_native(interpreter)?;
                
                // Check for zero step
                let is_zero = match step {
                    ShimValue::Integer(0) => true,
                    ShimValue::Float(f) if f == 0.0 => true,
                    _ => false,
                };
                
                if is_zero {
                    return Err(format!("Step cannot be zero"));
                }

                let iterator = RangeIterator {
                    current: range.start,
                    end: range.end,
                    step: step,
                };
                Ok(interpreter.mem.alloc_native(iterator))
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_range_step))
        } else if ident == b"iter" {
            fn shim_range_iter(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;

                let range: &RangeNative = obj.as_native(interpreter)?;
                let iterator = RangeIterator {
                    current: range.start,
                    end: range.end,
                    step: ShimValue::Integer(1),
                };
                Ok(interpreter.mem.alloc_native(iterator))
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_range_iter))
        } else {
            Err(format!("Can't get_attr {} on {}", debug_u8s(ident), type_name::<Self>()))
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.start, self.end]
    }
}

struct RangeIterator {
    current: ShimValue,
    end: ShimValue,
    step: ShimValue,
}

impl ShimNative for RangeIterator {
    fn get_attr(&self, self_as_val: &ShimValue, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_range_iter_next(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err(format!("Can't provide positional args to RangeIterator.next()"));
                }

                let itr: &mut RangeIterator = args.args[0].as_native(interpreter)?;
                
                // Determine if we've reached the end based on step direction
                // For positive steps: iterate while current < end
                // For negative steps: iterate while current > end
                let step_is_positive = match itr.step.gt(interpreter, &ShimValue::Integer(0))? {
                    ShimValue::Bool(b) => b,
                    _ => return Err(format!("Step comparison failed")),
                };
                
                let has_more = if step_is_positive {
                    // current < end
                    match itr.current.lt(interpreter, &itr.end)? {
                        ShimValue::Bool(b) => b,
                        _ => return Err(format!("Range comparison failed")),
                    }
                } else {
                    // current > end
                    match itr.current.gt(interpreter, &itr.end)? {
                        ShimValue::Bool(b) => b,
                        _ => return Err(format!("Range comparison failed")),
                    }
                };
                
                if !has_more {
                    Ok(ShimValue::None)
                } else {
                    let result = itr.current;
                    // current = current + step
                    let mut pending_args = ArgBundle::new();
                    match itr.current.add(interpreter, &itr.step, &mut pending_args)? {
                        CallResult::ReturnValue(new_current) => {
                            itr.current = new_current;
                            Ok(result)
                        }
                        CallResult::PC(pc, captured_scope) => {
                            let mut new_env = Environment::with_scope(captured_scope);
                            let new_current = interpreter.execute_bytecode_extended(
                                &mut (pc as usize),
                                pending_args,
                                &mut new_env,
                            )?;
                            itr.current = new_current;
                            Ok(result)
                        }
                    }
                }
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_range_iter_next))
        } else if ident == b"iter" {
            fn shim_range_iterator_iter(_interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_range_iterator_iter))
        } else {
            Err(format!("Can't get_attr {} on {}", debug_u8s(ident), type_name::<Self>()))
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.current, self.end, self.step]
    }
}

type NativeFn = fn(&mut Interpreter, &ArgBundle) -> Result<ShimValue, String>;
const _: () = {
    assert!(std::mem::size_of::<NativeFn>() == 8);
};

fn format_float(val: f32) -> String {
    let s = format!("{val}");
    if !s.contains('.') && !s.contains('e') {
        format!("{s}.0")
    } else {
        s
    }
}

#[derive(Debug, Clone, Copy)]
enum StructAttribute {
    MemberInstanceOffset(u8),
    MethodDef(Word),
}

#[derive(Debug)]
struct StructDef {
    name: Vec<u8>,
    member_count: u8,
    lookup: Vec<(Vec<u8>, StructAttribute)>,
}

const fn generate_size_table() -> [u32; 256] {
    let mut table = [0; 256];

    let mut i = 0;

    while i < 256 {
        table[i] = match i {
            0 => 0,
            1 => 4,
            2 => 16,
            3 => 32,
            // Multiply 1.5 the previous
            4 => 48,
            5 => 72,
            6 => 108,
            7 => 162,
            8 => 243,
            9 => 364,
            10 => 546,
            11 => 819,
            12 => 1228,
            13 => 1842,
            // Multiply x1.2 the previous
            14 => 2210,
            15 => 2652,
            16 => 3182,
            17 => 3818,
            18 => 4581,
            19 => 5497,
            20 => 6596,
            21 => 7915,
            22 => 9498,
            23 => 11397,
            24 => 13676,
            25 => 16411,
            26 => 19693,
            27 => 23631,
            28 => 28357,
            29 => 34028,
            30 => 40833,
            31 => 48999,
            32 => 58798,
            33 => 70557,
            34 => 84668,
            35 => 101601,
            36 => 121921,
            37 => 146305,
            38 => 175566,
            39 => 210679,
            40 => 252814,
            41 => 303376,
            42 => 364051,
            43 => 436861,
            44 => 524233,
            45 => 629079,
            46 => 754894,
            47 => 905872,
            48 => 1087046,
            49 => 1304455,
            50 => 1565346,
            51 => 1878415,
            52 => 2254098,
            53 => 2704917,
            54 => 3245900,
            55 => 3895080,
            56 => 4674096,
            57 => 5608915,
            58 => 6730698,
            59 => 8076837,
            60 => 9692204,
            61 => 11630644,
            62 => 13956772,
            63 => 16748126,
            _ => MAX_U24,
        };
        i += 1;
    }
    table
}

static LIST_CAPACITY_LUT: [u32; 256] = generate_size_table();

#[derive(Debug, Clone, Copy)]
struct DictEntry {
    hash: u64,
    key: ShimValue,
    value: ShimValue,
}

impl DictEntry {
    fn is_valid(&self) -> bool {
        self.hash != 0
            && !self.key.is_uninitialized()
            && !self.value.is_uninitialized()
    }

    fn invalidate(&mut self) {
        self.hash = 0;
        self.key = ShimValue::Uninitialized;
        self.value = ShimValue::Uninitialized;
    }
}

// Minimum non-zero size_pow for ShimDict. When the dict grows from empty,
// it starts with this size_pow value (2^3 = 8 index slots, capacity of ~5 entries).
const MIN_NON_ZERO_SIZE_POW: u8 = 3;

#[derive(Debug)]
struct ShimDict {
    // Size of the index array, always a power of 2
    size_pow: u8,

    // These could be u24, but are u32 to keep things simple

    // Number of valid entries + tombstoned entries
    entry_count: u32,
    // Non-tombstoned entries
    used: u32,

    // Memory position of the dict data
    indices: u24,
    entries: u24,
}

enum DictSlot<'a> {
    Occupied(usize, &'a mut DictEntry),
    // If it's Unoccupied, this is the idx in the indices array
    UnoccupiedU8(u32, usize),
    UnoccupiedU16(u32, usize),
    UnoccupiedU32(u32, usize),
}

#[derive(Debug)]
enum TypedIndices {
    Zero,
    U8(&'static mut [u8]),
    U16(&'static mut [u16]),
    U32(&'static mut [u32]),
}

impl TypedIndices {
    fn get(&self, index: usize) -> usize {
        match self {
            Self::Zero => panic!("Can't index empty TypedIndices"),
            Self::U8(data) => {data[index] as usize},
            Self::U16(data) => {data[index] as usize},
            Self::U32(data) => {data[index] as usize},
        }
    }

    fn set(&mut self, index: usize, value: usize) {
        match self {
            Self::Zero => panic!("Can't IndexMut empty TypedIndices"),
            Self::U8(data) => {data[index] = value as u8},
            Self::U16(data) => {data[index] = value as u16},
            Self::U32(data) => {data[index] = value as u32},
        }
    }

    fn is_unset(&self, index: usize) -> bool {
        match self {
            Self::Zero => panic!("Can't index empty TypedIndices"),
            Self::U8(data) => {data[index] == u8::MAX},
            Self::U16(data) => {data[index] == u16::MAX},
            Self::U32(data) => {data[index] == u32::MAX},
        }
    }

    fn is_tombstone(&self, index: usize) -> bool {
        match self {
            Self::Zero => panic!("Can't index empty TypedIndices"),
            Self::U8(data) => {data[index] == u8::MAX - 1},
            Self::U16(data) => {data[index] == u16::MAX - 1},
            Self::U32(data) => {data[index] == u32::MAX - 1},
        }
    }

    fn set_tombstone(&mut self, index: usize) {
        match self {
            Self::Zero => panic!("Can't index empty TypedIndices"),
            Self::U8(data) => {data[index] = u8::MAX - 1},
            Self::U16(data) => {data[index] = u16::MAX - 1},
            Self::U32(data) => {data[index] = u32::MAX - 1},
        }
    }
}

impl ShimDict {
    fn new() -> Self {
        Self {
            size_pow: 0,
            used: 0,
            entry_count: 0,
            indices: 0.into(),
            entries: 0.into(),
        }
    }

    fn len(&self) -> usize {
        self.used as usize
    }

    fn get(&self, interpreter: &mut Interpreter, key: ShimValue) -> Result<ShimValue, String> {
        // Check if dict is empty
        if self.size_pow == 0 {
            return Err(format!("Key {key:?} not in dict"));
        }
        
        match self.probe(interpreter, key)? {
            DictSlot::Occupied(_, entry) => {
                Ok(entry.value)
            },
            DictSlot::UnoccupiedU8(..) => {
                Err(format!("Key {key:?} not in dict"))
            },
            _ => todo!(),
        }
    }

    fn print_entries(&self, interpreter: &Interpreter) {
        eprintln!("Entries");
        let _entries: &[DictEntry] = unsafe {
            let u64_slice = &interpreter.mem.mem[
                usize::from(self.entries)..
                usize::from(self.entries)+3*(self.entry_count as usize)
            ];
            std::slice::from_raw_parts(
                u64_slice.as_ptr() as *const DictEntry,
                u64_slice.len() / 3,
            )
        };
    }

    fn expand_capacity(&mut self, interpreter: &mut Interpreter) {
        let _zone = zone_scoped!("ShimDict::expand_capacity");
        let old_size = self.index_size();
        let old_capacity = self.capacity();
        self.size_pow = if old_size == 0 {
            MIN_NON_ZERO_SIZE_POW
        } else {
            self.size_pow + 1
        };


        self.clear_and_alloc_indices(interpreter, old_size);
        self.realloc_entries(interpreter, old_capacity);
    }

    fn realloc_entries(&mut self, interpreter: &mut Interpreter, old_capacity: usize) {
        let old_entries_word = self.entries;
        let old_entries = self.entries_array(interpreter);

        let free_word_count = Word((old_capacity * 3).into());
        let alloc_word_count = Word((self.capacity() * 3).into());
        self.entries = alloc!(interpreter.mem, alloc_word_count, "Dict entry array").0;

        let new_entries = self.entries_mut(interpreter);

        let mut write_idx = 0;
        for read_idx in 0..old_entries.len() {
            let entry = old_entries[read_idx];
            if entry.is_valid() {
                new_entries[write_idx] = entry;
                new_entries[write_idx].is_valid();
                write_idx += 1;
            }
        }
        // This should be equal to or lower than the previous entry_count since
        // it will remove tombstones
        self.entry_count = write_idx as u32;

        let new_entries = self.entries_array(interpreter);
        let mut indices = self.typed_indices(interpreter);
        for (entry_idx, entry) in new_entries.iter().enumerate() {
            let index_idx = self.probe_entry_realloc(interpreter, entry.hash as u32);
            indices.set(index_idx, entry_idx);
        }

        interpreter.mem.free(Word(old_entries_word), free_word_count);
    }

    fn indices_stride_bytes(&self, size: usize) -> usize {
        if size == 0 {
            0
        } else if size <= (u8::MAX as usize) + 1 {
            1
        } else if size <= (u16::MAX as usize) + 1 {
            2
        } else {
            4
        }
    }

    fn typed_indices(&self, interpreter: &Interpreter) -> TypedIndices {
        match self.index_size() {
            0 => TypedIndices::Zero,
            x if x <= (u8::MAX as usize) + 1 => TypedIndices::U8(
                self.indicies_mut::<u8>(interpreter)
            ),
            x if x <= (u16::MAX as usize) + 1 => TypedIndices::U16(
                self.indicies_mut::<u16>(interpreter)
            ),
            x if x <= (u32::MAX as usize) + 1 => TypedIndices::U32(
                self.indicies_mut::<u32>(interpreter)
            ),
            _ => panic!("Size over u64"),
        }
    }

    /**
     * Clear the indices array with current size
     */
    fn clear_and_alloc_indices(&mut self, interpreter: &mut Interpreter, old_size: usize) {
        let new_size = self.index_size();
        let free_word_count = if old_size == 0 {
            Word(0.into())
        } else {
            Word(old_size.div_ceil(8 / self.indices_stride_bytes(old_size)).into())
        };
        let alloc_word_count = if new_size == 0 {
            Word(0.into())
        } else {
            Word(new_size.div_ceil(8 / self.indices_stride_bytes(new_size)).into())
        };


        interpreter.mem.free(Word(self.indices), free_word_count);
        self.indices = alloc!(interpreter.mem, alloc_word_count, "Dict index array").0;

        match self.typed_indices(interpreter) {
            TypedIndices::Zero => (),
            TypedIndices::U8(indices) => {
                for x in indices.iter_mut() {
                    *x = u8::MAX;
                }
            },
            TypedIndices::U16(indices) => {
                for x in indices.iter_mut() {
                    *x = u16::MAX;
                }
            },
            TypedIndices::U32(indices) => {
                for x in indices.iter_mut() {
                    *x = u32::MAX;
                }
            },
        }
    }

    fn capacity(&self) -> usize {
        Self::capacity_for_size_pow(self.size_pow)
    }

    fn capacity_for_size_pow(size_pow: u8) -> usize {
        if size_pow == 0 {
            0
        } else {
            let index_size = 1 << size_pow;
            ((index_size * 2) / 3) as usize
        }
    }

    fn index_size(&self) -> usize {
        if self.size_pow == 0 {
            0
        } else {
            (1 << self.size_pow) as usize
        }
    }

    fn mask(&self) -> usize {
        (self.index_size() - 1) as usize
    }

    fn probe_entry_realloc(&self, interpreter: &Interpreter, longhash: u32) -> usize {
        let mask = self.mask();

        let hash: usize = (longhash as usize) & mask;
        let mut idx = hash & mask;
        match self.typed_indices(interpreter) {
            TypedIndices::Zero => panic!("Can't probe empty dict"),
            TypedIndices::U8(indices) => {
                for _ in 0..self.index_size() {
                    if indices[idx] == u8::MAX {
                        return idx;
                    } else if indices[idx] == u8::MAX - 1 {
                        panic!("Found tombstone during dict entry realloc!");
                    }
                    idx = (idx + 1) & mask;
                }
            },
            TypedIndices::U16(indices) => {
                for _ in 0..self.index_size() {
                    if indices[idx] == u16::MAX {
                        return idx;
                    } else if indices[idx] == u16::MAX - 1 {
                        panic!("Found tombstone during dict entry realloc!");
                    }
                    idx = (idx + 1) & mask;
                }
            },
            TypedIndices::U32(indices) => {
                for _ in 0..self.index_size() {
                    if indices[idx] == u32::MAX {
                        return idx;
                    } else if indices[idx] == u32::MAX - 1 {
                        panic!("Found tombstone during dict entry realloc!");
                    }
                    idx = (idx + 1) & mask;
                }
            },
        }

        panic!("Probe entry realloc failed probing!");
    }

    fn probe(&self, interpreter: &mut Interpreter, key: ShimValue) -> Result<DictSlot<'_>, String> {
        let longhash = key.hash(interpreter)? as usize;
        let mask = self.mask();

        let mut idx = longhash & mask;

        let mut freeslot = None;

        let indices = self.typed_indices(interpreter);
        // Linear probe for now
        for _ in 0..self.index_size() {
            if indices.is_unset(idx) {
                if freeslot == None {
                    freeslot = Some(idx);
                }
                break
            } else if indices.is_tombstone(idx) {
                if freeslot == None {
                    freeslot = Some(idx);
                }
            } else {
                // Hash matches, let's check the entry and see if the key matches
                let entry_idx = indices.get(idx);
                let entry = self.get_entry_mut(interpreter, entry_idx as usize);
                if key.equal_inner(interpreter, &entry.key)? {
                    return Ok(DictSlot::Occupied(idx as usize, entry));
                }
                // Otherwise continue probing
            }
            idx = (idx + 1) & mask;
        }
        let idx = match freeslot {
            Some(idx) => idx,
            None => {
                eprintln!("{self:#?}");
                eprintln!("Capacity: {:#?}  Mask: {}", self.capacity(), mask);
                panic!("Could not find free slot");
            },
        };
        match indices {
            TypedIndices::Zero => panic!("probingn nothing"),
            TypedIndices::U8(_) => Ok(DictSlot::UnoccupiedU8(longhash as u32, idx as usize)),
            TypedIndices::U16(_) => Ok(DictSlot::UnoccupiedU16(longhash as u32, idx as usize)),
            TypedIndices::U32(_) => Ok(DictSlot::UnoccupiedU32(longhash as u32, idx as usize)),
        }
    }

    fn set(&mut self, interpreter: &mut Interpreter, key: ShimValue, val: ShimValue) -> Result<(), String> {
        if self.entry_count as usize == self.capacity() {
            self.expand_capacity(interpreter);
        }

        match self.probe(interpreter, key)? {
            DictSlot::Occupied(_, entry) => {
                entry.key = key;
                entry.value = val;
            },
            DictSlot::UnoccupiedU8(longhash, idx) => {
                let entry_idx = self.set_entry(interpreter, longhash, key, val);
                self.indicies_mut::<u8>(interpreter)[idx] = entry_idx as u8;
                self.entries_mut(interpreter)[entry_idx].is_valid();
                self.entries_array(interpreter)[entry_idx].is_valid();
                self.used += 1;
            },
            DictSlot::UnoccupiedU16(longhash, idx) => {
                let entry_idx = self.set_entry(interpreter, longhash, key, val);
                self.indicies_mut::<u16>(interpreter)[idx] = entry_idx as u16;
                self.used += 1;
            },
            DictSlot::UnoccupiedU32(longhash, idx) => {
                let entry_idx = self.set_entry(interpreter, longhash, key, val);
                self.indicies_mut::<u32>(interpreter)[idx] = entry_idx as u32;
                self.used += 1;
            },
        }

        Ok(())
    }

    fn pop(&mut self, interpreter: &mut Interpreter, key: ShimValue, default: Option<ShimValue>) -> Result<ShimValue, String> {
        match self.probe(interpreter, key) {
            Ok(DictSlot::Occupied(indices_idx, entry)) => {
                let value = entry.value;
                entry.hash = 0;
                entry.key = ShimValue::Uninitialized;
                entry.value = ShimValue::Uninitialized;

                let mut indices = self.typed_indices(interpreter);
                indices.set_tombstone(indices_idx);

                // We don't decrement the entry_count since that entry still exists
                self.used -= 1;

                Ok(value)
            },
            Ok(_) => {
                if let Some(default) = default {
                    Ok(default)
                } else {
                    Err(format!("Key {key:?} not found in dict"))
                }
            },
            _ => todo!(),
        }
    }

    fn indicies_mut<T>(&self, interpreter: &Interpreter) -> &'static mut [T] {
        let stride = std::mem::size_of::<T>();
        let size = 1 << self.size_pow;
        let start = usize::from(self.indices);
        let len = size / stride;
        let u64_slice = &interpreter.mem.mem[start..start + len];
        unsafe {
            std::slice::from_raw_parts_mut(
                u64_slice.as_ptr() as *mut T,
                u64_slice.len() * stride,
            )
        }
    }

    /**
     * Return the valid part of the entries array
     */
    fn entries_array(&self, interpreter: &Interpreter) -> &'static [DictEntry] {
        unsafe {
            let u64_slice = &interpreter.mem.mem[
                usize::from(self.entries)..
                usize::from(self.entries)+3*(self.entry_count as usize)
            ];
            std::slice::from_raw_parts(
                u64_slice.as_ptr() as *const DictEntry,
                u64_slice.len() / 3,
            )
        }
    }

    /**
     * Return the entire capacity of the entries table
     */
    fn entries_mut(&self, interpreter: &mut Interpreter) -> &'static mut [DictEntry] {
        unsafe {
            let u64_slice = &mut interpreter.mem.mem[
                usize::from(self.entries)..
                usize::from(self.entries)+3*(self.capacity() as usize)
            ];
            std::slice::from_raw_parts_mut(
                u64_slice.as_mut_ptr() as *mut DictEntry,
                u64_slice.len() / 3,
            )
        }
    }

    fn get_entry(&self, interpreter: &Interpreter, idx: usize) -> &DictEntry {
        unsafe{std::mem::transmute(&interpreter.mem.mem[
            usize::from(self.entries)+3*idx
        ])}
    }

    fn get_entry_mut(&self, interpreter: &mut Interpreter, idx: usize) -> &mut DictEntry {
        unsafe{std::mem::transmute(&mut interpreter.mem.mem[
            usize::from(self.entries)+3*idx
        ])}
    }

    fn set_entry(&mut self, interpreter: &mut Interpreter, hash: u32, key: ShimValue, val: ShimValue) -> usize {
        let entry = self.get_entry_mut(interpreter, self.entry_count as usize);
        entry.hash = hash as u64;
        entry.key = key;
        entry.value = val;

        let entry_idx = self.entry_count;
        self.entry_count += 1;
        entry_idx as usize
    }

    fn shrink_to_fit(&mut self, interpreter: &mut Interpreter) {
        if self.used == 0 {
            // Empty dict - reset to minimal size
            let old_size = self.index_size();
            let old_capacity = self.capacity();
            
            if old_size == 0 {
                return; // Already minimal
            }
            
            self.size_pow = 0;
            self.clear_and_alloc_indices(interpreter, old_size);
            
            // Free the old entries
            let free_word_count = Word((old_capacity * 3).into());
            interpreter.mem.free(Word(self.entries), free_word_count);
            self.entries = 0.into();
            self.entry_count = 0;
            return;
        }
        
        // Calculate the optimal size_pow for the current number of used entries
        // We want capacity to be at least used, and index_size = capacity * 3 / 2
        // Since index_size must be a power of 2, we find the smallest power of 2
        // such that (2^size_pow * 2 / 3) >= used
        let min_capacity = self.used as usize;
        // Start with MIN_NON_ZERO_SIZE_POW, which matches expand_capacity's initial size
        let mut optimal_size_pow = MIN_NON_ZERO_SIZE_POW;
        
        // Upper bound of 31 prevents undefined behavior from 1 << 32 and ensures
        // we stay within u32 limits for entry_count/used fields.
        // Loop condition is <= 31 to allow checking if size_pow=31 is sufficient.
        while optimal_size_pow <= 31 {
            let test_capacity = Self::capacity_for_size_pow(optimal_size_pow);
            if test_capacity >= min_capacity {
                break;
            }
            optimal_size_pow += 1;
        }
        
        // If the optimal size is the same or larger than current, no need to shrink
        if optimal_size_pow >= self.size_pow {
            return;
        }
        
        let old_size = self.index_size();
        let old_capacity = self.capacity();
        self.size_pow = optimal_size_pow;
        
        self.clear_and_alloc_indices(interpreter, old_size);
        self.realloc_entries(interpreter, old_capacity);
    }
}

struct ShimList {
    // The memory is limited to u24, so we know there can't be more than this
    // number of values
    len: u24,
    // We don't really need any more than 64 distinct capacities
    capacity_lut: u8,
    // Add 1 byte of padding so that ShimList is 8 bytes
    _pad: u8,
    // Memory position of the list data
    data: u24,
}

const _: () = {
    assert!(std::mem::size_of::<ShimList>() == 8);
};

impl ShimList {
    fn new() -> Self {
        Self {
            len: 0.into(),
            capacity_lut: 0,
            _pad: 0,
            data: 0.into(),
        }
    }

    fn len(&self) -> usize {
        self.len.into()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn capacity(&self) -> usize {
        LIST_CAPACITY_LUT[self.capacity_lut as usize] as usize
    }

    fn wrap_idx(&self, idx: isize) -> Result<usize, String> {
        if idx >= self.len() as isize {
            return Err(format!("Index {idx} is out of bounds"));
        }

        Ok(
            if idx < 0 {
                let updated_idx = self.len() as isize + idx;
                if updated_idx < 0 {
                    return Err(format!("Index {idx} is out of bounds"));
                } else {
                    updated_idx as usize
                }
            } else {
                idx as usize
            }
        )
    }

    fn get(&self, mem: &MMU, idx: isize) -> Result<ShimValue, String> {
        let idx = self.wrap_idx(idx)?;
        unsafe { Ok(ShimValue::from_u64(mem.mem[usize::from(self.data) + idx])) }
    }

    fn set(&self, mem: &mut MMU, idx: isize, value: ShimValue) -> Result<(), String> {
        let idx = self.wrap_idx(idx)?;
        mem.mem[usize::from(self.data) + idx] = value.to_u64();
        Ok(())
    }

    fn push(&mut self, mem: &mut MMU, val: ShimValue) {
        if self.len() == self.capacity() {
            let old_capacity = self.capacity();
            self.capacity_lut += 1;
            let new_capacity = self.capacity();

            let old_data = usize::from(self.data);
            let word_count: Word = new_capacity.into();
            self.data = alloc!(mem, word_count, "List data").0;

            let new_data = usize::from(self.data);

            for idx in 0..self.len() {
                mem.mem[new_data+idx] = mem.mem[old_data+idx];
            }

            mem.free(old_data.into(), Word(old_capacity.into()));
        }

        mem.mem[usize::from(self.data)+self.len()] = val.to_u64();
        self.len = (usize::from(self.len) + 1).into();
    }
}
const _: () = { assert!(std::mem::size_of::<ShimList>() == 8); };

// Stores function information in interpreter memory
struct ShimFn {
    // Program counter where the function code begins
    pc: u32,
    // Length of the function name string
    name_len: u16,
    // Memory position of the function name (stored as string)
    name: Word,
    // The environment scope where this function was defined (for closures)
    captured_scope: u32,
}

const _: () = {
    assert!(std::mem::size_of::<ShimFn>() == 16);
};

impl StructDef {
    fn find(&self, ident: &[u8]) -> Option<StructAttribute> {
        for (attr, loc) in self.lookup.iter() {
            if ident == attr {
                return Some(*loc)
            }
        }
        None
    }

    fn mem_size(&self) -> usize {
        // TODO: if the StructDef changes it might be effectively non const sized
        // in interpreter memory
        const _: () = {
            assert!(std::mem::size_of::<StructDef>() == 56);
        };
        std::mem::size_of::<StructDef>() / 8
    }
}

#[derive(Debug)]
enum CallResult {
    ReturnValue(ShimValue),
    PC(u32, u32), // PC and captured_scope
}

fn shim_dict(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    if args.args.len() != 0 {
        return Err(format!("Can't provide positional args to dict()"));
    }

    let retval = interpreter.mem.alloc_dict();
    let dict = retval.dict_mut(interpreter)?;

    for (key, val) in args.kwargs.clone().into_iter() {
        let key = interpreter.mem.alloc_str(&key);
        dict.set(interpreter, key, val)?;
    }

    Ok(retval)
}

fn shim_range(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let start = unpacker.required(b"start")?;
    let end = unpacker.required(b"end")?;
    unpacker.end()?;

    let range = RangeNative {
        start: start,
        end: end,
    };
    Ok(interpreter.mem.alloc_native(range))
}

fn shim_print(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let _zone = zone_scoped!("shim_print");
    for (idx, arg) in args.args.iter().enumerate() {
        if idx != 0 {
            print!(" ");
        }
        print!("{}", arg.to_string(interpreter));
    }

    println!();
    Ok(ShimValue::None)
}

fn shim_assert(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    if !args.kwargs.is_empty() {
        return Err(format!("Assert doesn't take keyword arguments"));
    }
    if args.len() > 2 {
        return Err(format!("Assert got more than two arguments! {:?}", args));
    }
    if args.len() == 0 {
        return Ok(ShimValue::None);
    }

    if !args.args[0].is_truthy(interpreter)? {
        let msg = if args.len() > 1 {
            args.args[1].to_string(interpreter)
        } else {
            format!("Assert Failed: {:?} not truthy", args.args[0])
        };
        Err(msg)
    } else {
        Ok(ShimValue::None)
    }
}

fn shim_panic(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut out = String::new();
    for (idx, arg) in args.args.iter().enumerate() {
        if idx != 0 {
            out.push(' ');
        }
        out.push_str(&format!("{}", arg.to_string(interpreter)));
    }

    out.push('\n');
    Err(out)
}

//enum ShimSortKey {
//    Bytes(&[u8]),
//    Int(i32),
//    Float(f32),
//}

fn shim_list_sort(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    let key = unpacker.optional(b"key");
    unpacker.end()?;

    // Create a vector of (index, value, sort_key) tuples to maintain stability
    let mut items_with_keys: Vec<(usize, ShimValue, ShimValue)> = Vec::new();
    
    for idx in 0..lst.len() {
        let item = lst.get(&interpreter.mem, idx as isize)?;
        
        let sort_key = if let Some(key) = key {
            let mut args = ArgBundle::new();
            args.args.push(item);
            match key.call(interpreter, &mut args)? {
                CallResult::ReturnValue(val) => val,
                CallResult::PC(pc, captured_scope) => {
                    let mut new_env = Environment::with_scope(captured_scope);
                    interpreter.execute_bytecode_extended(
                        &mut (pc as usize),
                        args,
                        &mut new_env,
                    )?
                },
            }
        } else {
            item
        };
        
        items_with_keys.push((idx, item, sort_key));
    }
    
    // Perform stable sort by comparing sort keys
    items_with_keys.sort_by(|a, b| {
        let (idx_a, _, key_a) = a;
        let (idx_b, _, key_b) = b;
        
        // Try to compare the keys
        match compare_values(interpreter, key_a, key_b) {
            Ok(ordering) => ordering,
            Err(_) => {
                // If comparison fails, maintain original order (stability)
                idx_a.cmp(idx_b)
            }
        }
    });
    
    // Mutate the list in place
    let lst_mut = obj.list_mut(interpreter)?;
    for (idx, (_, item, _)) in items_with_keys.iter().enumerate() {
        lst_mut.set(&mut interpreter.mem, idx as isize, *item)?;
    }
    
    Ok(ShimValue::None)
}

// Helper function to compare two ShimValues for sorting/ordering purposes.
// This function returns an Ordering to determine relative position in a sorted sequence.
// For equality checks, use ShimValue::equal_inner instead.
fn compare_values(interpreter: &mut Interpreter, a: &ShimValue, b: &ShimValue) -> Result<std::cmp::Ordering, String> {
    use std::cmp::Ordering;
    
    match (a, b) {
        (ShimValue::Integer(x), ShimValue::Integer(y)) => Ok(x.cmp(y)),
        (ShimValue::Float(x), ShimValue::Float(y)) => {
            // Handle NaN comparison by treating NaN as equal to itself
            if x.is_nan() && y.is_nan() {
                Ok(Ordering::Equal)
            } else if x.is_nan() {
                Ok(Ordering::Greater)
            } else if y.is_nan() {
                Ok(Ordering::Less)
            } else if x < y {
                Ok(Ordering::Less)
            } else if x > y {
                Ok(Ordering::Greater)
            } else {
                Ok(Ordering::Equal)
            }
        },
        (ShimValue::Integer(x), ShimValue::Float(y)) => {
            let x_f = *x as f32;
            if x_f < *y {
                Ok(Ordering::Less)
            } else if x_f > *y {
                Ok(Ordering::Greater)
            } else {
                Ok(Ordering::Equal)
            }
        },
        (ShimValue::Float(x), ShimValue::Integer(y)) => {
            let y_f = *y as f32;
            if *x < y_f {
                Ok(Ordering::Less)
            } else if *x > y_f {
                Ok(Ordering::Greater)
            } else {
                Ok(Ordering::Equal)
            }
        },
        (ShimValue::String(..), ShimValue::String(..)) => {
            let str_a = a.string(interpreter)?;
            let str_b = b.string(interpreter)?;
            Ok(str_a.cmp(&str_b))
        },
        (ShimValue::Bool(x), ShimValue::Bool(y)) => Ok(x.cmp(y)),
        (ShimValue::None, ShimValue::None) => Ok(Ordering::Equal),
        (ShimValue::List(_), ShimValue::List(_)) => {
            // Compare lists lexicographically
            let lst_a = a.list(interpreter)?;
            let lst_b = b.list(interpreter)?;
            
            let min_len = std::cmp::min(lst_a.len(), lst_b.len());
            for i in 0..min_len {
                let item_a = lst_a.get(&interpreter.mem, i as isize)?;
                let item_b = lst_b.get(&interpreter.mem, i as isize)?;
                match compare_values(interpreter, &item_a, &item_b)? {
                    Ordering::Equal => continue,
                    other => return Ok(other),
                }
            }
            Ok(lst_a.len().cmp(&lst_b.len()))
        },
        _ => Err(format!("Cannot compare {:?} and {:?}", a, b)),
    }
}

fn shim_list_filter(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    let key = unpacker.optional(b"key");
    unpacker.end()?;

    let new_lst_val = interpreter.mem.alloc_list();
    let new_lst = new_lst_val.list_mut(interpreter)?;

    for idx in 0..lst.len() {
        let input = lst.get(&interpreter.mem, idx as isize)?;
        let result = if let Some(key) = key {
            let mut args = ArgBundle::new();
            args.args.push(input);
            match key.call(interpreter, &mut args)? {
                CallResult::ReturnValue(val) => {
                    val
                },
                CallResult::PC(pc, captured_scope) => {
                    let mut new_env = Environment::with_scope(captured_scope);
                    let val = interpreter.execute_bytecode_extended(
                        &mut (pc as usize),
                        args,
                        // TODO: this doesn't even have print...
                        &mut new_env,
                    )?;
                    val
                },
            } 
        } else {
            input
        };
        if result.is_truthy(interpreter)? {
            new_lst.push(&mut interpreter.mem, input);
        }
    }

    Ok(new_lst_val)
}

fn shim_list_map(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    let key = unpacker.required(b"key")?;
    unpacker.end()?;

    let new_lst_val = interpreter.mem.alloc_list();
    let new_lst = new_lst_val.list_mut(interpreter)?;

    for idx in 0..lst.len() {
        let input = lst.get(&interpreter.mem, idx as isize)?;
        let mut args = ArgBundle::new();
        args.args.push(input);
        let output = match key.call(interpreter, &mut args)? {
            CallResult::ReturnValue(val) => {
                val
            },
            CallResult::PC(pc, captured_scope) => {
                let mut new_env = Environment::with_scope(captured_scope);
                let val = interpreter.execute_bytecode_extended(
                    &mut (pc as usize),
                    args,
                    // TODO: this doesn't even have print...
                    &mut new_env,
                )?;
                val
            },
        };
        new_lst.push(&mut interpreter.mem, output);
    }

    Ok(new_lst_val)
}

fn shim_list_len(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    unpacker.end()?;

    Ok(ShimValue::Integer(lst.len() as i32))
}

fn shim_list_append(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list_mut(interpreter)?;
    let item = unpacker.required(b"item")?;
    unpacker.end()?;

    lst.push(&mut interpreter.mem, item);

    Ok(ShimValue::None)
}

fn shim_list_iter(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter.mem.alloc_native(ListIterator {lst: obj, idx: 0}))
}

fn shim_list_clear(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list_mut(interpreter)?;
    unpacker.end()?;

    lst.len = 0.into();
    
    Ok(ShimValue::None)
}

fn shim_list_extend(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let iterable = unpacker.required(b"iterable")?;
    unpacker.end()?;

    // Get the iterator for the iterable
    let mut iter_args = ArgBundle::new();
    let iterator = iterable.get_attr(interpreter, b"iter")?.call(interpreter, &mut iter_args)?;
    let iterator = match iterator {
        CallResult::ReturnValue(val) => val,
        CallResult::PC(pc, captured_scope) => {
            let mut new_env = Environment::with_scope(captured_scope);
            interpreter.execute_bytecode_extended(
                &mut (pc as usize),
                iter_args,
                &mut new_env,
            )?
        },
    };

    // Get the next method
    let next_method = iterator.get_attr(interpreter, b"next")?;

    // Iterate and append each item
    loop {
        let mut next_args = ArgBundle::new();
        
        let result = match next_method.call(interpreter, &mut next_args)? {
            CallResult::ReturnValue(val) => val,
            CallResult::PC(pc, captured_scope) => {
                let mut new_env = Environment::with_scope(captured_scope);
                interpreter.execute_bytecode_extended(
                    &mut (pc as usize),
                    next_args,
                    &mut new_env,
                )?
            },
        };

        // Break if we get None (end of iteration)
        if result.is_none() {
            break;
        }

        // Append the item to the list
        let lst = obj.list_mut(interpreter)?;
        lst.push(&mut interpreter.mem, result);
    }

    Ok(ShimValue::None)
}

fn shim_list_index(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    let value = unpacker.required(b"value")?;
    let default = unpacker.optional(b"default");
    unpacker.end()?;

    for idx in 0..lst.len() {
        let item = lst.get(&interpreter.mem, idx as isize)?;
        if item.equal_inner(interpreter, &value)? {
            return Ok(ShimValue::Integer(idx as i32));
        }
    }

    Ok(default.unwrap_or(ShimValue::None))
}

fn shim_list_insert(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let index = unpacker.required(b"index")?;
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    let idx = index.integer()? as isize;

    let lst = obj.list_mut(interpreter)?;
    let len = lst.len();

    // Handle negative and out-of-bounds indices like Python
    let insert_idx = if idx < 0 {
        // Negative indices count from the end
        (len as isize + idx).max(0) as usize
    } else if idx as usize > len {
        // Positive indices beyond length append at the end
        len
    } else {
        idx as usize
    };

    // Add a new element at the end (this will resize if needed)
    lst.push(&mut interpreter.mem, ShimValue::None);

    // Shift elements to make room
    for i in (insert_idx..len).rev() {
        let val = lst.get(&interpreter.mem, i as isize)?;
        lst.set(&mut interpreter.mem, (i + 1) as isize, val)?;
    }

    // Insert the value
    lst.set(&mut interpreter.mem, insert_idx as isize, value)?;

    Ok(ShimValue::None)
}

fn shim_list_pop(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let index = unpacker.optional(b"index");
    let default = unpacker.optional(b"default");
    unpacker.end()?;

    let lst = obj.list_mut(interpreter)?;
    
    if lst.is_empty() {
        return Ok(default.unwrap_or(ShimValue::None));
    }

    // Determine which index to pop
    let pop_idx = if let Some(idx_val) = index {
        let idx = idx_val.integer()? as isize;
        lst.wrap_idx(idx)?
    } else {
        // Default to last element
        lst.len() - 1
    };

    // Get the value at the index
    let value = lst.get(&interpreter.mem, pop_idx as isize)?;

    // Shift elements after pop_idx to the left
    for i in pop_idx..(lst.len() - 1) {
        let next_val = lst.get(&interpreter.mem, (i + 1) as isize)?;
        lst.set(&mut interpreter.mem, i as isize, next_val)?;
    }

    // Decrease the length
    lst.len = (lst.len() - 1).into();

    Ok(value)
}

fn shim_list_sorted(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    let key = unpacker.optional(b"key");
    unpacker.end()?;

    // Create a new list with the same elements
    let new_lst_val = interpreter.mem.alloc_list();
    let new_lst = new_lst_val.list_mut(interpreter)?;
    
    for idx in 0..lst.len() {
        let item = lst.get(&interpreter.mem, idx as isize)?;
        new_lst.push(&mut interpreter.mem, item);
    }

    // Sort the new list using the existing sort logic
    let mut sort_args = ArgBundle::new();
    sort_args.args.push(new_lst_val);
    if let Some(k) = key {
        sort_args.kwargs.push((b"key".to_vec(), k));
    }
    shim_list_sort(interpreter, &sort_args)?;

    Ok(new_lst_val)
}

fn shim_list_reverse(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list_mut(interpreter)?;
    unpacker.end()?;

    let len = lst.len();
    for i in 0..(len / 2) {
        let left = lst.get(&interpreter.mem, i as isize)?;
        let right = lst.get(&interpreter.mem, (len - 1 - i) as isize)?;
        lst.set(&mut interpreter.mem, i as isize, right)?;
        lst.set(&mut interpreter.mem, (len - 1 - i) as isize, left)?;
    }

    Ok(ShimValue::None)
}

fn shim_list_reversed(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    unpacker.end()?;

    // Create a new list with reversed elements
    let new_lst_val = interpreter.mem.alloc_list();
    let new_lst = new_lst_val.list_mut(interpreter)?;
    
    for idx in (0..lst.len()).rev() {
        let item = lst.get(&interpreter.mem, idx as isize)?;
        new_lst.push(&mut interpreter.mem, item);
    }

    Ok(new_lst_val)
}

fn shim_dict_iter(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter.mem.alloc_native(DictKeysIterator {dict: obj, idx: 0}))
}

fn shim_dict_keys(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter.mem.alloc_native(DictKeysIterator {dict: obj, idx: 0}))
}

fn shim_dict_values(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter.mem.alloc_native(DictValuesIterator {dict: obj, idx: 0}))
}

fn shim_dict_items(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter.mem.alloc_native(DictItemsIterator {dict: obj, idx: 0}))
}

fn shim_dict_pop(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    let key = unpacker.required(b"key")?;
    let default = unpacker.optional(b"default");
    unpacker.end()?;

    dict.pop(interpreter, key, default)
}

fn shim_dict_index_set(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    let key = unpacker.required(b"key")?;
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    dict.set(interpreter, key, value)?;

    Ok(ShimValue::None)
}

fn shim_dict_index_get(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    let key = unpacker.required(b"key")?;
    unpacker.end()?;

    dict.get(interpreter, key)
}

fn shim_dict_index_has(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    let key = unpacker.required(b"key")?;
    unpacker.end()?;

    Ok(ShimValue::Bool(dict.get(interpreter, key).is_ok()))
}

fn shim_dict_len(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    unpacker.end()?;

    Ok(ShimValue::Integer(dict.len() as i32))
}

fn shim_dict_shrink_to_fit(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    unpacker.end()?;

    dict.shrink_to_fit(interpreter);
    Ok(ShimValue::None)
}

fn shim_str_len(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?;
    unpacker.end()?;

    Ok(ShimValue::Integer(s.len() as i32))
}

fn get_type_name(value: &ShimValue) -> &'static str {
    match value {
        ShimValue::Uninitialized => "uninitialized",
        ShimValue::Unit => "unit",
        ShimValue::None => "none",
        ShimValue::Integer(_) => "int",
        ShimValue::Float(_) => "float",
        ShimValue::Bool(_) => "bool",
        ShimValue::Fn(_) => "function",
        ShimValue::BoundMethod(_, _) => "bound method",
        ShimValue::BoundNativeMethod(_) => "bound native method",
        ShimValue::NativeFn(_) => "native function",
        ShimValue::String(..) => "string",
        ShimValue::List(_) => "list",
        ShimValue::Dict(_) => "dict",
        ShimValue::StructDef(_) => "struct definition",
        ShimValue::Struct(_) => "struct",
        ShimValue::Native(_) => "native object",
        ShimValue::Environment(_) => "environment",
    }
}

fn trim_bytes(s: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = s.len();
    
    // Trim from start
    while start < end && s[start].is_ascii_whitespace() {
        start += 1;
    }
    
    // Trim from end
    while end > start && s[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    
    &s[start..end]
}

fn parse_string_to<T: std::str::FromStr>(
    s: &[u8],
    type_name: &str,
) -> Result<T, String> {
    let trimmed = trim_bytes(s);
    unsafe {
        std::str::from_utf8_unchecked(trimmed).parse::<T>()
            .map_err(|_| {
                let string_repr = std::str::from_utf8(s).unwrap_or("<invalid utf8>");
                format!("Cannot convert string '{}' to {}", string_repr, type_name)
            })
    }
}

fn shim_str(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    let string_repr = value.to_string(interpreter);
    let bytes = string_repr.as_bytes();
    Ok(interpreter.mem.alloc_str(bytes))
}

fn shim_int(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    match value {
        ShimValue::Integer(i) => Ok(ShimValue::Integer(i)),
        ShimValue::Float(f) => Ok(ShimValue::Integer(f as i32)),
        ShimValue::Bool(b) => Ok(ShimValue::Integer(if b { 1 } else { 0 })),
        ShimValue::String(..) => {
            let s = value.string(interpreter)?;
            parse_string_to::<i32>(s, "int").map(ShimValue::Integer)
        },
        _ => Err(format!("Cannot convert {} to int", get_type_name(&value)))
    }
}

fn shim_float(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    match value {
        ShimValue::Integer(i) => Ok(ShimValue::Float(i as f32)),
        ShimValue::Float(f) => Ok(ShimValue::Float(f)),
        ShimValue::Bool(b) => Ok(ShimValue::Float(if b { 1.0 } else { 0.0 })),
        ShimValue::String(..) => {
            let s = value.string(interpreter)?;
            parse_string_to::<f32>(s, "float").map(ShimValue::Float)
        },
        _ => Err(format!("Cannot convert {} to float", get_type_name(&value)))
    }
}

fn shim_try_int(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    let result = match value {
        ShimValue::Integer(i) => Some(ShimValue::Integer(i)),
        ShimValue::Float(f) => Some(ShimValue::Integer(f as i32)),
        ShimValue::Bool(b) => Some(ShimValue::Integer(if b { 1 } else { 0 })),
        ShimValue::String(..) => {
            let s = value.string(interpreter)?;
            parse_string_to::<i32>(s, "int")
                .map(ShimValue::Integer)
                .ok()
        },
        _ => None
    };

    Ok(result.unwrap_or(ShimValue::None))
}

fn shim_try_float(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    let result = match value {
        ShimValue::Integer(i) => Some(ShimValue::Float(i as f32)),
        ShimValue::Float(f) => Some(ShimValue::Float(f)),
        ShimValue::Bool(b) => Some(ShimValue::Float(if b { 1.0 } else { 0.0 })),
        ShimValue::String(..) => {
            let s = value.string(interpreter)?;
            parse_string_to::<f32>(s, "float")
                .map(ShimValue::Float)
                .ok()
        },
        _ => None
    };

    Ok(result.unwrap_or(ShimValue::None))
}

#[derive(Debug)]
pub struct ArgBundle {
    args: Vec<ShimValue>,
    kwargs: Vec<(Ident, ShimValue)>,
}

impl ArgBundle {
    pub fn new() -> Self {
        Self {
            args: Vec::new(),
            kwargs: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.args.len() + self.kwargs.len()
    }

    fn clear(&mut self) {
        self.args.clear();
        self.kwargs.clear();
    }
}

struct ArgUnpacker<'a> {
    bundle: &'a ArgBundle,
    pos: usize,
    kwargs_consumed: usize,
}

impl<'a> ArgUnpacker<'a> {
    fn new(bundle: &'a ArgBundle) -> Self {
        Self { bundle, pos: 0, kwargs_consumed: 0 }
    }

    fn required(&mut self, name: &[u8]) -> Result<ShimValue, String> {
        self.optional(name).ok_or_else(|| format!("Missing required argument: '{}'", debug_u8s(name)))
    }

    fn optional(&mut self, name: &[u8]) -> Option<ShimValue> {
        for (ident, arg) in self.bundle.kwargs.iter() {
            if ident == name {
                self.kwargs_consumed += 1;
                return Some(*arg);
            }
        }
        // Return next positional argument
        match self.bundle.args.get(self.pos) {
            Some(val) => {
                self.pos += 1;
                Some(*val)
            },
            None => None,
        }
    }

    fn end(&self) -> Result<(), String> {
        let consumed = self.pos + self.kwargs_consumed;
        if self.bundle.len() != consumed {
            Err(format!("Got {} arguments, but only used {}", self.bundle.len(), consumed))
        } else {
            Ok(())
        }
    }
}

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

pub fn fnv1a_hash(key: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;

    for &byte in key {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    hash
}

macro_rules! numeric_op {
    ($lhs:tt $op:tt $rhs:expr) => {
        match ($lhs, $rhs) {
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Integer(*a $op *b)),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Float(*a $op *b)),
            (ShimValue::Integer(a), ShimValue::Float(b)) => Ok(ShimValue::Float((*a as f32) $op *b)),
            (ShimValue::Float(a), ShimValue::Integer(b)) => Ok(ShimValue::Float(*a $op (*b as f32))),
            (a, b) => Err(format!(
                "Operation '{}' not supported between {:?} and {:?}",
                stringify!($op), a, b
            )),
        }
    };
}

impl ShimValue {
    fn is_uninitialized(&self) -> bool {
        if let ShimValue::Uninitialized = self {
            true
        } else {
            false
        }
    }

    fn is_none(&self) -> bool {
        matches!(self, ShimValue::None)
    }

    fn hash(&self, interpreter: &mut Interpreter) -> Result<u32, String> {
        let hashcode: u64 = match self {
            ShimValue::Integer(i) => fnv1a_hash(&i.to_be_bytes()),
            ShimValue::Float(f) => fnv1a_hash(&f.to_be_bytes()),
            ShimValue::String(..) => {
                fnv1a_hash(&self.string(interpreter).unwrap().to_vec())
            },
            // We might want to salt these to reduce collisions with other type,
            // but I expect there is a fairly trivial difference in performance
            // and would imply heterogenous dicts.
            ShimValue::None => fnv1a_hash(&[0x00]),
            ShimValue::Bool(false) => fnv1a_hash(&[0x00]),
            ShimValue::Bool(true) => fnv1a_hash(&[0x01]),
            _ => return Err(format!("Can't hash {:?}", self))
        };

        Ok(hashcode as u32)
    }

    fn as_native<T: ShimNative>(&self, interpreter: &mut Interpreter) -> Result<&mut T, String> {
        match self {
            ShimValue::Native(position) => unsafe {
                let boxobj: &mut Box<dyn ShimNative> =
                    std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)]);

                let mutboxobj = boxobj.as_any_mut();
                let name = type_name_of_val(mutboxobj);
                match mutboxobj.downcast_mut::<T>() {
                    Some(obj) => Ok(obj),
                    _ => Err(format!("Can't get {} as {}", name, type_name::<T>()))
                }
            },
            _ => Err(format!("Can't try_into non-native {:?}", self))
        }
    }

    fn call(
        &self,
        interpreter: &mut Interpreter,
        args: &mut ArgBundle,
    ) -> Result<CallResult, String> {
        match self {
            ShimValue::None => Err(format!("Can't call None as a function")),
            ShimValue::Fn(fn_pos) => {
                let shim_fn: &ShimFn = unsafe { interpreter.mem.get(*fn_pos) };
                Ok(CallResult::PC(shim_fn.pc, shim_fn.captured_scope))
            }
            ShimValue::BoundMethod(pos, fn_pos) => {
                // push struct pos to start of arg list then return the pc of the method
                args.args.insert(0, ShimValue::Struct(*pos));
                let shim_fn: &ShimFn = unsafe { interpreter.mem.get(*fn_pos) };
                Ok(CallResult::PC(shim_fn.pc, shim_fn.captured_scope))
            }
            ShimValue::BoundNativeMethod(pos) => {
                let obj: &ShimValue = unsafe { interpreter.mem.get(*pos) };
                let native_fn: &NativeFn = unsafe { interpreter.mem.get(*pos + 1) };

                args.args.insert(0, *obj);
                Ok(CallResult::ReturnValue(native_fn(interpreter, args)?))
            }
            ShimValue::StructDef(struct_def_pos) => {
                let struct_def: &StructDef = unsafe { interpreter.mem.get(*struct_def_pos) };
                if struct_def.member_count as usize != args.len() || !args.kwargs.is_empty()  {
                    // Call the internal __init__ to handle default/kw arguments
                    // If we're not using defaults we could handle kw arguments here,
                    // but for now it simplifies things to push all the special cases to __init__
                    if let Some(StructAttribute::MethodDef(fn_pos)) = struct_def.find(b"__init__") {
                        let shim_fn: &ShimFn = unsafe { interpreter.mem.get(fn_pos) };
                        return Ok(CallResult::PC(shim_fn.pc, shim_fn.captured_scope));
                    } else {
                        return Err(format!("INTERNAL: no __init__ on StructDef"));
                    }
                }

                // Allocate space for each member, plus the header
                let word_count = Word((struct_def.member_count as u32 + 1).into());
                let new_pos = alloc!(
                    interpreter.mem,
                    word_count,
                    "Struct instantiation"
                );

                // The first word points to the StructDef
                interpreter.mem.mem[usize::from(new_pos.0)] = u64::from(struct_def_pos.0);

                // The remaining words get copies of the arguments to the initializer
                for (idx, arg) in args.args.iter().enumerate() {
                    interpreter.mem.mem[usize::from(new_pos.0) + 1 + idx] = arg.to_u64();
                }

                Ok(CallResult::ReturnValue(ShimValue::Struct(new_pos)))
            }
            ShimValue::NativeFn(pos) => {
                let native_fn: &NativeFn = unsafe { interpreter.mem.get(*pos) };
                Ok(CallResult::ReturnValue(native_fn(interpreter, args)?))
            }
            other => Err(format!(
                "Can't call value {:?} as a function",
                other.to_string(interpreter)
            )),
        }
    }

    fn dict_mut(&self, interpreter: &mut Interpreter) -> Result<&mut ShimDict, String> {
        match self {
            ShimValue::Dict(position) => {
                let dict: &mut ShimDict = unsafe {
                    std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)])
                };
                Ok(dict)
            },
            _ => {
                Err(format!("Not a dict"))
            }
        }
    }

    fn dict(&self, interpreter: &Interpreter) -> Result<&ShimDict, String> {
        match self {
            ShimValue::Dict(position) => {
                let dict: &ShimDict = unsafe {
                    std::mem::transmute(&interpreter.mem.mem[usize::from(position.0)])
                };
                Ok(dict)
            },
            _ => {
                Err(format!("Not a dict"))
            }
        }
    }

    fn list_mut(&self, interpreter: &mut Interpreter) -> Result<&mut ShimList, String> {
        match self {
            ShimValue::List(position) => {
                unsafe {
                    Ok(std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)]))
                }
            },
            _ => {
                Err(format!("Not a list"))
            }
        }
    }

    fn list(&self, interpreter: &Interpreter) -> Result<&ShimList, String> {
        match self {
            ShimValue::List(position) => {
                unsafe {
                    Ok(std::mem::transmute(&interpreter.mem.mem[usize::from(position.0)]))
                }
            },
            _ => {
                Err(format!("Not a list"))
            }
        }
    }

    fn native(&self, interpreter: &mut Interpreter) -> Result<&mut Box<dyn ShimNative>, String> {
        match self {
            ShimValue::Native(position) => unsafe {
                let ptr: *mut Box<dyn ShimNative> =
                    std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)]);
                Ok(&mut *ptr)
            },
            _ => {
                Err(format!("Not a native"))
            }
        }
    }

    fn expect_string(&self, interpreter: &Interpreter) -> &[u8] {
        self.string(interpreter).unwrap()
    }

    fn string(&self, interpreter: &Interpreter) -> Result<&[u8], String> {
        match self {
            ShimValue::String(len, offset, position) => {
                let len = *len as usize;
                let offset = *offset as usize;
                let position_usize = usize::from(*position);
                let total_len: usize = (offset + len).div_ceil(8);

                let bytes: &[u8] = unsafe {
                    let u64_slice = &interpreter.mem.mem[
                        position_usize..
                        (position_usize+total_len)
                    ];
                    std::slice::from_raw_parts(
                        (u64_slice.as_ptr() as *const u8).add(offset),
                        len,
                    )
                };
                Ok(bytes)
            },
            _ => {
                Err(format!("Not a string"))
            }
        }
    }

    fn integer(&self) -> Result<i32, String> {
        match self {
            ShimValue::Integer(i) => Ok(*i),
            _ => Err(format!("Not an integer")),
        }
    }

    fn index(&self, interpreter: &mut Interpreter, index: &ShimValue) -> Result<ShimValue, String> {
        match (self, index) {
            (ShimValue::String(..), ShimValue::Integer(index)) => {
                let index = *index as isize;

                let val = self.string(interpreter)?;

                let len = val.len() as isize;
                let index: isize = if index < -len || index >= len {
                    return Err(format!("Index {} is out of bounds", index));
                } else if index < 0 {
                    len + index as isize
                } else {
                    index as isize
                };

                let b: u8 = val[index as usize];

                Ok(interpreter.mem.alloc_str(&[b]))
            },
            (ShimValue::List(position), ShimValue::Integer(idx)) => {
                unsafe {
                    let lst: &ShimList =
                        std::mem::transmute(&interpreter.mem.mem[usize::from(position.0)]);
                    lst.get(&interpreter.mem, *idx as isize)
                }
            },
            (ShimValue::Dict(_), some_key) => {
                let dict = self.dict_mut(interpreter)?;

                dict.get(interpreter, *some_key)
            }
            (a, b) => Err(format!("Can't index {:?} with {:?}", a, b)),
        }
    }

    fn set_index(
        &self,
        interpreter: &mut Interpreter,
        index: &ShimValue,
        value: &ShimValue,
    ) -> Result<(), String> {
        match (self, index) {
            (ShimValue::List(position), ShimValue::Integer(index)) => {
                let index = *index as usize;
                let list: &mut ShimList = unsafe {
                    std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)])
                };
                list.set(&mut interpreter.mem, index as isize, *value)?;
                Ok(())
            }
            (ShimValue::Dict(position), index) => {
                let dict: &mut ShimDict = unsafe {
                    std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)])
                };

                dict.set(interpreter, *index, *value)
            }
            (a, b) => Err(format!("Can't set index {:?} with {:?}", a, b)),
        }
    }

    fn to_shimvalue_string(&self, interpreter: &mut Interpreter) -> ShimValue {
        let s = self.to_string(interpreter);
        interpreter.mem.alloc_str(&s.into_bytes())
    }

    pub fn to_string(&self, interpreter: &mut Interpreter) -> String {
        match self {
            ShimValue::Uninitialized => format!("Uninitialized"),
            ShimValue::None => "None".to_string(),
            ShimValue::Integer(i) => i.to_string(),
            ShimValue::Float(f) => format_float(*f),
            ShimValue::Bool(false) => "false".to_string(),
            ShimValue::Bool(true) => "true".to_string(),
            ShimValue::String(..) => {
                String::from_utf8(self.string(interpreter).unwrap().to_vec()).expect("valid utf-8 string stored")
            },
            ShimValue::List(_) => {
                let lst = self.list(interpreter).unwrap();

                let mut out = "[".to_string();
                for idx in 0..lst.len() {
                    if idx != 0 {
                        out.push_str(",");
                        out.push_str(" ");
                    }
                    let item = lst.get(&interpreter.mem, idx as isize).unwrap();
                    out.push_str(&item.to_string(interpreter));
                }
                out.push_str("]");

                out
            },
            ShimValue::Native(_) => {
                self.native(interpreter).unwrap().to_string(interpreter)
            }
            ShimValue::Struct(pos) => {
                unsafe {
                    let def_pos: u64 = *interpreter.mem.get(*pos);
                    let def_pos: Word = Word((def_pos as u32).into());
                    let def: &StructDef = interpreter.mem.get(def_pos);
                    
                    // Get the struct name
                    let struct_name = debug_u8s(&def.name).to_string();
                    
                    // Collect member names and values first to avoid borrowing issues
                    let mut members: Vec<(String, ShimValue)> = Vec::new();
                    for (attr, loc) in def.lookup.iter() {
                        // Only collect member variables, not methods
                        if let StructAttribute::MemberInstanceOffset(offset) = loc {
                            let attr_name = debug_u8s(attr).to_string();
                            let val: ShimValue = *interpreter.mem.get(*pos + *offset as u32 + 1);
                            members.push((attr_name, val));
                        }
                    }
                    
                    // Build output like "Point(x=2.0, y=3.0)"
                    let mut out = struct_name;
                    out.push('(');
                    
                    for (idx, (attr_name, val)) in members.iter().enumerate() {
                        if idx != 0 {
                            out.push_str(", ");
                        }
                        out.push_str(attr_name);
                        out.push('=');
                        out.push_str(&val.to_string(interpreter));
                    }
                    
                    out.push(')');
                    out
                }
            }
            value => format!("{:?}", value),
        }
    }

    fn is_truthy(&self, interpreter: &mut Interpreter) -> Result<bool, String> {
        match self {
            ShimValue::None => Ok(false),
            ShimValue::Integer(i) => Ok(*i != 0),
            ShimValue::Float(f) => Ok(*f != 0.0),
            ShimValue::Bool(false) => Ok(false),
            ShimValue::Bool(true) => Ok(true),
            ShimValue::String(..) => {
                Ok(!self.expect_string(interpreter).is_empty())
            },
            ShimValue::List(_) => {
                Ok(!self.list(interpreter)?.is_empty())
            },
            _ => Ok(true),
        }
    }

    fn add(&self, interpreter: &mut Interpreter, other: &Self, pending_args: &mut ArgBundle) -> Result<CallResult, String> {
        match (self, other) {
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(CallResult::ReturnValue(ShimValue::Integer(*a + *b))),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(CallResult::ReturnValue(ShimValue::Float(*a + *b))),
            (ShimValue::Integer(a), ShimValue::Float(b)) => Ok(CallResult::ReturnValue(ShimValue::Float((*a as f32) + *b))),
            (ShimValue::Float(a), ShimValue::Integer(b)) => Ok(CallResult::ReturnValue(ShimValue::Float(*a + (*b as f32)))),
            (a @ ShimValue::String(..), b @ ShimValue::String(..)) => {
                let a = a.string(interpreter)?;
                let b = b.string(interpreter)?;

                let c = interpreter.mem.alloc_str(
                    &format!("{}{}",
                        unsafe { std::str::from_utf8_unchecked(a) },
                        unsafe { std::str::from_utf8_unchecked(b) },
                    ).into_bytes()
                );

                Ok(CallResult::ReturnValue(c))
            }
            (ShimValue::Struct(_), b) => {
                // TODO: why do we need to take in `pending_args` when we could
                // construct a new ArgBundle?
                pending_args.args.clear();
                pending_args.args.push(*b);
                self.get_attr(interpreter, b"add")?.call(interpreter, pending_args)
            },
            (a, b) => Err(format!(
                "Operation '+' not supported between {:?} and {:?}",
                a, b
            )),
        }
    }

    fn sub(&self, other: &Self) -> Result<ShimValue, String> {
        numeric_op!(self - other)
    }

    fn equal_inner(&self, interpreter: &mut Interpreter, other: &Self) -> Result<bool, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => Ok(a == b),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(a == b),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(a == b),
            (a @ ShimValue::String(..), b @ ShimValue::String(..)) => {
                let a = a.string(interpreter)?;
                let b = b.string(interpreter)?;
                Ok(a == b)
            }
            (ShimValue::None, ShimValue::None) => Ok(true),
            (a @ ShimValue::List(_), b @ ShimValue::List(_)) => {
                let a = a.list(interpreter)?;
                let b = b.list(interpreter)?;
                if a.len() != b.len() {
                    return Ok(false)
                }
                for idx in 0..a.len() {
                    let item_a = a.get(&interpreter.mem, idx as isize)?;
                    let item_b = b.get(&interpreter.mem, idx as isize)?;
                    if !item_a.equal_inner(interpreter, &item_b)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            },
            _ => Ok(false),
        }
    }

    fn equal(&self, interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        Ok(ShimValue::Bool(self.equal_inner(interpreter, other)?))
    }

    fn not_equal(&self, _interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => Ok(ShimValue::Bool(a != b)),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Bool(a != b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(a != b)),
            (ShimValue::None, ShimValue::None) => Ok(ShimValue::Bool(true)),
            _ => Ok(ShimValue::Bool(true)),
        }
    }

    fn mul(&self, _interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        numeric_op!(self * other)
    }

    fn div(&self, _interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        // NOTE: All division is floating point division
        match (self, other) {
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Float(a / b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Integer(a / b)),
            (ShimValue::Integer(a), ShimValue::Float(b)) => Ok(ShimValue::Float((*a as f32) / *b)),
            (ShimValue::Float(a), ShimValue::Integer(b)) => Ok(ShimValue::Float(*a / (*b as f32))),
            (a, b) => Err(format!("Can't Divide {:?} and {:?}", a, b)),
        }
    }

    fn modulus(&self, _interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        // NOTE: All division is floating point division
        match (self, other) {
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Float(a % b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Integer(a % b)),
            (ShimValue::Integer(a), ShimValue::Float(b)) => Ok(ShimValue::Float((*a as f32) % *b)),
            (ShimValue::Float(a), ShimValue::Integer(b)) => Ok(ShimValue::Float(*a % (*b as f32))),
            (a, b) => Err(format!("Can't Divide {:?} and {:?}", a, b)),
        }
    }

    fn gt(&self, interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => {
                Ok(ShimValue::Bool(*a == true && *b == false))
            }
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Bool(a > b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(a > b)),
            (ShimValue::Integer(a), ShimValue::Float(b)) => Ok(ShimValue::Bool((*a as f32) > *b)),
            (ShimValue::Float(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(*a > (*b as f32))),
            (ShimValue::None, ShimValue::None) => Ok(ShimValue::Bool(false)),
            (ShimValue::String(..), ShimValue::String(..)) => {
                Ok(ShimValue::Bool(self.string(interpreter)? > other.string(interpreter)?))
            },
            (ShimValue::List(_), ShimValue::List(_)) => {
                match compare_values(interpreter, self, other) {
                    Ok(std::cmp::Ordering::Greater) => Ok(ShimValue::Bool(true)),
                    Ok(_) => Ok(ShimValue::Bool(false)),
                    Err(e) => Err(e),
                }
            },
            (a, b) => Err(format!("Can't GT {:?} and {:?}", a, b)),
        }
    }

    fn gte(&self, interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => Ok(ShimValue::Bool(a == b)),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Bool(a >= b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(a >= b)),
            (ShimValue::Integer(a), ShimValue::Float(b)) => Ok(ShimValue::Bool((*a as f32) >= *b)),
            (ShimValue::Float(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(*a >= (*b as f32))),
            (ShimValue::None, ShimValue::None) => Ok(ShimValue::Bool(true)),
            (ShimValue::String(..), ShimValue::String(..)) => {
                Ok(ShimValue::Bool(self.string(interpreter)? >= other.string(interpreter)?))
            },
            (ShimValue::List(_), ShimValue::List(_)) => {
                match compare_values(interpreter, self, other) {
                    Ok(std::cmp::Ordering::Greater) | Ok(std::cmp::Ordering::Equal) => Ok(ShimValue::Bool(true)),
                    Ok(std::cmp::Ordering::Less) => Ok(ShimValue::Bool(false)),
                    Err(e) => Err(e),
                }
            },
            (a, b) => Err(format!("Can't GTE {:?} and {:?}", a, b)),
        }
    }

    fn lt(&self, interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => {
                Ok(ShimValue::Bool(*a == false && *b == true))
            }
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Bool(a < b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(a < b)),
            (ShimValue::Integer(a), ShimValue::Float(b)) => Ok(ShimValue::Bool((*a as f32) < *b)),
            (ShimValue::Float(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(*a < (*b as f32))),
            (ShimValue::String(..), ShimValue::String(..)) => {
                Ok(ShimValue::Bool(self.string(interpreter)? < other.string(interpreter)?))
            },
            (ShimValue::None, ShimValue::None) => Ok(ShimValue::Bool(false)),
            (ShimValue::List(_), ShimValue::List(_)) => {
                match compare_values(interpreter, self, other) {
                    Ok(std::cmp::Ordering::Less) => Ok(ShimValue::Bool(true)),
                    Ok(_) => Ok(ShimValue::Bool(false)),
                    Err(e) => Err(e),
                }
            },
            (a, b) => Err(format!("Can't LT {:?} and {:?}", a, b)),
        }
    }

    fn lte(&self, interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => Ok(ShimValue::Bool(a == b)),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Bool(a <= b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(a <= b)),
            (ShimValue::Integer(a), ShimValue::Float(b)) => Ok(ShimValue::Bool((*a as f32) <= *b)),
            (ShimValue::Float(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(*a <= (*b as f32))),
            (ShimValue::None, ShimValue::None) => Ok(ShimValue::Bool(true)),
            (ShimValue::String(..), ShimValue::String(..)) => {
                Ok(ShimValue::Bool(self.string(interpreter)? <= other.string(interpreter)?))
            },
            (ShimValue::List(_), ShimValue::List(_)) => {
                match compare_values(interpreter, self, other) {
                    Ok(std::cmp::Ordering::Less) | Ok(std::cmp::Ordering::Equal) => Ok(ShimValue::Bool(true)),
                    Ok(std::cmp::Ordering::Greater) => Ok(ShimValue::Bool(false)),
                    Err(e) => Err(e),
                }
            },
            (a, b) => Err(format!("Can't LTE {:?} and {:?}", a, b)),
        }
    }

    fn contains(
        &self,
        interpreter: &mut Interpreter,
        some_key: &Self,
    ) -> Result<ShimValue, String> {
        match self {
            ShimValue::Dict(position) => {
                let dict: &mut ShimDict = unsafe {
                    let ptr: &mut ShimDict =
                        std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)]);
                    ptr
                };

                if let Ok(_) = dict.get(interpreter, *some_key) {
                    return Ok(ShimValue::Bool(true));
                } else {
                    return Ok(ShimValue::Bool(false));
                }
            }
            _ => Err(format!("Can't `in` {:?} and {:?}", self, some_key)),
        }
    }

    fn not(&self, interpreter: &mut Interpreter) -> Result<ShimValue, String> {
        match self {
            ShimValue::Bool(a) => Ok(ShimValue::Bool(!a)),
            ShimValue::Float(a) => Ok(ShimValue::Bool(*a == 0.0)),
            ShimValue::Integer(a) => Ok(ShimValue::Bool(*a == 0)),
            ShimValue::None => Ok(ShimValue::Bool(true)),
            ShimValue::List(_) => Ok(ShimValue::Bool(!self.is_truthy(interpreter)?)),
            _ => Ok(ShimValue::Bool(false)),
        }
    }

    fn neg(&self, _interpreter: &mut Interpreter) -> Result<ShimValue, String> {
        match self {
            ShimValue::Float(a) => Ok(ShimValue::Float(-a)),
            ShimValue::Integer(a) => Ok(ShimValue::Integer(-a)),
            _ => Err(format!("Can't Negate {:?}", self)),
        }
    }

    fn get_attr(&self, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        match self {
            ShimValue::Struct(pos) => {
                // Handle __type__ special attribute
                if ident == b"__type__" {
                    unsafe {
                        let def_pos: u64 = *interpreter.mem.get(*pos);
                        let def_pos: Word = Word((def_pos as u32).into());
                        return Ok(ShimValue::StructDef(def_pos));
                    }
                }
                
                unsafe {
                    let def_pos: u64 = *interpreter.mem.get(*pos);
                    let def_pos: Word = Word((def_pos as u32).into());
                    let def: &StructDef = interpreter.mem.get(def_pos);
                    for (attr, loc) in def.lookup.iter() {
                        if ident == attr {
                            return match loc {
                                StructAttribute::MemberInstanceOffset(offset) => {
                                    Ok(*interpreter.mem.get(*pos + *offset as u32 + 1))
                                }
                                StructAttribute::MethodDef(fn_pos) => {
                                    // Return the bound method with the pre-allocated function
                                    Ok(ShimValue::BoundMethod(*pos, *fn_pos))
                                }
                            };
                        }
                    }
                }
                Err(format!("Ident {:?} not found for {:?}", debug_u8s(ident), self))
            }
            ShimValue::StructDef(def_pos) => {
                // Handle __name__ special attribute
                if ident == b"__name__" {
                    unsafe {
                        let def: &StructDef = interpreter.mem.get(*def_pos);
                        let name = def.name.clone();
                        return Ok(interpreter.mem.alloc_str(&name));
                    }
                }
                
                unsafe {
                    let def: &StructDef = interpreter.mem.get(*def_pos);
                    for (attr, loc) in def.lookup.iter() {
                        if ident == attr {
                            return match loc {
                                StructAttribute::MemberInstanceOffset(_) => Err(format!(
                                    "Can't access member {:?} on StructDef {:?}",
                                    ident, self
                                )),
                                StructAttribute::MethodDef(fn_pos) => {
                                    // Return the pre-allocated method function
                                    Ok(ShimValue::Fn(*fn_pos))
                                }
                            };
                        }
                    }
                }
                Err(format!("Ident {:?} not found for {:?}", debug_u8s(ident), self))
            }
            ShimValue::String(..) => {
                let func = match ident {
                    b"len" => shim_str_len,
                    _ => return Err(format!("No ident {:?} on str", debug_u8s(ident))),
                };
                Ok(interpreter.mem.alloc_bound_native_fn(self, func))
            }
            ShimValue::List(_) => {
                let func = match ident {
                    b"map" => shim_list_map,
                    b"filter" => shim_list_filter,
                    b"len" => shim_list_len,
                    b"iter" => shim_list_iter,
                    b"sort" => shim_list_sort,
                    b"append" => shim_list_append,
                    b"clear" => shim_list_clear,
                    b"extend" => shim_list_extend,
                    b"index" => shim_list_index,
                    b"insert" => shim_list_insert,
                    b"pop" => shim_list_pop,
                    b"sorted" => shim_list_sorted,
                    b"reverse" => shim_list_reverse,
                    b"reversed" => shim_list_reversed,
                    _ => return Err(format!("No ident {:?} on list", debug_u8s(ident))),
                };
                Ok(interpreter.mem.alloc_bound_native_fn(self, func))
            }
            ShimValue::Dict(_) => {
                let func = match ident {
                    b"set" => shim_dict_index_set,
                    b"get" => shim_dict_index_get,
                    b"has" => shim_dict_index_has,
                    b"len" => shim_dict_len,
                    b"pop" => shim_dict_pop,
                    b"iter" => shim_dict_iter,
                    b"keys" => shim_dict_keys,
                    b"values" => shim_dict_values,
                    b"items" => shim_dict_items,
                    b"shrink_to_fit" => shim_dict_shrink_to_fit,
                    _ => return Err(format!("No ident {:?} on dict", debug_u8s(ident))),
                };
                Ok(interpreter.mem.alloc_bound_native_fn(self, func))
            }
            ShimValue::Native(_) => {
                self.native(interpreter).unwrap().get_attr(self, interpreter, ident)
            }
            val => Err(format!("Ident {:?} not available on {:?}", debug_u8s(ident), val)),
        }
    }

    fn set_attr(
        &self,
        interpreter: &mut Interpreter,
        ident: &[u8],
        val: ShimValue,
    ) -> Result<(), String> {
        match self {
            ShimValue::Struct(pos) => {
                unsafe {
                    let def_pos: u64 = *interpreter.mem.get(*pos);
                    let def_pos: Word = Word((def_pos as u32).into());
                    let def: &StructDef = interpreter.mem.get(def_pos);
                    for (attr, loc) in def.lookup.iter() {
                        if ident == attr {
                            return match loc {
                                StructAttribute::MemberInstanceOffset(offset) => {
                                    let slot: &mut ShimValue =
                                        interpreter.mem.get_mut(*pos + *offset as u32 + 1);
                                    *slot = val;
                                    Ok(())
                                }
                                StructAttribute::MethodDef(_) => Err(format!(
                                    "Can't assign to struct method {:?} for {:?}",
                                    ident, self
                                )),
                            };
                        }
                    }
                }
                Err(format!("Ident {:?} not found for {:?}", ident, self))
            }
            ShimValue::Native(_) => {
                self.native(interpreter).unwrap().set_attr(interpreter, ident, val)
            }
            val => Err(format!("Ident {:?} not available on {:?}", ident, val)),
        }
    }

    fn to_u64(&self) -> u64 {
        unsafe {
            let mut tmp: u64 = 0;
            // Copy raw bytes of e into tmp
            std::ptr::copy_nonoverlapping(
                self as *const Self as *const u8,
                &mut tmp as *mut u64 as *mut u8,
                size_of::<Self>(),
            );
            tmp
        }
    }

    fn to_bytes(&self) -> [u8; 8] {
        unsafe { std::mem::transmute(*self) }
    }

    fn from_bytes(bytes: [u8; 8]) -> Self {
        unsafe { std::mem::transmute(bytes) }
    }

    unsafe fn from_u64(data: u64) -> Self {
        unsafe {
            let mut tmp: Self = std::mem::zeroed(); // Will be overwritten
            std::ptr::copy_nonoverlapping(
                &data as *const u64 as *const u8,
                &mut tmp as *mut Self as *mut u8,
                size_of::<Self>(),
            );
            tmp
        }
    }
}

#[repr(u8)]
enum ByteCode {
    NoOp,
    Pad0,
    Pad1,
    Pad2,
    Pad3,
    Pad4,
    Pad5,
    Pad6,
    Pad7,
    Pad8,
    Pad9,
    UnpackArgs = 128,
    AssignArg,
    Pop,
    Add,
    Sub,
    Equal,
    NotEqual,
    Multiply,
    Divide,
    Modulus,
    GT,
    GTE,
    LT,
    LTE,
    In,
    Not,
    Negate,
    // And,
    // ToString,
    // ToBool,
    // JumpZ,
    // JumpNZ,
    Copy,
    LiteralShimValue,
    LiteralString,
    LiteralNone,
    CreateFn,
    CreateList,
    CreateStruct,
    VariableDeclaration,
    Assignment,
    VariableLoad,
    GetAttr,
    SetAttr,
    StartScope,
    EndScope,
    LoopStart,
    LoopEnd,
    Stringify,
    Break,
    Continue,
    Call,
    Index,
    SetIndex,
    Return,
    Jmp,
    JmpUp,
    JmpNZ,
    JmpZ,
    JmpInitArg,
    Range,
}

pub struct Program {
    pub bytecode: Vec<u8>,
    spans: Vec<Span>,
    script: Vec<u8>,
}

pub fn compile_ast(ast: &Ast) -> Result<Program, String> {
    let mut program = Vec::new();
    let ast_span = Span { start: 0, end: ast.script.len() as u32 };
    compile_block_inner(&ast.block, true, ast_span, &mut program)?;
    let (bytecode, spans): (Vec<u8>, Vec<Span>) = program.into_iter().unzip();
    Ok(Program {
        bytecode: bytecode,
        spans: spans,
        script: ast.script.clone(),
    })
}

pub fn u16_to_u8s(val: u16) -> [u8; 2] {
    [(val >> 8) as u8, (val & 0xff) as u8]
}

pub fn u8s_to_u16(val: [u8; 2]) -> u16 {
    ((val[0] as u16) << 8) + val[1] as u16
}

pub fn compile_fn_body_inner(
    pos_args_required: &Vec<Vec<u8>>,
    pos_args_optional: &Vec<(Vec<u8>, ExprNode)>,
    body: &Block,
    fn_span: Span,
) -> Result<Vec<(u8, Span)>, String> {

    let mut asm = Vec::new();
    asm.push((ByteCode::UnpackArgs as u8, fn_span));
    asm.push((
        pos_args_required.len() as u8,
        fn_span,
    ));
    asm.push((
        pos_args_optional.len() as u8,
        fn_span,
    ));

    for param in pos_args_required.iter() {
        asm.push((
            param.len().try_into().expect("Param len should into u8"),
            fn_span,
        ));
        for b in param {
            asm.push((*b, fn_span));
        }
    }

    for (param, _) in pos_args_optional.iter() {
        asm.push((
            param.len().try_into().expect("Param len should into u8"),
            fn_span,
        ));
        for b in param {
            asm.push((*b, fn_span));
        }
    }

    for (idx, (_param, expr)) in pos_args_optional.iter().enumerate() {
        let jmp_idx = asm.len();
        asm.push((ByteCode::JmpInitArg as u8, expr.span));
        asm.push((0, expr.span));
        asm.push((0, expr.span));

        asm.extend(compile_expression(expr)?);
        asm.push((ByteCode::AssignArg as u8, expr.span));
        asm.push((idx as u8, expr.span));

        let expr_offset = u16_to_u8s(asm.len() as u16 - jmp_idx as u16);
        asm[jmp_idx + 1].0 = expr_offset[0];
        asm[jmp_idx + 2].0 = expr_offset[1];
    }

    for stmt in body.stmts.iter() {
        asm.extend(compile_statement(&stmt)?);
    }

    if let Some(expr) = &body.last_expr {
        let val: Option<&ExprNode> = Some(expr);
        asm.extend(compile_return(&val, fn_span)?);
    } else {
        let needs_implicit_return = if body.stmts.len() > 1 {
            match &body.stmts[body.stmts.len() - 1].data {
                Statement::Return(_) => false,
                _ => true,
            }
        } else {
            true
        };

        if needs_implicit_return {
            let expr = ExprNode {
                data: Expression::Primary(Primary::None),
                span: fn_span,
            };
            let val: Option<&ExprNode> = Some(&expr);
            asm.extend(compile_return(&val, fn_span)?);
        }
    }

    if asm.len() > u16::MAX as usize {
        return Err(format!("Function has more than {} instructions", u16::MAX));
    }
    Ok(asm)
}

pub fn compile_fn_expression(
    pos_args_required: &Vec<Vec<u8>>,
    pos_args_optional: &Vec<(Vec<u8>, ExprNode)>,
    body: &Block,
    fn_span: Span,
) -> Result<Vec<(u8, Span)>, String> {
    // This will be replaced with a relative jump to after the function
    // declaration
    let mut asm = vec![
        (ByteCode::Jmp as u8, fn_span),
        (0, fn_span),
        (0, fn_span),
    ];
    asm.extend(
        compile_fn_body_inner(
            pos_args_required,
            pos_args_optional,
            body,
            fn_span,
        )?
    );

    // Fix the jump offset at the function declaration now that we know
    // the size of the body
    let pc_offset = asm.len() as u16;
    asm[1].0 = (pc_offset >> 8) as u8;
    asm[2].0 = (pc_offset & 0xff) as u8;

    // Assign the value to the ident
    let pc_offset = asm.len() as u16 - 3;
    asm.push((ByteCode::CreateFn as u8, fn_span));
    asm.push(((pc_offset >> 8) as u8, fn_span));
    asm.push(((pc_offset & 0xff) as u8, fn_span));

    Ok(asm)
}

pub fn compile_fn(func: &Fn, fn_span: Span) -> Result<Vec<(u8, Span)>, String> {
    let ident = if let Some(ident) = &func.ident {
        ident
    } else {
        return Err(format!("No ident for function declaration!"));
    };

    let mut asm = compile_fn_expression(
        &func.pos_args_required,
        &func.pos_args_optional,
        &func.body,
        fn_span,
    )?;

    asm.push((
        ByteCode::VariableDeclaration as u8,
        fn_span,
    ));
    asm.push((
        ident
            .len()
            .try_into()
            .expect("Ident len should into u8"),
        fn_span,
    ));
    for b in ident.iter() {
        asm.push((*b, fn_span));
    }

    Ok(asm)
}

pub fn compile_fn_body(func: &Fn, fn_span: Span) -> Result<Vec<(u8, Span)>, String> {
    compile_fn_body_inner(
        &func.pos_args_required,
        &func.pos_args_optional,
        &func.body,
        fn_span,
    )
}

pub fn compile_return(expr: &Option<&ExprNode>, span: Span) -> Result<Vec<(u8, Span)>, String> {
    let mut res = Vec::new();
    if let Some(expr) = expr {
        res.extend(compile_expression(expr)?);
    } else {
        res.push((ByteCode::LiteralNone as u8, span));
    }
    res.push((ByteCode::Return as u8, span));
    Ok(res)
}

pub fn compile_statement(stmt_node: &StatementNode) -> Result<Vec<(u8, Span)>, String> {
    let stmt_span = stmt_node.span;
    match &stmt_node.data {
        Statement::Let(ident, expr) => {
            // Expression evaluates to a value that's on the top of the stack
            let mut expr_asm = compile_expression(expr)?;
            // When getting VariableDeclaration the next byte is the length of
            // the identifier, followed by the
            expr_asm.push((ByteCode::VariableDeclaration as u8, expr.span));
            expr_asm.push((
                ident.len().try_into().expect("Ident len should into u8"),
                expr.span,
            ));
            for b in ident.into_iter() {
                expr_asm.push((*b, expr.span));
            }

            Ok(expr_asm)
        }
        Statement::Assignment(ident, expr) => {
            let mut expr_asm = compile_expression(expr)?;
            expr_asm.push((ByteCode::Assignment as u8, expr.span));
            expr_asm.push((
                ident.len().try_into().expect("Ident len should into u8"),
                expr.span,
            ));
            for b in ident.into_iter() {
                expr_asm.push((*b, expr.span));
            }

            Ok(expr_asm)
        }
        Statement::AttributeAssignment(obj_expr, ident, expr) => {
            let mut expr_asm = compile_expression(obj_expr)?;
            expr_asm.extend(compile_expression(expr)?);
            expr_asm.push((ByteCode::SetAttr as u8, expr.span));
            expr_asm.push((
                ident.len().try_into().expect("Ident len should into u8"),
                expr.span,
            ));
            for b in ident.into_iter() {
                expr_asm.push((*b, expr.span));
            }

            Ok(expr_asm)
        }
        Statement::IndexAssignment(obj_expr, index_expr, expr) => {
            let mut expr_asm = compile_expression(obj_expr)?;
            expr_asm.extend(compile_expression(index_expr)?);
            expr_asm.extend(compile_expression(expr)?);
            expr_asm.push((ByteCode::SetIndex as u8, expr.span));

            Ok(expr_asm)
        }
        Statement::Fn(func) => {
            compile_fn(func, stmt_span)
        }
        Statement::Struct(Struct {
            ident,
            members_required,
            members_optional,
            methods,
        }) => {
            let mut asm = vec![
                (ByteCode::CreateStruct as u8, stmt_span),
                (0, stmt_span),
                (0, stmt_span),
            ];
            asm.push(((members_required.len() + members_optional.len()) as u8, stmt_span));

            // The +1 is for the constructor
            asm.push(((methods.len() + 1) as u8, stmt_span));

            // Add struct name
            asm.push((
                ident
                    .len()
                    .try_into()
                    .expect("Struct name len should into u8"),
                stmt_span,
            ));
            for b in ident.into_iter() {
                asm.push((*b, stmt_span));
            }

            let member_names: Vec<Vec<u8>> = members_required.iter().cloned().chain(members_optional.iter().map(|(x, _)| x.clone())).collect();

            for member in member_names.iter() {
                asm.push((
                    member
                        .len()
                        .try_into()
                        .expect("Member ident len should into u8"),
                    stmt_span,
                ));
                for b in member.into_iter() {
                    asm.push((*b, stmt_span));
                }
            }

            let arg_list = member_names.into_iter()
                .map(|name| Node {
                        data: Expression::Primary(Primary::Identifier(name)),
                        span: stmt_span,
                    }
                )
                .collect();

            let mut method_defs: Vec<(&[u8], Vec<(u8, Span)>)> = Vec::new();
            method_defs.push(
                (
                    b"__init__",
                    compile_fn_body_inner(
                        members_required,
                        members_optional,
                        &Block {
                            stmts: Vec::new(),
                            last_expr: Some(
                                Box::new(
                                    Node {
                                        data: Expression::Call(
                                            Box::new(
                                                Node {
                                                    data: Expression::Primary(Primary::Identifier(ident.clone())),
                                                    span: stmt_span,
                                                }
                                            ),
                                            arg_list,
                                            Vec::new()
                                        ),
                                        span: stmt_span,
                                    }
                                )
                            ),
                        },
                        stmt_span,
                    )?
                )
            );
            for method in methods {
                let ident = if let Some(ident) = &method.ident {
                    ident
                } else {
                    return Err(format!("Method does not have ident!"));
                };
                method_defs.push((&ident, compile_fn_body(method, stmt_span)?));
            }

            let mut jump_asm_idx = Vec::new();
            for (ident, _method_def) in method_defs.iter() {
                jump_asm_idx.push(asm.len());
                asm.push((0, stmt_span));
                asm.push((0, stmt_span));
                asm.push((
                    ident
                        .len()
                        .try_into()
                        .expect("Method ident len should into u8"),
                    stmt_span,
                ));
                for b in ident.into_iter() {
                    asm.push((*b, stmt_span));
                }
            }

            for (method_idx, (_, method_def)) in method_defs.into_iter().enumerate() {
                let jump_idx = jump_asm_idx[method_idx];
                let pc_offset = asm.len() as u16;
                asm[jump_idx].0 = (pc_offset >> 8) as u8;
                asm[jump_idx + 1].0 = (pc_offset & 0xff) as u8;
                asm.extend(method_def);
            }

            let pc_offset = asm.len() as u16;
            asm[1].0 = (pc_offset >> 8) as u8;
            asm[2].0 = (pc_offset & 0xff) as u8;

            asm.push((
                ByteCode::VariableDeclaration as u8,
                stmt_span,
            ));
            asm.push((
                ident.len().try_into().expect("Ident len should into u8"),
                stmt_span,
            ));

            for b in ident.into_iter() {
                asm.push((*b, stmt_span));
            }

            Ok(asm)
        }
        Statement::For(ident, expr, body) => {
            let mut asm = compile_expression(expr)?;

            // Call .iter() and put its .next on the top of the stack
            {
                asm.push((ByteCode::GetAttr as u8, expr.span));
                asm.extend([
                    (4, expr.span),
                    (b'i', expr.span),
                    (b't', expr.span),
                    (b'e', expr.span),
                    (b'r', expr.span),
                ]);

                asm.push((ByteCode::Call as u8, expr.span));
                asm.push((0, expr.span));
                asm.push((0, expr.span));

                asm.push((ByteCode::GetAttr as u8, expr.span));
                asm.extend([
                    (4, expr.span),
                    (b'n', expr.span),
                    (b'e', expr.span),
                    (b'x', expr.span),
                    (b't', expr.span),
                ]);
            }

            let loop_start_idx = asm.len();
            asm.extend(vec![
                (ByteCode::LoopStart as u8, stmt_span),
                (0, stmt_span),
                (0, stmt_span),
            ]);

            // Copy the .next bound method and call it
            asm.push((ByteCode::Copy as u8, expr.span));
            asm.push((ByteCode::Call as u8, expr.span));
            asm.push((0, expr.span));
            asm.push((0, expr.span));

            // Copy the result of .next() so we can later check if it's None
            asm.push((ByteCode::Copy as u8, expr.span));
            asm.push((
                ByteCode::VariableDeclaration as u8,
                stmt_span,
            ));
            asm.push((
                ident
                    .len()
                    .try_into()
                    .expect("For loop ident len should into u8"),
                stmt_span,
            ));
            for b in ident {
                asm.push((*b, stmt_span));
            }

            asm.push((ByteCode::LiteralNone as u8, expr.span));
            asm.push((ByteCode::Equal as u8, expr.span));

            // Jump to `LoopEnd` if calling .next() returns None
            let none_check_idx = asm.len();
            asm.push((ByteCode::JmpNZ as u8, stmt_span));
            asm.push((0, stmt_span));
            asm.push((0, stmt_span));

            asm.extend(compile_block(body, false, stmt_span)?);

            // Jump to start of loop to check the condition again
            let loop_start_offset = u16_to_u8s(asm.len() as u16 - loop_start_idx as u16 - 3);
            asm.push((ByteCode::JmpUp as u8, stmt_span));
            asm.push((loop_start_offset[0], stmt_span));
            asm.push((loop_start_offset[1], stmt_span));

            // This is the offset from none_check_idx that will get us
            // out of the loop
            let pc_offset = u16_to_u8s(asm.len() as u16 - none_check_idx as u16);
            asm[none_check_idx + 1].0 = pc_offset[0];
            asm[none_check_idx + 2].0 = pc_offset[1];

            let loop_end = u16_to_u8s(asm.len() as u16 - loop_start_idx as u16);
            asm[loop_start_idx + 1].0 = loop_end[0];
            asm[loop_start_idx + 2].0 = loop_end[1];

            asm.push((ByteCode::LoopEnd as u8, stmt_span));

            // Remove the `iter` method from the stack
            asm.push((ByteCode::Pop as u8, stmt_span));

            Ok(asm)
        }
        Statement::While(conditional, body) => {
            let mut asm = vec![
                (ByteCode::LoopStart as u8, stmt_span),
                (0, stmt_span),
                (0, stmt_span),
            ];
            asm.extend(compile_expression(conditional)?);

            // Jump to `LoopEnd` if the condition is falsy
            let conditional_check_idx = asm.len();
            asm.push((ByteCode::JmpZ as u8, stmt_span));
            asm.push((0, stmt_span));
            asm.push((0, stmt_span));

            asm.extend(compile_block(body, false, stmt_span)?);

            // Jump to start of loop to check the condition again
            let loop_start_offset = u16_to_u8s(asm.len() as u16 - 3);
            asm.push((ByteCode::JmpUp as u8, stmt_span));
            asm.push((loop_start_offset[0], stmt_span));
            asm.push((loop_start_offset[1], stmt_span));

            // This is the offset from conditional_check_idx that will get us
            // out of the loop
            let pc_offset = u16_to_u8s(asm.len() as u16 - conditional_check_idx as u16);
            asm[conditional_check_idx + 1].0 = pc_offset[0];
            asm[conditional_check_idx + 2].0 = pc_offset[1];

            let loop_end = u16_to_u8s(asm.len() as u16);
            asm[1].0 = loop_end[0];
            asm[2].0 = loop_end[1];

            asm.push((ByteCode::LoopEnd as u8, stmt_span));

            Ok(asm)
        }
        Statement::Break => Ok(vec![(ByteCode::Break as u8, stmt_span)]),
        Statement::Continue => Ok(vec![(ByteCode::Continue as u8, stmt_span)]),
        Statement::Return(expr) => compile_return(&expr.as_ref(), stmt_span),
        Statement::If(conditional, if_body, else_body) => {
            compile_if(conditional, if_body, else_body, false, stmt_span)
        }
        Statement::Expression(expr) => {
            // Expression evaluates to a value that's on the top of the stack
            let mut expr_asm = compile_expression(expr)?;
            // Pop the value since it's not used
            expr_asm.push((ByteCode::Pop as u8, expr.span));

            Ok(expr_asm)
        }
    }
}

pub fn compile_block_inner(block: &Block, is_expr: bool, block_span: Span, asm: &mut Vec<(u8, Span)>) -> Result<(), String> {
    for stmt in block.stmts.iter() {
        asm.extend(compile_statement(&stmt)?);
    }
    if let Some(last_expr) = &block.last_expr {
        asm.extend(compile_expression(&last_expr)?);
    } else {
        asm.push((ByteCode::LiteralNone as u8, block_span));
    }
    if !is_expr {
        asm.push((ByteCode::Pop as u8, block_span));
    }

    Ok(())
}

pub fn compile_block(block: &Block, is_expr: bool, block_span: Span) -> Result<Vec<(u8, Span)>, String> {
    let mut asm = Vec::new();

    asm.push((ByteCode::StartScope as u8, block_span));
    compile_block_inner(block, is_expr, block_span, &mut asm)?;
    asm.push((ByteCode::EndScope as u8, block_span));

    Ok(asm)
}

pub fn compile_if(
    conditional: &ExprNode,
    if_body: &Block,
    else_body: &Block,
    is_expr: bool,
    span: Span,
) -> Result<Vec<(u8, Span)>, String> {
    let mut asm = compile_expression(conditional)?;
    let conditional_check_idx = asm.len();
    asm.push((ByteCode::JmpZ as u8, span));
    asm.push((0, span));
    asm.push((0, span));
    asm.extend(compile_block(if_body, is_expr, span)?);
    asm.push((ByteCode::Jmp as u8, span));
    asm.push((0, span));
    asm.push((0, span));
    // We jump to here when the condition is false
    let else_case_start_idx = asm.len();

    asm.extend(compile_block(else_body, is_expr, span)?);

    // Offset from conditional to the else branch
    let else_jump_offset = u16_to_u8s(else_case_start_idx as u16 - conditional_check_idx as u16);
    asm[conditional_check_idx + 1].0 = else_jump_offset[0];
    asm[conditional_check_idx + 2].0 = else_jump_offset[1];

    // Offset from the end of the if branch to after the else branch
    let if_jump_offset = u16_to_u8s(asm.len() as u16 - else_case_start_idx as u16 + 3);

    asm[else_case_start_idx - 2].0 = if_jump_offset[0];
    asm[else_case_start_idx - 1].0 = if_jump_offset[1];

    Ok(asm)
}

pub fn compile_expression(expr: &ExprNode) -> Result<Vec<(u8, Span)>, String> {
    let span = expr.span;
    match &expr.data {
        Expression::Primary(Primary::None) => {
            let val = ShimValue::None;
            let mut res = vec![(ByteCode::LiteralShimValue as u8, expr.span)];
            for b in val.to_bytes().into_iter() {
                res.push((b, expr.span));
            }
            Ok(res)
        }
        Expression::Primary(Primary::Bool(b)) => {
            let val = ShimValue::Bool(*b);
            let mut res = vec![(ByteCode::LiteralShimValue as u8, expr.span)];
            for b in val.to_bytes().into_iter() {
                res.push((b, expr.span));
            }
            Ok(res)
        }
        Expression::Primary(Primary::Integer(i)) => {
            let val = ShimValue::Integer(*i);
            let mut res = vec![(ByteCode::LiteralShimValue as u8, expr.span)];
            for b in val.to_bytes().into_iter() {
                res.push((b, expr.span));
            }
            Ok(res)
        }
        Expression::Primary(Primary::Float(f)) => {
            let val = ShimValue::Float(*f);
            let mut res = vec![(ByteCode::LiteralShimValue as u8, expr.span)];
            for b in val.to_bytes().into_iter() {
                res.push((b, expr.span));
            }
            Ok(res)
        }
        Expression::Primary(Primary::Identifier(ident)) => {
            let mut res = Vec::new();
            res.push((ByteCode::VariableLoad as u8, expr.span));
            res.push((
                ident.len().try_into().expect("Ident len should into u8"),
                expr.span,
            ));
            for b in ident.into_iter() {
                res.push((*b, expr.span));
            }
            Ok(res)
        }
        Expression::Primary(Primary::String(s)) => {
            let mut res = Vec::new();
            res.push((ByteCode::LiteralString as u8, expr.span));
            res.push((s.len().try_into().expect("Ident should into u8"), expr.span));
            for b in s.into_iter() {
                res.push((*b, expr.span));
            }
            Ok(res)
        }
        Expression::Primary(Primary::List(items)) => {
            let mut res = Vec::new();
            for expr in items {
                res.extend(compile_expression(&expr)?);
            }
            res.push((ByteCode::CreateList as u8, expr.span));
            let len: u16 = items.len().try_into().expect("List should fit into u16");
            res.push(((len >> 8) as u8, expr.span));
            res.push(((len & 0xff) as u8, expr.span));
            Ok(res)
        }
        Expression::Primary(Primary::Expression(expr)) => compile_expression(&expr),
        Expression::BooleanOp(BooleanOp::And(a, b)) => {
            let mut asm = compile_expression(&a)?;
            asm.push((ByteCode::Copy as u8, span));

            let short_circuit_idx = asm.len();
            asm.push((ByteCode::JmpZ as u8, span));
            asm.push((0, span));
            asm.push((0, span));

            // Since the result of a is truthy we get rid of it and return the result of b
            asm.push((ByteCode::Pop as u8, span));

            asm.extend(compile_expression(&b)?);

            let short_circuit_offset = u16_to_u8s(asm.len() as u16 - short_circuit_idx as u16);
            asm[short_circuit_idx + 1].0 = short_circuit_offset[0];
            asm[short_circuit_idx + 2].0 = short_circuit_offset[1];

            Ok(asm)
        }
        Expression::BooleanOp(BooleanOp::Or(a, b)) => {
            let mut asm = compile_expression(&a)?;
            asm.push((ByteCode::Copy as u8, span));

            let short_circuit_idx = asm.len();
            asm.push((ByteCode::JmpNZ as u8, span));
            asm.push((0, span));
            asm.push((0, span));

            // Since the result of a is falsy we get rid of it and return the result of b
            asm.push((ByteCode::Pop as u8, span));

            asm.extend(compile_expression(&b)?);

            let short_circuit_offset = u16_to_u8s(asm.len() as u16 - short_circuit_idx as u16);
            asm[short_circuit_idx + 1].0 = short_circuit_offset[0];
            asm[short_circuit_idx + 2].0 = short_circuit_offset[1];

            Ok(asm)
        }
        Expression::BinaryOp(op) => {
            let (opcode, a, b) = match op {
                BinaryOp::Add(a, b) => (ByteCode::Add, a, b),
                BinaryOp::Subtract(a, b) => (ByteCode::Sub, a, b),
                BinaryOp::Equal(a, b) => (ByteCode::Equal, a, b),
                BinaryOp::NotEqual(a, b) => (ByteCode::NotEqual, a, b),
                BinaryOp::Multiply(a, b) => (ByteCode::Multiply, a, b),
                BinaryOp::Modulus(a, b) => (ByteCode::Modulus, a, b),
                BinaryOp::Divide(a, b) => (ByteCode::Divide, a, b),
                BinaryOp::GT(a, b) => (ByteCode::GT, a, b),
                BinaryOp::GTE(a, b) => (ByteCode::GTE, a, b),
                BinaryOp::LT(a, b) => (ByteCode::LT, a, b),
                BinaryOp::LTE(a, b) => (ByteCode::LTE, a, b),
                BinaryOp::In(a, b) => (ByteCode::In, a, b),
                BinaryOp::Range(a, b) => (ByteCode::Range, a, b),
            };
            let mut res = compile_expression(&a)?;
            res.extend(compile_expression(&b)?);
            res.push((opcode as u8, expr.span));
            Ok(res)
        }
        Expression::UnaryOp(op) => {
            let (opcode, a) = match op {
                UnaryOp::Not(a) => (ByteCode::Not, a),
                UnaryOp::Negate(a) => (ByteCode::Negate, a),
            };
            let mut res = compile_expression(&a)?;
            res.push((opcode as u8, expr.span));
            Ok(res)
        }
        Expression::Stringify(expr) => {
            let mut asm = compile_expression(&expr)?;
            asm.push((ByteCode::Stringify as u8, expr.span));
            Ok(asm)
        }
        Expression::Index(obj_expr, index_expr) => {
            let mut asm = compile_expression(&obj_expr)?;
            asm.extend(compile_expression(index_expr)?);
            asm.push((ByteCode::Index as u8, expr.span));

            Ok(asm)
        }
        Expression::Call(expr, args, kwargs) => {
            // First we evaluate the thing that needs to be called
            let mut res = compile_expression(&expr)?;

            // Then we evaluate each argument
            for arg_expr in args.iter() {
                res.extend(compile_expression(arg_expr)?);
            }

            for (ident, kwarg_expr) in kwargs.iter() {
                res.push((ByteCode::LiteralString as u8, kwarg_expr.span));
                res.push((
                    ident.len().try_into().expect("Ident should into u8"),
                    kwarg_expr.span,
                ));
                for b in ident.into_iter() {
                    res.push((*b, kwarg_expr.span));
                }

                res.extend(compile_expression(kwarg_expr)?);
            }

            res.push((ByteCode::Call as u8, span));
            res.push((args.len() as u8, span));
            res.push((kwargs.len() as u8, span));
            Ok(res)
        }
        Expression::Attribute(expr, ident) => {
            let mut res = compile_expression(&expr)?;
            res.push((ByteCode::GetAttr as u8, span));
            res.push((
                ident.len().try_into().expect("Ident len should into u8"),
                span,
            ));
            for b in ident.into_iter() {
                res.push((*b, span));
            }
            Ok(res)
        }
        Expression::Block(block) => compile_block(block, true, span),
        Expression::If(conditional, if_body, else_body) => {
            compile_if(conditional, if_body, else_body, true, span)
        },
        Expression::Fn(func) => {
            compile_fn_expression(
                &func.pos_args_required,
                &func.pos_args_optional,
                &func.body,
                span,
            )
        },
    }
}

#[derive(Debug)]
pub struct Bitmask {
    data: Vec<u64>,
}

impl Bitmask {
    pub fn new(num_bits: usize) -> Self {
        // Round up if we don't have a number of bits that's cleanly divisible by 64
        let blocks = num_bits.div_ceil(64);

        Bitmask {
            data: vec![0; blocks],
        }
    }

    pub fn set(&mut self, index: usize) {
        let (block_idx, bit_offset) = self.pos(index);
        self.data[block_idx] |= 1 << bit_offset;
    }

    pub fn is_set(&self, index: usize) -> bool {
        let (block_idx, bit_offset) = self.pos(index);
        (self.data[block_idx] & (1 << bit_offset)) != 0
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    pub fn find_zeros(&self) -> Vec<Range<usize>> {
        let _zone = zone_scoped!("find_zeros");
        let mut ranges = Vec::new();
        let mut start_of_run: Option<usize> = None;

        for (idx, word) in self.data.iter().enumerate() {
            if *word == 0 {
                if start_of_run.is_none() {
                    start_of_run = Some(idx*64);
                }
            } else if *word == u64::MAX {
                if let Some(start_bit) = start_of_run {
                    ranges.push(start_bit..(idx*64));
                    start_of_run = None;
                }
            } else {
                let bit_offset: usize;
                match start_of_run {
                    Some(start_bit) => {
                        bit_offset = word.trailing_zeros() as usize;
                        ranges.push(start_bit..(idx*64 + bit_offset));
                        start_of_run = None
                    }
                    None => {
                        bit_offset = word.trailing_ones() as usize;
                        start_of_run = Some(idx*64 + bit_offset);
                    }
                }
                let mut shifted_word = word >> bit_offset;
                for i in (bit_offset as usize)..64 {
                    let is_zero = (shifted_word & 1) == 0;

                    if is_zero {
                        if start_of_run == None {
                            start_of_run = Some(i);
                        }
                    } else {
                        if let Some(start) = start_of_run {
                            ranges.push(start..i);
                            start_of_run = None;
                        }
                    }
                    shifted_word >>= 1;
                }
            }
        }


        if let Some(start_bit) = start_of_run {
            ranges.push(start_bit..self.data.len()*64);
        }

        ranges
    }

    fn pos(&self, index: usize) -> (usize, usize) {
        (index / 64, index % 64)
    }
}

struct GC<'a> {
    mem: &'a mut MMU,
    mask: Bitmask,
}

impl<'a> GC<'a> {
    fn new(mem: &'a mut MMU) -> Self {
        let last_block_start = mem.free_list[mem.free_list.len()-1].pos;
        let mut mask = Bitmask::new(last_block_start.into());
        // Mark word 0 so the GC never frees the sentinel reserved by MMU::with_capacity
        mask.set(0);
        Self {
            mem,
            mask,
        }
    }

    fn mark(&mut self, mut vals: Vec<ShimValue>) {
        let _zone = zone_scoped!("GC mark");
        unsafe {
            while !vals.is_empty() {
                let _zone = zone_scoped!("GC mark item");
                match vals.pop().unwrap() {
                    ShimValue::Integer(_) | ShimValue::Float(_) | ShimValue::Bool(_) | ShimValue::Unit | ShimValue::None | ShimValue::Uninitialized => (),
                    ShimValue::Fn(fn_pos) => {
                        let pos: usize = fn_pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        // Mark the ShimFn struct (8 bytes = 1 word: u32 pc + Word name)
                        self.mask.set(pos);
                        
                        // Mark the function name string
                        let shim_fn: &ShimFn = self.mem.get(fn_pos);
                        vals.push(ShimValue::String(shim_fn.name_len, 0, shim_fn.name.0));
                    },
                    ShimValue::List(pos) => {
                        let pos: usize = pos.into();
                        let lst: &ShimList = self.mem.get(pos.into());
                        for idx in 0..lst.len() {
                            vals.push(lst.get(self.mem, idx as isize).unwrap());
                        }

                        let contents_pos = usize::from(lst.data);
                        for idx in contents_pos..(contents_pos + lst.capacity()) {
                            self.mask.set(idx);
                        }
                    },
                    ShimValue::String(len, offset, pos) => {
                        let len = len as usize;
                        let offset = offset as usize;
                        let pos: usize = usize::from(pos);
                        // TODO: check this...
                        for idx in pos..(pos + (offset + len).div_ceil(8)) {
                            self.mask.set(idx);
                        }
                    },
                    ShimValue::Dict(pos) => {
                        let pos: usize = pos.into();
                        let dict: &ShimDict = std::mem::transmute(&self.mem.mem[pos]);
                        let u64_slice = &self.mem.mem[
                            usize::from(dict.entries)..
                            usize::from(dict.entries)+3*(dict.entry_count as usize)
                        ];
                        let entries: &[DictEntry] = std::slice::from_raw_parts(
                            u64_slice.as_ptr() as *const DictEntry,
                            u64_slice.len() / 3,
                        );

                        // Push the keys/vals
                        let count: usize = dict.entry_count as usize;
                        for entry in &entries[..count] {
                            if entry.key.is_uninitialized() {
                                vals.push(entry.key);
                                vals.push(entry.value);
                            }
                        }

                        // Mark the sapce for the dict struct
                        for idx in pos..(pos + (std::mem::size_of::<ShimDict>()/8)) {
                            self.mask.set(idx);
                        }

                        let size = 1 << dict.size_pow;

                        // Mark the indices array
                        let indices_pos: usize = dict.indices.into();
                        for idx in indices_pos..(indices_pos + size) {
                            self.mask.set(idx);
                        }

                        // Mark the entries array
                        let entries_pos: usize = dict.entries.into();
                        for idx in entries_pos..(entries_pos + size*3) {
                            self.mask.set(idx);
                        }
                    },
                    ShimValue::StructDef(pos) => {
                        let pos: usize = pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        let def: &StructDef = self.mem.get(pos.into());
                        for idx in pos..(pos + def.mem_size()) {
                            self.mask.set(idx);
                        }
                    },
                    ShimValue::Struct(pos) => {
                        let pos: usize = pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        let def_pos: usize = self.mem.mem[pos] as usize;
                        let def: &StructDef = self.mem.get(def_pos.into());

                        for idx in pos..(pos + def.member_count as usize + 1) {
                            self.mask.set(idx);
                            // Push the members
                            if idx != pos {
                                vals.push(
                                    ShimValue::from_u64(self.mem.mem[idx])
                                );
                            }
                        }
                        vals.push(ShimValue::StructDef(def_pos.into()));
                    },
                    ShimValue::NativeFn(pos) => {
                        let pos: usize = pos.into();
                        self.mask.set(pos);
                    },
                    ShimValue::Native(pos) => {
                        let pos: usize = pos.into();
                        assert!(std::mem::size_of::<Box<dyn ShimNative>>() == 16);
                        self.mask.set(pos);
                        self.mask.set(pos+1);

                        let ptr: &Box<dyn ShimNative> = std::mem::transmute(&self.mem.mem[pos]);

                        vals.extend(ptr.gc_vals());
                    },
                    ShimValue::BoundMethod(pos, fn_pos) => {
                        // Mark the bound struct
                        let val = ShimValue::Struct(pos);
                        vals.push(val);
                        // Mark the function
                        vals.push(ShimValue::Fn(fn_pos));
                    },
                    ShimValue::BoundNativeMethod(pos) => {
                        let pos: usize = pos.into();
                        // Native ShimValue
                        self.mask.set(pos);
                        // Pointer to the fn
                        self.mask.set(pos+1);

                        // Any values that the obj holds
                        let ptr: &Box<dyn ShimNative> = std::mem::transmute(&self.mem.mem[pos]);
                        vals.extend(ptr.gc_vals());
                        vals.push(ShimValue::Native(pos.into()));
                    },
                    ShimValue::Environment(pos) => {
                        let scope: &EnvScope = self.mem.get(pos);

                        // Chunk of memory that store the EnvScope metadata
                        let pos: usize = pos.into();
                        for bit in pos..(pos + std::mem::size_of::<EnvScope>().div_ceil(8)) {
                            self.mask.set(bit);
                        }

                        // Data block
                        let start = usize::from(scope.data);
                        let end = start + scope.capacity as usize;
                        for bit in start..end {
                            self.mask.set(bit);
                        }
                        
                        // Walk the contiguous data block and collect values
                        let bytes = scope.raw_bytes(&self.mem);
                        let mut off = 0usize;
                        while off < bytes.len() {
                            let key_len = bytes[off] as usize;
                            let value_offset = off + 1 + key_len;
                            let val: ShimValue = {
                                let mut val_bytes = [0u8; 8];
                                std::ptr::copy_nonoverlapping(bytes[value_offset..].as_ptr(), val_bytes.as_mut_ptr(), 8);
                                std::mem::transmute(val_bytes)
                            };
                            vals.push(val);
                            off = value_offset + 8;
                        }

                        if scope.parent != 0.into() {
                            vals.push(ShimValue::Environment(Word(scope.parent)));
                        }
                    }
                }
            }
        }
    }

    fn sweep(&mut self) {
        let _zone = zone_scoped!("GC sweep");

        // TODO: need to add the original last block from the free list
        let last_block = self.mem.free_list[self.mem.free_list.len()-1];
        self.mem.free_list = self.mask
            .find_zeros()
            .iter()
            .map(|block| FreeBlock { pos: block.start.into(), size: (block.end-block.start).into() }).collect();
        let new_last_block = self.mem.free_list[self.mem.free_list.len()-1];
        if new_last_block.end() >= last_block.pos {
            // Merge with the new last block
            let len = self.mem.free_list.len();
            self.mem.free_list[len - 1].size = last_block.end() - self.mem.free_list[len - 1].pos;
        } else {
            // Append the previous last free block (which was not included in the bitmask)
            self.mem.free_list.push(last_block);
        }
    }
}

// TODO: uncomment #[derive(Facet)]
pub struct Interpreter {
    pub mem: MMU,
    pub source: HashMap<String, String>,
    pub program: Rc<Program>,
}

impl Interpreter {
    pub fn print_mem(&self) {
        let _zone = zone_scoped!("print_mem");
        let mut count = 0;
        let mut idx = 0;
        for block in self.mem.free_list.iter() {
            while idx < block.pos.into() {
                println!("{:06}: {:016x}", idx, self.mem.mem[idx]);
                idx += 1;
                count += 1
            }

            if count > 100 {
                break;
            }
        }
    }

    pub fn print_env(&self, env: &Environment) {
        let _zone = zone_scoped!("print_env");
        let mut current_scope_pos = env.current_scope;
        let mut idx = 0;
        
        loop {
            if current_scope_pos == 0 {
                break;
            }
            
            println!("Scope {idx}");
            
            // Get the EnvScope
            let scope: &EnvScope = unsafe {
                self.mem.get(Word(current_scope_pos.into()))
            };
            
            // Walk the contiguous data block and print entries
            let bytes = unsafe { scope.raw_bytes(&self.mem) };
            let mut off = 0usize;
            while off < bytes.len() {
                let key_len = bytes[off] as usize;
                let key_bytes = &bytes[off + 1..off + 1 + key_len];
                let value_offset = off + 1 + key_len;
                let val: ShimValue = unsafe {
                    let mut val_bytes = [0u8; 8];
                    std::ptr::copy_nonoverlapping(bytes[value_offset..].as_ptr(), val_bytes.as_mut_ptr(), 8);
                    std::mem::transmute(val_bytes)
                };
                println!("{:>12}: {:?}", debug_u8s(key_bytes), val);
                match val {
                    ShimValue::Struct(pos) => {
                        unsafe {
                            let def_pos: u64 = *self.mem.get(pos);
                            let def_pos: Word = Word((def_pos as u32).into());
                            let def: &StructDef = self.mem.get(def_pos);
                            for (attr, loc) in def.lookup.iter() {
                                match loc {
                                    StructAttribute::MemberInstanceOffset(offset) => {
                                        let val: ShimValue = *self.mem.get(pos + *offset as u32 + 1);
                                        println!("                - {} = {:?}", debug_u8s(&attr), val);
                                    },
                                    StructAttribute::MethodDef(_) => (),
                                };
                            }
                        }
                    },
                    ShimValue::StructDef(pos) => {
                        unsafe {
                            let def: &StructDef = self.mem.get(pos);
                            for (attr, loc) in def.lookup.iter() {
                                match loc {
                                    StructAttribute::MemberInstanceOffset(_) => {
                                        println!("                - {}", debug_u8s(&attr));
                                    },
                                    StructAttribute::MethodDef(_) => {
                                        println!("                - {}()", debug_u8s(&attr));
                                    }
                                };
                            }
                        }
                    },
                    _ => (),
                }
                off = value_offset + 8;
            }
            
            // Move to parent scope
            let parent: u32 = scope.parent.into();
            current_scope_pos = parent;
            idx += 1;
        }
    }

    pub fn gc(&mut self, env: &Environment) {
        let _zone = zone_scoped!("GC");
        //self.print_mem();
        //self.print_env(env);
        
        unsafe {
            let _scope: &EnvScope = self.mem.get(Word(env.current_scope.into()));
        }

        let mut roots: Vec<ShimValue> = Vec::new();
        roots.push(ShimValue::Environment(Word(env.current_scope.into())));
        
        // Now create GC and process roots
        let mut gc = {
            let _zone = zone_scoped!("Init GC");
            GC::new(&mut self.mem)
        };
        gc.mark(roots);
        gc.sweep();
    }

    pub fn create(config: &Config, program: Program) -> Self {
        let mmu = MMU::with_capacity(Word((config.memory_space_bytes / 8).into()));

        Self {
            mem: mmu,
            source: HashMap::new(),
            program: Rc::new(program),
        }
    }

    pub fn append_program(&mut self, program: Program) -> Result<(), String> {
        let span_offset = self.program.script.len() as u32;
        Rc::<Program>::get_mut(&mut self.program).unwrap().bytecode.extend(program.bytecode);
        Rc::<Program>::get_mut(&mut self.program).unwrap().spans.extend(
            program.spans.into_iter().map(|span| Span {
                start: span.start + span_offset,
                end: span.end + span_offset,
            })
        );
        Rc::<Program>::get_mut(&mut self.program).unwrap().script.extend(program.script);

        Ok(())
    }

    pub fn execute_bytecode_extended(
        &mut self,
        mod_pc: &mut usize,
        mut pending_args: ArgBundle,
        env: &mut Environment,
    ) -> Result<ShimValue, String> {
        let _zone = zone_scoped!("Execute Bytecode");
        let mut pc = *mod_pc;
        // These are values that are operated on. Expressions push and pop to
        // this stack, return values go on this stack etc.
        let mut stack: Vec<ShimValue> = Vec::new();


        // This is the (PC, loop_info, scope_count, caller_scope, fn_optional_param_names,
        // fn_optional_param_name_idx) call stack
        let mut stack_frame: Vec<(
            // PC
            usize,
            // loop_info
            Vec<(usize, usize, usize)>,
            // scope_count
            usize,
            // caller_scope
            u32,
            // fn_optional_param_names
            Vec<Ident>,
            // fn_optional_param_name_idx
            usize,
        )> = Vec::new();

        // This is the PC of the (start, end, scope_count) of the current loop for the
        // current function
        let mut loop_info: Vec<(usize, usize, usize)> = Vec::new();

        let mut fn_optional_param_name_idx = 0;
        let mut fn_optional_param_names: Vec<Ident> = Vec::new();

        let bytes = &self.program.clone().bytecode;
        while pc < bytes.len() {
            //let _zone = zone_scoped!("Execute Single Instruction");
            match bytes[pc] {
                val if val == ByteCode::Pop as u8 => {
                    stack.pop();
                }
                val if val == ByteCode::Add as u8 => {
                    let b = stack.pop().expect("Operand for add");
                    let a = stack.pop().expect("Operand for add");

                    match a.add(self, &b, &mut pending_args).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })? {
                        CallResult::ReturnValue(res) => stack.push(res),
                        CallResult::PC(new_pc, captured_scope) => {
                            stack_frame.push((
                                pc + 1,
                                loop_info.clone(),
                                env.scope_depth(&self.mem),
                                env.current_scope,
                                fn_optional_param_names.clone(),
                                fn_optional_param_name_idx,
                            ));
                            loop_info = Vec::new();
                            // Restore the captured environment and push a new scope for function locals
                            env.current_scope = captured_scope;
                            env.push_scope(&mut self.mem);
                            pc = new_pc as usize;
                            continue;
                        }
                    }
                }
                val if val == ByteCode::Sub as u8 => {
                    let b = stack.pop().expect("Operand for add");
                    let a = stack.pop().expect("Operand for add");
                    stack.push(a.sub(&b)?);
                }
                val if val == ByteCode::Equal as u8 => {
                    let b = stack.pop().expect("Operand for add");
                    let a = stack.pop().expect("Operand for add");
                    stack.push(a.equal(self, &b)?);
                }
                val if val == ByteCode::NotEqual as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::NotEqual");
                    let a = stack.pop().expect("Operand for ByteCode::NotEqual");
                    stack.push(a.not_equal(self, &b)?);
                }
                val if val == ByteCode::Multiply as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::Multiply");
                    let a = stack.pop().expect("Operand for ByteCode::Multiply");
                    stack.push(a.mul(self, &b)?);
                }
                val if val == ByteCode::Divide as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::Divide");
                    let a = stack.pop().expect("Operand for ByteCode::Divide");
                    stack.push(a.div(self, &b)?);
                }
                val if val == ByteCode::Modulus as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::Modulus");
                    let a = stack.pop().expect("Operand for ByteCode::Modulus");
                    stack.push(a.modulus(self, &b)?);
                }
                val if val == ByteCode::GT as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::GT");
                    let a = stack.pop().expect("Operand for ByteCode::GT");
                    stack.push(a.gt(self, &b)?);
                }
                val if val == ByteCode::GTE as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::GTE");
                    let a = stack.pop().expect("Operand for ByteCode::GTE");
                    stack.push(a.gte(self, &b)?);
                }
                val if val == ByteCode::LT as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::LT");
                    let a = stack.pop().expect("Operand for ByteCode::LT");
                    stack.push(a.lt(self, &b)?);
                }
                val if val == ByteCode::LTE as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::LTE");
                    let a = stack.pop().expect("Operand for ByteCode::LTE");
                    stack.push(a.lte(self, &b)?);
                }
                val if val == ByteCode::In as u8 => {
                    let b = stack.pop().expect("Operand for ByteCode::In");
                    let a = stack.pop().expect("Operand for ByteCode::In");
                    stack.push(a.contains(self, &b)?);
                }
                val if val == ByteCode::Range as u8 => {
                    let end = stack.pop().expect("Operand for ByteCode::Range");
                    let start = stack.pop().expect("Operand for ByteCode::Range");
                    
                    let range = RangeNative {
                        start: start,
                        end: end,
                    };
                    stack.push(self.mem.alloc_native(range));
                }
                val if val == ByteCode::Not as u8 => {
                    let a = stack.pop().expect("Operand for ByteCode::Not");
                    stack.push(a.not(self)?);
                }
                val if val == ByteCode::Negate as u8 => {
                    let a = stack.pop().expect("Operand for ByteCode::Negate");
                    stack.push(a.neg(self)?);
                }
                val if val == ByteCode::Stringify as u8 => {
                    let a = stack.pop().expect("Operand for ByteCode::Stringify");
                    stack.push(a.to_shimvalue_string(self));
                }
                val if val == ByteCode::LiteralNone as u8 => {
                    stack.push(ShimValue::None);
                }
                val if val == ByteCode::Copy as u8 => {
                    stack.push(*stack.last().expect("non-empty stack"));
                }
                val if val == ByteCode::LoopStart as u8 => {
                    let loop_end = pc + (((bytes[pc + 1] as usize) << 8) + bytes[pc + 2] as usize);
                    loop_info.push((pc + 3, loop_end, env.scope_depth(&self.mem)));
                    pc += 2;
                }
                val if val == ByteCode::LoopEnd as u8 => {
                    loop_info.pop().expect("loop end should have loop info");
                }
                val if val == ByteCode::Break as u8 => {
                    let (_, end_pc, scope_count) =
                        loop_info.last().expect("break should have loop info");
                    while env.scope_depth(&self.mem) > *scope_count {
                        env.pop_scope(&self.mem).unwrap();
                    }
                    pc = *end_pc;
                    continue;
                }
                val if val == ByteCode::Continue as u8 => {
                    let (start_pc, _, scope_count) =
                        loop_info.last().expect("continue should have loop info");
                    while env.scope_depth(&self.mem) > *scope_count {
                        env.pop_scope(&self.mem).unwrap();
                    }
                    pc = *start_pc;
                    continue;
                }
                val if val == ByteCode::UnpackArgs as u8 => {
                    let required_arg_count = bytes[pc + 1] as usize;
                    let optional_arg_count = bytes[pc + 2] as usize;

                    let mut pos_arg_idx = 0;

                    fn_optional_param_names.clear();
                    fn_optional_param_name_idx = 0;

                    // Assign each parameter in the function to something
                    let mut idx = pc + 3;
                    for param_idx in 0..(required_arg_count + optional_arg_count) {
                        let len = bytes[idx];
                        let param_name = &bytes[idx + 1..idx + 1 + len as usize];

                        if param_idx >= required_arg_count {
                            fn_optional_param_names.push(param_name.to_vec());
                        }

                        // If the parameter was provided as a kwarg, set that now
                        let mut set_arg = false;
                        let mut found_idx = None;
                        for (idx, (ident, _val)) in pending_args.kwargs.iter().enumerate() {
                            if ident == param_name {
                                found_idx = Some(idx);
                                break;
                            }
                        }
                        if let Some(idx) = found_idx {
                            let (_ident, val) = pending_args.kwargs.remove(idx);
                            env.insert_new(self, param_name.to_vec(), val);
                            set_arg = true;
                        }

                        // If it wasn't set as a kwarg, assign it the next positional arg
                        if !set_arg {
                            let val = if pos_arg_idx < pending_args.args.len() {
                                pos_arg_idx += 1;
                                pending_args.args[pos_arg_idx - 1]
                            } else {
                                // We ran out of positional args

                                // If we haven't finished assigning the required
                                // arguments then the function wasn't provided
                                // enough and we need to exit
                                if param_idx < required_arg_count {
                                    return Err(format_script_err(
                                        self.program.spans[stack_frame[stack_frame.len() - 1].0 - 3],
                                        &self.program.script,
                                        &format!("Not enough positional args, arg_count: {}, kwarg_count: {}", pending_args.args.len(), pending_args.kwargs.len()),
                                    ));
                                }

                                ShimValue::Uninitialized
                            };
                            env.insert_new(self, param_name.to_vec(), val);
                        }

                        idx += 1 + len as usize;
                    }
                    if pos_arg_idx != pending_args.args.len() {
                        let remaining = pending_args.args.len() - pos_arg_idx;
                        return Err(format_script_err(
                            self.program.spans[stack_frame[stack_frame.len() - 1].0 - 3],
                            &self.program.script,
                            &format!("Too many positional args, {} remaining", remaining),
                        ));
                    }
                    if !pending_args.kwargs.is_empty() {
                        let mut msg = "Unused kwargs remaining:".to_string();
                        for (ident, _) in pending_args.kwargs.iter() {
                            msg.push(' ');
                            msg.push_str(debug_u8s(ident));
                        }
                        return Err(format_script_err(
                            self.program.spans[stack_frame[stack_frame.len() - 1].0 - 3],
                            &self.program.script,
                            &msg,
                        ));
                    }
                    pc = idx;
                    continue;
                }
                val if val == ByteCode::JmpInitArg as u8 => {
                    let optional_param_name = &fn_optional_param_names[fn_optional_param_name_idx];
                    fn_optional_param_name_idx += 1;

                    match env.get(self, optional_param_name) {
                        Some(ShimValue::Uninitialized) => (),
                        Some(_) => {
                            let new_pc =
                                pc + (((bytes[pc + 1] as usize) << 8) + bytes[pc + 2] as usize);
                            pc = new_pc;
                            continue;
                        }
                        None => {
                            return Err(format!(
                                "Expected UnpackArgs to set indent that doesn't exist!"
                            ));
                        }
                    }
                    pc += 2;
                }
                val if val == ByteCode::AssignArg as u8 => {
                    let arg_num = bytes[pc + 1] as usize;
                    let optional_param_name = &fn_optional_param_names[arg_num];
                    env.update(self, optional_param_name, stack.pop().unwrap())?;
                    pc += 1;
                }
                val if val == ByteCode::LiteralShimValue as u8 => {
                    let bytes = [
                        bytes[pc + 1],
                        bytes[pc + 2],
                        bytes[pc + 3],
                        bytes[pc + 4],
                        bytes[pc + 5],
                        bytes[pc + 6],
                        bytes[pc + 7],
                        bytes[pc + 8],
                    ];
                    stack.push(ShimValue::from_bytes(bytes));
                    pc += 8;
                }
                val if val == ByteCode::LiteralString as u8 => {
                    let str_len = bytes[pc + 1] as usize;
                    let contents = &bytes[pc + 2..pc + 2 + str_len as usize];

                    stack.push(self.mem.alloc_str(contents));
                    pc += 1 + str_len;
                }
                val if val == ByteCode::VariableDeclaration as u8 => {
                    let val = stack.pop().expect("Value for declaration");
                    let ident_len = bytes[pc + 1] as usize;
                    let ident = &bytes[pc + 2..pc + 2 + ident_len as usize];
                    env.insert_new(self, ident.to_vec(), val);
                    pc += 1 + ident_len;
                }
                val if val == ByteCode::Assignment as u8 => {
                    let val = stack.pop().expect("Value for assignment");
                    let ident_len = bytes[pc + 1] as usize;
                    let ident = &bytes[pc + 2..pc + 2 + ident_len as usize];

                    if !env.contains_key(self, ident) {
                        return Err(format_script_err(
                            self.program.spans[pc],
                            &self.program.script,
                            &format!("Identifier {:?} not found", ident),
                        ));
                    }
                    env.update(self, ident, val)?;

                    pc += 1 + ident_len;
                }
                val if val == ByteCode::VariableLoad as u8 => {
                    let ident_len = bytes[pc + 1] as usize;
                    let ident = &bytes[pc + 2..pc + 2 + ident_len as usize];
                    if let Some(value) = env.get(self, ident) {
                        stack.push(value);
                    } else {
                        return Err(format_script_err(
                            self.program.spans[pc],
                            &self.program.script,
                            &format!("Unknown identifier {:?}", debug_u8s(ident)),
                        ));
                    }
                    pc += 1 + ident_len;
                }
                val if val == ByteCode::GetAttr as u8 => {
                    let ident_len = bytes[pc + 1] as usize;
                    let ident = &bytes[pc + 2..pc + 2 + ident_len as usize];

                    let obj = stack.pop().expect("val to access");

                    let res = match obj.get_attr(self, ident) {
                        Ok(val) => val,
                        Err(msg) => return Err(format_script_err(
                            self.program.spans[pc],
                            &self.program.script,
                            &msg,
                        )),
                    };


                    stack.push(res);

                    pc += 1 + ident_len;
                }
                val if val == ByteCode::SetAttr as u8 => {
                    let ident_len = bytes[pc + 1] as usize;
                    let ident = &bytes[pc + 2..pc + 2 + ident_len as usize];

                    let val = stack.pop().expect("val to assign");
                    let obj = stack.pop().expect("obj to set");
                    obj.set_attr(self, ident, val).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })?;

                    pc += 1 + ident_len;
                }
                val if val == ByteCode::Index as u8 => {
                    let index = stack.pop().expect("index val");
                    let obj = stack.pop().expect("index obj");

                    let val = obj.index(self, &index).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })?;

                    stack.push(val);
                }
                val if val == ByteCode::SetIndex as u8 => {
                    let val = stack.pop().expect("index assigned val");
                    let index = stack.pop().expect("index index");
                    let obj = stack.pop().expect("index obj");

                    obj.set_index(self, &index, &val).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })?;
                }
                val if val == ByteCode::Call as u8 => {
                    let arg_count = bytes[pc + 1];
                    let kwarg_count = bytes[pc + 2];

                    pending_args.clear();

                    for _ in 0..kwarg_count {
                        let val = stack.pop().unwrap();
                        let ident = match stack.pop().unwrap() {
                            val @ ShimValue::String(..) => {
                                val.string(self)?.to_vec()
                            },
                            other => return Err(format!("Invalid kwarg ident {:?}", other)),
                        };
                        pending_args.kwargs.push((ident, val));
                    }

                    for _ in 0..arg_count {
                        pending_args.args.push(stack.pop().unwrap());
                    }
                    pending_args.args.reverse();
                    pending_args.kwargs.reverse();

                    let callable = stack.pop().expect("callable not on stack");

                    match callable.call(self, &mut pending_args).map_err(|err_str| {
                        format_script_err(self.program.spans[pc], &self.program.script, &err_str)
                    })? {
                        CallResult::ReturnValue(res) => stack.push(res),
                        CallResult::PC(new_pc, captured_scope) => {
                            stack_frame.push((
                                pc + 3,
                                loop_info.clone(),
                                env.scope_depth(&self.mem),
                                env.current_scope,
                                fn_optional_param_names.clone(),
                                fn_optional_param_name_idx,
                            ));
                            loop_info = Vec::new();
                            // Restore the captured environment and push a new scope for function locals
                            env.current_scope = captured_scope;
                            env.push_scope(&mut self.mem);
                            pc = new_pc as usize;
                            continue;
                        }
                    }
                    pc += 2;
                }
                val if val == ByteCode::StartScope as u8 => {
                    env.push_scope(&mut self.mem);
                }
                val if val == ByteCode::EndScope as u8 => {
                    env.pop_scope(&self.mem)?;
                }
                val if val == ByteCode::Return as u8 => {
                    if stack_frame.is_empty() {
                        // We're assuming that we were called to run just a
                        // particular function

                        // There should be a single value on that stack that we return
                        if stack.len() != 1 {
                            return Err(format!("Expected one element on stack: {stack:?}"));
                        }

                        // TODO: we should supply `pc` as a `&mut usize`, but
                        // that requires changing far too much code here that
                        // works with `pc` as a value.
                        *mod_pc = pc;
                        return Ok(stack[0]);
                    }

                    // The value at the top of the stack is the return value of
                    // the function, so we just need to pop the PC
                    let scope_count;
                    let caller_scope;
                    (
                        pc,
                        loop_info,
                        scope_count,
                        caller_scope,
                        fn_optional_param_names,
                        fn_optional_param_name_idx,
                    ) = stack_frame.pop().expect("stack frame to return to");
                    while env.scope_depth(&self.mem) > scope_count {
                        env.pop_scope(&self.mem).unwrap();
                    }
                    // Restore the caller's environment scope
                    env.current_scope = caller_scope;
                    continue;
                }
                val if val == ByteCode::JmpUp as u8 => {
                    let new_pc = pc - (((bytes[pc + 1] as usize) << 8) + bytes[pc + 2] as usize);
                    pc = new_pc;
                    continue;
                }
                val if val == ByteCode::Jmp as u8 => {
                    // TODO: signed jumps
                    let new_pc = pc + ((bytes[pc + 1] as usize) << 8) + bytes[pc + 2] as usize;
                    pc = new_pc;
                    continue;
                }
                val if val == ByteCode::JmpNZ as u8 => {
                    let conditional = stack.pop().expect("JMPNZ val to check");
                    if conditional.is_truthy(self)? {
                        // TODO: signed jumps
                        let new_pc = pc + ((bytes[pc + 1] as usize) << 8) + bytes[pc + 2] as usize;
                        pc = new_pc;
                        continue;
                    }
                    pc += 2;
                }
                val if val == ByteCode::JmpZ as u8 => {
                    let conditional = stack.pop().expect("JMP val to check");
                    if !conditional.is_truthy(self)? {
                        // TODO: signed jumps
                        let new_pc = pc + ((bytes[pc + 1] as usize) << 8) + bytes[pc + 2] as usize;
                        pc = new_pc;
                        continue;
                    }
                    pc += 2;
                }
                val if val == ByteCode::CreateList as u8 => {
                    let len = ((bytes[pc + 1] as usize) << 8) + bytes[pc + 2] as usize;

                    let lst_val = self.mem.alloc_list();
                    let lst = lst_val.list_mut(self)?;
                    for item in stack.drain(stack.len() - len..) {
                        lst.push(&mut self.mem, item);
                    }

                    stack.push(lst_val);

                    pc += 2;
                }
                val if val == ByteCode::CreateFn as u8 => {
                    let instruction_offset = ((bytes[pc + 1] as u32) << 8) + bytes[pc + 2] as u32;
                    let fn_pc = pc as u32 - instruction_offset;
                    // Use descriptive name for anonymous functions
                    // Capture the current environment scope
                    let fn_val = self.mem.alloc_fn(fn_pc, b"<anonymous>", env.current_scope);
                    stack.push(fn_val);
                    pc += 2;
                }
                val if val == ByteCode::CreateStruct as u8 => {
                    // Everything after the first two bytes is data for the
                    // struct definition.
                    let new_pc = pc + ((bytes[pc + 1] as usize) << 8) + bytes[pc + 2] as usize;

                    let member_count = bytes[pc + 3];
                    let method_count = bytes[pc + 4];

                    let mut idx = pc + 5;
                    
                    // Read struct name
                    let name_len = bytes[idx];
                    let name = bytes[idx + 1..idx + 1 + name_len as usize].to_vec();
                    idx = idx + 1 + name_len as usize;

                    let mut struct_table = Vec::new();

                    for member_idx in 0..member_count {
                        let ident_len = bytes[idx];
                        let ident = &bytes[idx + 1..idx + 1 + ident_len as usize];
                        struct_table.push((
                            ident.to_vec(),
                            StructAttribute::MemberInstanceOffset(member_idx),
                        ));
                        idx = idx + 1 + ident_len as usize;
                    }

                    for _ in 0..method_count {
                        let method_pc = pc + ((bytes[idx] as usize) << 8) + bytes[idx + 1] as usize;

                        idx += 2;

                        let ident_len = bytes[idx];
                        let ident = &bytes[idx + 1..idx + 1 + ident_len as usize];
                        
                        // Allocate a function object for this method
                        // Methods capture the environment where the struct is defined
                        let fn_val = self.mem.alloc_fn(method_pc as u32, ident, env.current_scope);
                        let fn_pos = match fn_val {
                            ShimValue::Fn(pos) => pos,
                            _ => panic!("alloc_fn should return Fn"),
                        };
                        
                        struct_table.push((
                            ident.to_vec(),
                            StructAttribute::MethodDef(fn_pos),
                        ));
                        idx = idx + 1 + ident_len as usize;
                    }
                    const _: () = {
                        assert!(std::mem::size_of::<StructDef>() == 56);
                    };
                    let pos = alloc!(
                        self.mem,
                        Word(7.into()),
                        &format!("ByteCode::CreateStruct def PC {pc}")
                    );

                    unsafe {
                        let ptr: *mut StructDef =
                            std::mem::transmute(&mut self.mem.mem[usize::from(pos.0)]);
                        ptr.write(StructDef {
                            name,
                            member_count,
                            lookup: struct_table,
                        });
                    }

                    // Then push the struct definition to the stack
                    stack.push(ShimValue::StructDef(pos));

                    pc = new_pc;
                    continue;
                }
                b => {
                    print_asm(bytes);
                    return Err(format!("Unknown bytecode {b} at PC {pc}"));
                }
            }
            pc += 1;
        }

        *mod_pc = pc;
        if stack.len() > 0 {
            Ok(stack.pop().unwrap())
        } else {
            Ok(ShimValue::Uninitialized)
        }
    }
}

pub fn print_asm(bytes: &[u8]) {
    println!("{}", format_asm(bytes));
}

pub fn format_asm(bytes: &[u8]) -> String {
    let mut out = String::new();

    let mut idx = 0;
    while idx < bytes.len() {
        let b = &bytes[idx];
        let start_idx = idx;
        
        out.push_str(&format!("{start_idx:4}:  "));

        if *b == ByteCode::Jmp as u8 {
            let offset = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            let target = idx + offset;
            out.push_str(&format!("JMP -> {}", target));
            idx += 2;
        } else if *b == ByteCode::VariableDeclaration as u8 {
            let len = bytes[idx + 1] as usize;
            let slice = &bytes[idx + 2..idx + 2 + len];
            out.push_str(&format!(r#"let "{}""#, debug_u8s(slice)));
            idx += len + 1;
        } else if *b == ByteCode::NoOp as u8 {
            out.push_str("no-op");
        } else if *b == ByteCode::Pop as u8 {
            out.push_str("pop");
        } else if *b == ByteCode::Assignment as u8 {
            let len = bytes[idx + 1] as usize;
            let slice = &bytes[idx + 2..idx + 2 + len];
            out.push_str(&format!(r#"assign "{}""#, debug_u8s(slice)));
            idx += len + 1;
        } else if *b == ByteCode::Call as u8 {
            let arg_count = bytes[idx+1] as usize;
            let kwarg_count = bytes[idx+2] as usize;
            out.push_str(&format!("call args={}  kwargs={}", arg_count, kwarg_count));
            idx += 2;
        } else if *b == ByteCode::Not as u8 {
            out.push_str("Not");
        } else if *b == ByteCode::GT as u8 {
            out.push_str("GT");
        } else if *b == ByteCode::GTE as u8 {
            out.push_str("GTE");
        } else if *b == ByteCode::LT as u8 {
            out.push_str("LT");
        } else if *b == ByteCode::LTE as u8 {
            out.push_str("LTE");
        } else if *b == ByteCode::In as u8 {
            out.push_str("In");
        } else if *b == ByteCode::Negate as u8 {
            out.push_str("negate");
        } else if *b == ByteCode::Index as u8 {
            out.push_str("index");
        } else if *b == ByteCode::SetIndex as u8 {
            out.push_str("set_index");
        } else if *b == ByteCode::Add as u8 {
            out.push_str("add");
        } else if *b == ByteCode::Sub as u8 {
            out.push_str("sub");
        } else if *b == ByteCode::Multiply as u8 {
            out.push_str("multiply");
        } else if *b == ByteCode::Divide as u8 {
            out.push_str("divide");
        } else if *b == ByteCode::Modulus as u8 {
            out.push_str("modulus");
        } else if *b == ByteCode::Equal as u8 {
            out.push_str("equal");
        } else if *b == ByteCode::NotEqual as u8 {
            out.push_str("not_equal");
        } else if *b == ByteCode::JmpZ as u8 {
            let offset = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            let target = idx + offset;
            out.push_str(&format!("JMPZ -> {}", target));
            idx += 2;
        } else if *b == ByteCode::JmpNZ as u8 {
            let offset = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            let target = idx + offset;
            out.push_str(&format!("JMPNZ -> {}", target));
            idx += 2;
        } else if *b == ByteCode::JmpUp as u8 {
            let offset = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            let target = idx.saturating_sub(offset);
            out.push_str(&format!("JMPUP -> {}", target));
            idx += 2;
        } else if *b == ByteCode::JmpInitArg as u8 {
            let offset = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            let target = idx + offset;
            out.push_str(&format!("JmpInitArg -> {}", target));
            idx += 2;
        } else if *b == ByteCode::UnpackArgs as u8 {
            let required_arg_count = bytes[idx + 1] as usize;
            let optional_arg_count = bytes[idx + 2] as usize;
            
            let mut param_names = Vec::new();
            let mut param_idx = idx + 3;
            for _ in 0..(required_arg_count + optional_arg_count) {
                let len = bytes[param_idx] as usize;
                let param_name = &bytes[param_idx + 1..param_idx + 1 + len];
                param_names.push(debug_u8s(param_name).to_string());
                param_idx += 1 + len;
            }
            
            out.push_str(&format!("unpack_args required={} optional={} [{}]", 
                required_arg_count, optional_arg_count, param_names.join(", ")));
            idx = param_idx - 1;
        } else if *b == ByteCode::AssignArg as u8 {
            out.push_str("assign arg");
        } else if *b == ByteCode::CreateFn as u8 {
            let instruction_offset = ((bytes[idx + 1] as u16) << 8) + bytes[idx + 2] as u16;
            // The function points backwards by this offset
            let target_pc = idx.saturating_sub(instruction_offset as usize);
            out.push_str(&format!("CreateFn -> PC {}", target_pc));
            idx += 2;
        } else if *b == ByteCode::CreateStruct as u8 {
            let member_count = bytes[idx + 3];
            let method_count = bytes[idx + 4];
            
            let mut parse_idx = idx + 5;
            
            // Read struct name
            let name_len = bytes[parse_idx];
            let name = &bytes[parse_idx + 1..parse_idx + 1 + name_len as usize];
            parse_idx = parse_idx + 1 + name_len as usize;
            
            // Read member names
            let mut member_names = Vec::new();
            for _ in 0..member_count {
                let ident_len = bytes[parse_idx];
                let ident = &bytes[parse_idx + 1..parse_idx + 1 + ident_len as usize];
                member_names.push(debug_u8s(ident).to_string());
                parse_idx = parse_idx + 1 + ident_len as usize;
            }
            
            // Read method names and PCs
            let mut methods = Vec::new();
            for _ in 0..method_count {
                let method_pc = idx + ((bytes[parse_idx] as usize) << 8) + bytes[parse_idx + 1] as usize;
                parse_idx += 2;
                
                let ident_len = bytes[parse_idx];
                let ident = &bytes[parse_idx + 1..parse_idx + 1 + ident_len as usize];
                methods.push(format!("{}@{}", debug_u8s(ident), method_pc));
                parse_idx = parse_idx + 1 + ident_len as usize;
            }
            
            out.push_str(&format!("CreateStruct \"{}\" members=[{}] methods=[{}]", 
                                  debug_u8s(name), 
                                  member_names.join(", "),
                                  methods.join(", ")));
            
            // Skip to the end of the struct header (not the entire definition)
            // This allows the method bodies to be disassembled normally
            // Note: The outer loop will add 1, so we subtract 1 here
            idx = parse_idx - 1;
        } else if *b == ByteCode::GetAttr as u8 {
            let len = bytes[idx + 1] as usize;
            let slice = &bytes[idx + 2..idx + 2 + len];
            out.push_str(&format!("get .{}", debug_u8s(slice)));
            idx += len + 1;
        } else if *b == ByteCode::SetAttr as u8 {
            let len = bytes[idx + 1] as usize;
            let slice = &bytes[idx + 2..idx + 2 + len];
            out.push_str(&format!("set .{}", debug_u8s(slice)));
            idx += len + 1;
        } else if *b == ByteCode::VariableLoad as u8 {
            let len = bytes[idx+1] as usize;
            let slice = &bytes[idx+2..idx+2+len];
            out.push_str(&format!(r#"load "{}""#, debug_u8s(slice)));
            idx += len + 1;
        } else if *b == ByteCode::Break as u8 {
            out.push_str("break");
        } else if *b == ByteCode::Continue as u8 {
            out.push_str("continue");
        } else if *b == ByteCode::LiteralShimValue as u8 {
            let shim_bytes: [u8; 8] = bytes[idx + 1..idx + 9].try_into().unwrap();
            let val = ShimValue::from_bytes(shim_bytes);
            let val_str = match val {
                ShimValue::Integer(i) => format!("{}", i),
                ShimValue::Float(f) => format_float(f),
                ShimValue::Bool(true) => "true".to_string(),
                ShimValue::Bool(false) => "false".to_string(),
                ShimValue::None => "None".to_string(),
                ShimValue::Unit => "Unit".to_string(),
                _ => format!("{:?}", val),
            };
            out.push_str(&format!("ShimValue {}", val_str));
            idx += 8;
        } else if *b == ByteCode::LiteralString as u8 {
            let len = bytes[idx+1] as usize;
            let slice = &bytes[idx+2..idx+2+len];
            out.push_str(&format!(r#"String "{}""#, debug_u8s(slice)));
            idx += len + 1;
        } else if *b == ByteCode::LiteralNone as u8 {
            out.push_str("None");
        } else if *b == ByteCode::CreateList as u8 {
            let list_size = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            out.push_str(&format!("CreateList size={}", list_size));
            idx += 2;
        } else if *b == ByteCode::Copy as u8 {
            out.push_str("Copy");
        } else if *b == ByteCode::LoopStart as u8 {
            let offset = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            let target = idx + offset;
            out.push_str(&format!("Loop Start -> {}", target));
            idx += 2;
        } else if *b == ByteCode::LoopEnd as u8 {
            out.push_str("Loop End");
        } else if *b == ByteCode::Stringify as u8 {
            out.push_str("stringify");
        } else if *b == ByteCode::StartScope as u8 {
            out.push_str("start_scope");
        } else if *b == ByteCode::EndScope as u8 {
            out.push_str("end_scope");
        } else if *b == ByteCode::Return as u8 {
            out.push_str("return");
        } else if *b == ByteCode::Range as u8 {
            out.push_str("Range");
        } else {
            // Unformatted byte (including Pad0-Pad9) - show the decimal value
            out.push_str(&format!("{b:3}  "));
        }
        
        out.push('\n');
        idx += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u24_conversion() {
        assert_eq!(u24::from(1u32), u24([0, 0, 1]));
        assert_eq!(u32::from(u24::from(1u32)), 1u32);

        assert_eq!(u24::from(1u32).0, [0, 0, 1]);
    }

    #[test]
    fn scan_for_key_empty() {
        let bytes: &[u8] = &[];
        assert_eq!(scan_for_key(bytes, b"x"), None);
    }

    #[test]
    fn scan_for_key_single_entry() {
        // Entry: [3] "foo" [8 bytes value]
        let mut data = vec![3u8]; // len
        data.extend_from_slice(b"foo");
        data.extend_from_slice(&[0xAA; 8]); // value placeholder
        assert!(scan_for_key(&data, b"foo").is_some());
        assert_eq!(scan_for_key(&data, b"foo"), Some(4)); // offset of value
        assert_eq!(scan_for_key(&data, b"bar"), None);
    }

    #[test]
    fn scan_for_key_multiple_entries() {
        let mut data = Vec::new();
        // Entry 1: "ab" -> 8 bytes
        data.push(2u8);
        data.extend_from_slice(b"ab");
        data.extend_from_slice(&[0x11; 8]);
        // Entry 2: "cde" -> 8 bytes
        data.push(3u8);
        data.extend_from_slice(b"cde");
        data.extend_from_slice(&[0x22; 8]);

        assert_eq!(scan_for_key(&data, b"ab"), Some(3));
        // entry1 = 1+2+8 = 11 bytes, entry2: len at 11, key at 12..15, value at 15
        assert_eq!(scan_for_key(&data, b"cde"), Some(15));
        assert_eq!(scan_for_key(&data, b"xyz"), None);
    }

    fn test_interpreter() -> Interpreter {
        let config = Config::default();
        let program = Program {
            bytecode: Vec::new(),
            spans: Vec::new(),
            script: Vec::new(),
        };
        Interpreter::create(&config, program)
    }

    #[test]
    fn env_scope_insert_and_get() {
        let mut interpreter = test_interpreter();
        let mut env = Environment::new(&mut interpreter.mem);

        env.insert_new(&mut interpreter, b"x".to_vec(), ShimValue::Integer(42));
        let val = env.get(&mut interpreter, b"x");
        assert!(val.is_some());
        match val.unwrap() {
            ShimValue::Integer(42) => {},
            other => panic!("Expected Integer(42), got {:?}", other),
        }
    }

    #[test]
    fn env_scope_update() {
        let mut interpreter = test_interpreter();
        let mut env = Environment::new(&mut interpreter.mem);

        env.insert_new(&mut interpreter, b"y".to_vec(), ShimValue::Integer(1));
        env.update(&mut interpreter, b"y", ShimValue::Integer(99)).unwrap();
        match env.get(&mut interpreter, b"y").unwrap() {
            ShimValue::Integer(99) => {},
            other => panic!("Expected Integer(99), got {:?}", other),
        }
    }

    #[test]
    fn env_scope_parent_lookup() {
        let mut interpreter = test_interpreter();
        let mut env = Environment::new(&mut interpreter.mem);

        env.insert_new(&mut interpreter, b"root_var".to_vec(), ShimValue::Integer(10));
        env.push_scope(&mut interpreter.mem);
        env.insert_new(&mut interpreter, b"child_var".to_vec(), ShimValue::Integer(20));

        // Can see child var
        match env.get(&mut interpreter, b"child_var").unwrap() {
            ShimValue::Integer(20) => {},
            other => panic!("Expected Integer(20), got {:?}", other),
        }
        // Can see parent var through scope chain
        match env.get(&mut interpreter, b"root_var").unwrap() {
            ShimValue::Integer(10) => {},
            other => panic!("Expected Integer(10), got {:?}", other),
        }

        // Pop scope and child var is gone
        env.pop_scope(&interpreter.mem).unwrap();
        assert!(env.get(&mut interpreter, b"child_var").is_none());
        match env.get(&mut interpreter, b"root_var").unwrap() {
            ShimValue::Integer(10) => {},
            other => panic!("Expected Integer(10), got {:?}", other),
        }
    }

    #[test]
    fn env_scope_grow() {
        let mut interpreter = test_interpreter();
        let mut env = Environment::new(&mut interpreter.mem);

        // Insert enough variables to force at least one grow
        for i in 0..20u8 {
            let name = format!("var_{}", i);
            env.insert_new(&mut interpreter, name.into_bytes(), ShimValue::Integer(i as i32));
        }
        // Verify all are retrievable
        for i in 0..20u8 {
            let name = format!("var_{}", i);
            match env.get(&mut interpreter, name.as_bytes()).unwrap() {
                ShimValue::Integer(v) if v == i as i32 => {},
                other => panic!("Expected Integer({}), got {:?}", i, other),
            }
        }
    }
}

/**
 *
 * Struct Bytecode Format
 *  - CreateStruct OpCode
 *    - Two byte relative jump to end of struct def
 *    - u8 member count
 *    - u8 method count
 *    - List of members
 *      - u8 len followed by that number of bytes for the ident
 *    - List of methods
 *      - u16 relative jump to method, u8 len, ident bytes
 *    - Method defs
 *
 * Struct Instance Data Format:
 *  - Header value that points to object metadata
 *    - Contains mapping of ident to member offset or method PC
 *  - Member 0
 *  - Member 1
 *  - ...
 *
 * Struct Metadata Format:
 *  - Just a list for now
 *    - Vec<(Vec<u8>, Offset | PC)>
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 */
const _TODO: u8 = 42;
