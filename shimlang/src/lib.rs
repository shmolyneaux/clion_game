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

#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    fn start() -> Self {
        Self { start: 0, end: 1 }
    }
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
    fn lazy(val: T) -> Self {
        Self {
            data: val,
            span: Span::start(),
        }
    }
}

// Now redefine your types using the wrapper
pub type ExprNode = Node<Expression>;
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
}

#[derive(Debug)]
pub enum BooleanOp {
    And(Box<ExprNode>, Box<ExprNode>),
    Or(Box<ExprNode>, Box<ExprNode>),
}

#[derive(Debug)]
pub struct Block {
    stmts: Vec<Statement>,
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

    // The `gutter_size` includes everything up to the first character of the line
    let gutter_size = script_lines.len().to_string().len() + 4;
    for (lineno, line_info) in script_lines.iter().enumerate() {
        let lineno = lineno + 1;

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
            // Last line?
            out.push_str("\n");
        }

        let span_start_idx = span.start;
        let span_end_idx = span.end;

        // TODO: handle token going across line breaks
        if line_info.start_idx <= span_start_idx && span_start_idx <= line_info.end_idx {
            let line_span_start = span_start_idx - line_info.start_idx;
            let line_span_end = span_end_idx - line_info.start_idx;

            out.push_str(&" ".repeat(gutter_size));
            out.push_str(&" ".repeat(line_span_start as usize));
            out.push_str(&"^".repeat((line_span_end - line_span_start) as usize));
            out.push('\n');
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
                                            Node::lazy(expr)
                                        ),
                                        Box::new(
                                            Node::lazy(
                                                Expression::Stringify(Box::new(interp_expr))
                                            )
                                        ),
                                    )
                                );
                                expr = Expression::BinaryOp(
                                    BinaryOp::Add(
                                        Box::new(Node::lazy(expr)),
                                        Box::new(Node::lazy(Expression::Primary(Primary::String(s)))),
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
                tokens.unadvance()?;
                let end_span = tokens.peek_span()?;
                tokens.advance()?;
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
    let mut expr = parse_equality(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match tokens.peek()? {
            Token::And => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BooleanOp(BooleanOp::And(
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

pub fn parse_conditional(tokens: &mut TokenStream) -> Result<Conditional, String> {
    tokens.consume(Token::If)?;
    let conditional = parse_expression(tokens)?;

    let if_body = parse_block(tokens)?;
    let else_body = if *tokens.peek()? == Token::Else {
        tokens.advance()?;
        // TODO: implement `else if`
        parse_block(tokens)?
    } else {
        Block {
            stmts: Vec::new(),
            last_expr: None,
        }
    };

    Ok(Conditional::new(conditional, if_body, else_body))
}

pub fn parse_expression(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    match *tokens.peek()? {
        Token::If => {
            let cond = parse_conditional(tokens)?;
            Ok(ExprNode {
                data: Expression::If(Box::new(cond.conditional), cond.if_body, cond.else_body),
                span: Span::start(),
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

            Statement::Let(ident, expr)
        } else if *tokens.peek()? == Token::Fn {
            Statement::Fn(parse_function(tokens)?)
        } else if *tokens.peek()? == Token::If {
            let cond = parse_conditional(tokens)?;

            // Do we treat this as an expression or statement?
            if *tokens.peek()? == Token::RCurly {
                let expr = Expression::If(Box::new(cond.conditional), cond.if_body, cond.else_body);
                last_expr = Some(Box::new(ExprNode {
                    data: expr,
                    span: Span::start(),
                }));
                break;
            } else {
                Statement::If(cond.conditional, cond.if_body, cond.else_body)
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

            Statement::For(ident, expr, body)
        } else if *tokens.peek()? == Token::While {
            tokens.advance()?;
            let conditional = parse_expression(tokens)?;
            let loop_body = parse_block(tokens)?;

            Statement::While(conditional, loop_body)
        } else if *tokens.peek()? == Token::Break {
            tokens.advance()?;
            tokens.consume(Token::Semicolon)?;
            Statement::Break
        } else if *tokens.peek()? == Token::Continue {
            tokens.advance()?;
            tokens.consume(Token::Semicolon)?;
            Statement::Continue
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
            tokens.consume(Token::RCurly)?;

            Statement::Struct(Struct {
                ident,
                members_required: members,
                members_optional: optional_members,
                methods,
            })
        } else if *tokens.peek()? == Token::Return {
            tokens.advance()?;
            if *tokens.peek()? == Token::Semicolon {
                tokens.advance()?;
                Statement::Return(None)
            } else {
                let expr = parse_expression(tokens)?;
                match tokens.pop()? {
                    Token::Semicolon => (),
                    token => {
                        tokens.unadvance()?;
                        return Err(tokens.format_peek_err(&format!(
                            "Expected semicolon after `return <expr>`, found {:?}",
                            token
                        )));
                    }
                }
                Statement::Return(Some(expr))
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
                    tokens.pop()?;
                    Statement::Expression(expr)
                }
                Token::Equal => {
                    tokens.pop()?;
                    match expr.data {
                        Expression::Primary(Primary::Identifier(ident)) => {
                            let expr_to_assign = parse_expression(tokens)?;
                            tokens.consume(Token::Semicolon)?;
                            Statement::Assignment(ident.clone(), expr_to_assign)
                        }
                        Expression::Attribute(expr, ident) => {
                            let expr_to_assign = parse_expression(tokens)?;
                            tokens.consume(Token::Semicolon)?;
                            Statement::AttributeAssignment(*expr, ident.clone(), expr_to_assign)
                        }
                        Expression::Index(expr, index_expr) => {
                            let expr_to_assign = parse_expression(tokens)?;
                            tokens.consume(Token::Semicolon)?;
                            Statement::IndexAssignment(*expr, *index_expr, expr_to_assign)
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
            b'!' => tokens.push(Token::Bang),
            b'.' => tokens.push(Token::Dot),
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
#[derive(Debug)]
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

    fn end(&self) -> Word {
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
        let free_list = vec![FreeBlock::new(Word(0.into()), word_count)];
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

    fn alloc_str(&mut self, contents: &[u8]) -> ShimValue {
        // Length + contents + padding
        let total_len = 1 + contents.len().div_ceil(8);
        let word_count = Word(total_len.into());
        let position = alloc!(self, word_count, &format!("str `{}`", debug_u8s(contents)));

        self.mem[usize::from(position.0)] = contents.len() as u64;

        let bytes: &mut [u8] = unsafe {
            let u64_slice = &mut self.mem[
                (1+usize::from(position.0))..
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

        ShimValue::String(position)
    }

    fn alloc_dict(&mut self) -> ShimValue {
        let word_count = Word((std::mem::size_of::<NewShimDict>() as u32).div_ceil(8).into());
        let position = alloc!(self, word_count, "Dict");
        unsafe {
            let ptr: *mut NewShimDict =
                std::mem::transmute(&mut self.mem[usize::from(position.0)]);
            ptr.write(NewShimDict::new());
        }
        ShimValue::Dict(position)
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

#[derive(Debug)]
pub struct Environment {
    env_chain: Vec<HashMap<Vec<u8>, u64>>,
}

impl Environment {
    pub fn new() -> Self {
        Self {
            env_chain: vec![HashMap::new()],
        }
    }

    pub fn new_with_builtins(mem: &mut MMU) -> Self {
        let mut env = Self::new();
        let builtins: &[(&[u8], Box<NativeFn>)] = &[
            (b"print", Box::new(shim_print)),
            (b"panic", Box::new(shim_panic)),
            (b"dict", Box::new(shim_dict)),
            (b"assert", Box::new(shim_assert)),
            (b"str", Box::new(shim_str)),
            (b"int", Box::new(shim_int)),
            (b"float", Box::new(shim_float)),
            (b"try_int", Box::new(shim_try_int)),
            (b"try_float", Box::new(shim_try_float)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
            (b"__PLACEHOLDER", Box::new(shim_print)),
        ];

        for (name, func) in builtins {
            let position = alloc!(mem, Word(1.into()), &format!("builtin func {}", debug_u8s(name)));
            unsafe {
                let ptr: *mut NativeFn = std::mem::transmute(&mut mem.mem[usize::from(position.0)]);
                ptr.write(**func);
            }

            env.insert_new(name.to_vec(), ShimValue::NativeFn(position));
        }

        env
    }


    fn insert_new(&mut self, key: Vec<u8>, val: ShimValue) {
        let idx = self.env_chain.len() - 1;
        self.env_chain[idx].insert(key, val.to_u64());
    }

    fn update(&mut self, key: &[u8], val: ShimValue) -> Result<(), String> {
        for env in self.env_chain.iter_mut().rev() {
            if let Some(entry_val) = env.get_mut(key) {
                *entry_val = val.to_u64();
                return Ok(());
            }
        }

        Err(format!("Key {:?} not found in environment", key))
    }

    fn get(&self, key: &[u8]) -> Option<ShimValue> {
        for env in self.env_chain.iter().rev() {
            if env.contains_key(key) {
                unsafe {
                    return Some(ShimValue::from_u64(*env.get(key).unwrap()));
                }
            }
        }
        None
    }

    fn contains_key(&self, key: &[u8]) -> bool {
        for env in self.env_chain.iter().rev() {
            if env.contains_key(key) {
                return true;
            }
        }
        false
    }

    fn push_scope(&mut self) {
        self.env_chain.push(HashMap::new());
    }

    fn pop_scope(&mut self) -> Result<(), String> {
        match self.env_chain.pop() {
            Some(_) => Ok(()),
            None => {
                return Err(format!("Ran out of scopes to pop!"));
            }
        }
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
    // This is a program counter, TODO should be a memory position?
    Fn(u32),
    BoundMethod(
        // Object
        Word,
        // Fn, TODO: should be a memory position instead of PC
        u32,
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
    String(Word),
    List(Word),
    Dict(Word),
    StructDef(Word),
    Struct(Word),
    Native(Word),
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
    MethodDefPC(u32),
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

struct NewShimDict {
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

impl NewShimDict {
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
        let old_size = self.index_size();
        let old_capacity = self.capacity();
        self.size_pow = if old_size == 0 {
            3
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
            x if x <= (u8::MAX as usize) => TypedIndices::U8(
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
        ((self.index_size() * 2) / 3) as usize
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
            None => panic!("Could not find free slot"),
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
    PC(u32),
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

fn shim_print(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
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
    let _lst = obj.list(interpreter)?;
    let _key = unpacker.optional(b"key");
    unpacker.end()?;

    todo!();
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
                CallResult::PC(pc) => {
                    let val = interpreter.execute_bytecode_extended(
                        &mut (pc as usize),
                        args,
                        // TODO: this doesn't even have print...
                        &mut Environment::new(),
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
            CallResult::PC(pc) => {
                let val = interpreter.execute_bytecode_extended(
                    &mut (pc as usize),
                    args,
                    // TODO: this doesn't even have print...
                    &mut Environment::new(),
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
        ShimValue::String(_) => "string",
        ShimValue::List(_) => "list",
        ShimValue::Dict(_) => "dict",
        ShimValue::StructDef(_) => "struct definition",
        ShimValue::Struct(_) => "struct",
        ShimValue::Native(_) => "native object",
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
        ShimValue::String(_) => {
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
        ShimValue::String(_) => {
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
        ShimValue::String(_) => {
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
        ShimValue::String(_) => {
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

    fn hash(&self, interpreter: &mut Interpreter) -> Result<u32, String> {
        let hashcode: u64 = match self {
            ShimValue::Integer(i) => fnv1a_hash(&i.to_be_bytes()),
            ShimValue::Float(f) => fnv1a_hash(&f.to_be_bytes()),
            ShimValue::String(_) => {
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
            ShimValue::Fn(pc) => Ok(CallResult::PC(*pc)),
            ShimValue::BoundMethod(pos, pc) => {
                // push struct pos to start of arg list then return the pc of the method
                args.args.insert(0, ShimValue::Struct(*pos));
                Ok(CallResult::PC(*pc))
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
                    if let Some(StructAttribute::MethodDefPC(pc)) = struct_def.find(b"__init__") {
                        return Ok(CallResult::PC(pc));
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

    fn dict_mut(&self, interpreter: &mut Interpreter) -> Result<&mut NewShimDict, String> {
        match self {
            ShimValue::Dict(position) => {
                let dict: &mut NewShimDict = unsafe {
                    std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)])
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
            ShimValue::String(position) => {
                let len = interpreter.mem.mem[usize::from(position.0)];
                let total_len: usize = 1 + len.div_ceil(8) as usize;

                let bytes: &[u8] = unsafe {
                    let u64_slice = &interpreter.mem.mem[
                        (1+usize::from(position.0))..
                        (usize::from(position.0)+total_len)
                    ];
                    std::slice::from_raw_parts(
                        u64_slice.as_ptr() as *const u8,
                        len as usize,
                    )
                };
                Ok(bytes)
            },
            _ => {
                Err(format!("Not a string"))
            }
        }
    }

    fn index(&self, interpreter: &mut Interpreter, index: &ShimValue) -> Result<ShimValue, String> {
        match (self, index) {
            (ShimValue::String(_), ShimValue::Integer(index)) => {
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
                let dict: &mut NewShimDict = unsafe {
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
            ShimValue::String(_) => {
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
            ShimValue::String(_) => {
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
            (a @ ShimValue::String(_), b @ ShimValue::String(_)) => {
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
            (a @ ShimValue::String(_), b @ ShimValue::String(_)) => {
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
            (ShimValue::String(_), ShimValue::String(_)) => {
                Ok(ShimValue::Bool(self.string(interpreter)? > other.string(interpreter)?))
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
            (ShimValue::String(_), ShimValue::String(_)) => {
                Ok(ShimValue::Bool(self.string(interpreter)? >= other.string(interpreter)?))
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
            (ShimValue::String(_), ShimValue::String(_)) => {
                Ok(ShimValue::Bool(self.string(interpreter)? < other.string(interpreter)?))
            },
            (ShimValue::None, ShimValue::None) => Ok(ShimValue::Bool(false)),
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
            (ShimValue::String(_), ShimValue::String(_)) => {
                Ok(ShimValue::Bool(self.string(interpreter)? <= other.string(interpreter)?))
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
                let dict: &mut NewShimDict = unsafe {
                    let ptr: &mut NewShimDict =
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
                                StructAttribute::MethodDefPC(pc) => {
                                    Ok(ShimValue::BoundMethod(*pos, *pc))
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
                                StructAttribute::MethodDefPC(pc) => {
                                    // Return the method
                                    Ok(ShimValue::Fn(*pc))
                                }
                            };
                        }
                    }
                }
                Err(format!("Ident {:?} not found for {:?}", debug_u8s(ident), self))
            }
            ShimValue::String(_) => {
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
                                StructAttribute::MethodDefPC(_) => Err(format!(
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
}

pub struct Program {
    pub bytecode: Vec<u8>,
    spans: Vec<Span>,
    script: Vec<u8>,
}

pub fn compile_ast(ast: &Ast) -> Result<Program, String> {
    let mut program = Vec::new();
    compile_block_inner(&ast.block, true, &mut program)?;
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
) -> Result<Vec<(u8, Span)>, String> {

    let mut asm = Vec::new();
    asm.push((ByteCode::UnpackArgs as u8, Span { start: 0, end: 1 }));
    asm.push((
        pos_args_required.len() as u8,
        Span { start: 0, end: 1 },
    ));
    asm.push((
        pos_args_optional.len() as u8,
        Span { start: 0, end: 1 },
    ));

    for param in pos_args_required.iter() {
        asm.push((
            param.len().try_into().expect("Param len should into u8"),
            Span { start: 0, end: 1 },
        ));
        for b in param {
            asm.push((*b, Span { start: 0, end: 1 }));
        }
    }

    for (param, _) in pos_args_optional.iter() {
        asm.push((
            param.len().try_into().expect("Param len should into u8"),
            Span { start: 0, end: 1 },
        ));
        for b in param {
            asm.push((*b, Span { start: 0, end: 1 }));
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
        asm.extend(compile_return(&val)?);
    } else {
        let needs_implicit_return = if body.stmts.len() > 1 {
            match body.stmts[body.stmts.len() - 1] {
                Statement::Return(_) => false,
                _ => true,
            }
        } else {
            true
        };

        if needs_implicit_return {
            let expr = ExprNode {
                data: Expression::Primary(Primary::None),
                span: Span::start(),
            };
            let val: Option<&ExprNode> = Some(&expr);
            asm.extend(compile_return(&val)?);
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
) -> Result<Vec<(u8, Span)>, String> {
    // This will be replaced with a relative jump to after the function
    // declaration
    let mut asm = vec![
        (ByteCode::Jmp as u8, Span { start: 0, end: 1 }),
        (0, Span { start: 0, end: 1 }),
        (0, Span { start: 0, end: 1 }),
    ];
    asm.extend(
        compile_fn_body_inner(
            pos_args_required,
            pos_args_optional,
            body,
        )?
    );

    // Fix the jump offset at the function declaration now that we know
    // the size of the body
    let pc_offset = asm.len() as u16;
    asm[1].0 = (pc_offset >> 8) as u8;
    asm[2].0 = (pc_offset & 0xff) as u8;

    // Assign the value to the ident
    let pc_offset = asm.len() as u16 - 3;
    asm.push((ByteCode::CreateFn as u8, Span { start: 0, end: 1 }));
    asm.push(((pc_offset >> 8) as u8, Span { start: 0, end: 1 }));
    asm.push(((pc_offset & 0xff) as u8, Span { start: 0, end: 1 }));

    Ok(asm)
}

pub fn compile_fn(func: &Fn) -> Result<Vec<(u8, Span)>, String> {
    let ident = if let Some(ident) = &func.ident {
        ident
    } else {
        return Err(format!("No ident for function declaration!"));
    };

    let mut asm = compile_fn_expression(
        &func.pos_args_required,
        &func.pos_args_optional,
        &func.body,
    )?;

    asm.push((
        ByteCode::VariableDeclaration as u8,
        Span { start: 0, end: 1 },
    ));
    asm.push((
        ident
            .len()
            .try_into()
            .expect("Ident len should into u8"),
        Span { start: 0, end: 1 },
    ));
    for b in ident.iter() {
        asm.push((*b, Span { start: 0, end: 1 }));
    }

    Ok(asm)
}

pub fn compile_fn_body(func: &Fn) -> Result<Vec<(u8, Span)>, String> {
    compile_fn_body_inner(
        &func.pos_args_required,
        &func.pos_args_optional,
        &func.body,
    )
}

pub fn compile_return(expr: &Option<&ExprNode>) -> Result<Vec<(u8, Span)>, String> {
    let mut res = Vec::new();
    if let Some(expr) = expr {
        res.extend(compile_expression(expr)?);
    } else {
        res.push((ByteCode::LiteralNone as u8, Span { start: 0, end: 1 }));
    }
    res.push((ByteCode::Return as u8, Span { start: 0, end: 1 }));
    Ok(res)
}

pub fn compile_statement(stmt: &Statement) -> Result<Vec<(u8, Span)>, String> {
    match stmt {
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
            compile_fn(func)
        }
        Statement::Struct(Struct {
            ident,
            members_required,
            members_optional,
            methods,
        }) => {
            let mut asm = vec![
                (ByteCode::CreateStruct as u8, Span { start: 0, end: 1 }),
                (0, Span { start: 0, end: 1 }),
                (0, Span { start: 0, end: 1 }),
            ];
            asm.push(((members_required.len() + members_optional.len()) as u8, Span { start: 0, end: 1 }));

            // The +1 is for the constructor
            asm.push(((methods.len() + 1) as u8, Span { start: 0, end: 1 }));

            // Add struct name
            asm.push((
                ident
                    .len()
                    .try_into()
                    .expect("Struct name len should into u8"),
                Span { start: 0, end: 1 },
            ));
            for b in ident.into_iter() {
                asm.push((*b, Span { start: 0, end: 1 }));
            }

            let member_names: Vec<Vec<u8>> = members_required.iter().cloned().chain(members_optional.iter().map(|(x, _)| x.clone())).collect();

            for member in member_names.iter() {
                asm.push((
                    member
                        .len()
                        .try_into()
                        .expect("Member ident len should into u8"),
                    Span { start: 0, end: 1 },
                ));
                for b in member.into_iter() {
                    asm.push((*b, Span { start: 0, end: 1 }));
                }
            }

            let arg_list = member_names.into_iter()
                .map(|name| Node {
                        data: Expression::Primary(Primary::Identifier(name)),
                        span: Span::start(),
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
                                                    span: Span::start(),
                                                }
                                            ),
                                            arg_list,
                                            Vec::new()
                                        ),
                                        span: Span::start(),
                                    }
                                )
                            ),
                        }
                    )?
                )
            );
            for method in methods {
                let ident = if let Some(ident) = &method.ident {
                    ident
                } else {
                    return Err(format!("Method does not have ident!"));
                };
                method_defs.push((&ident, compile_fn_body(method)?));
            }

            let mut jump_asm_idx = Vec::new();
            for (ident, _method_def) in method_defs.iter() {
                jump_asm_idx.push(asm.len());
                asm.push((0, Span { start: 0, end: 1 }));
                asm.push((0, Span { start: 0, end: 1 }));
                asm.push((
                    ident
                        .len()
                        .try_into()
                        .expect("Method ident len should into u8"),
                    Span { start: 0, end: 1 },
                ));
                for b in ident.into_iter() {
                    asm.push((*b, Span { start: 0, end: 1 }));
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
                Span { start: 0, end: 1 },
            ));
            asm.push((
                ident.len().try_into().expect("Ident len should into u8"),
                Span { start: 0, end: 1 },
            ));

            for b in ident.into_iter() {
                asm.push((*b, Span { start: 0, end: 1 }));
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
                (ByteCode::LoopStart as u8, Span { start: 0, end: 1 }),
                (0, Span { start: 0, end: 1 }),
                (0, Span { start: 0, end: 1 }),
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
                Span { start: 0, end: 1 },
            ));
            asm.push((
                ident
                    .len()
                    .try_into()
                    .expect("For loop ident len should into u8"),
                Span { start: 0, end: 1 },
            ));
            for b in ident {
                asm.push((*b, Span { start: 0, end: 1 }));
            }

            asm.push((ByteCode::LiteralNone as u8, expr.span));
            asm.push((ByteCode::Equal as u8, expr.span));

            // Jump to `LoopEnd` if calling .next() returns None
            let none_check_idx = asm.len();
            asm.push((ByteCode::JmpNZ as u8, Span { start: 0, end: 0 }));
            asm.push((0, Span { start: 0, end: 0 }));
            asm.push((0, Span { start: 0, end: 0 }));

            asm.extend(compile_block(body, false)?);

            // Jump to start of loop to check the condition again
            let loop_start_offset = u16_to_u8s(asm.len() as u16 - loop_start_idx as u16 - 3);
            asm.push((ByteCode::JmpUp as u8, Span { start: 0, end: 0 }));
            asm.push((loop_start_offset[0], Span { start: 0, end: 0 }));
            asm.push((loop_start_offset[1], Span { start: 0, end: 0 }));

            // This is the offset from none_check_idx that will get us
            // out of the loop
            let pc_offset = u16_to_u8s(asm.len() as u16 - none_check_idx as u16);
            asm[none_check_idx + 1].0 = pc_offset[0];
            asm[none_check_idx + 2].0 = pc_offset[1];

            let loop_end = u16_to_u8s(asm.len() as u16 - loop_start_idx as u16);
            asm[loop_start_idx + 1].0 = loop_end[0];
            asm[loop_start_idx + 2].0 = loop_end[1];

            asm.push((ByteCode::LoopEnd as u8, Span { start: 0, end: 0 }));

            // Remove the `iter` method from the stack
            asm.push((ByteCode::Pop as u8, Span { start: 0, end: 0 }));

            Ok(asm)
        }
        Statement::While(conditional, body) => {
            let mut asm = vec![
                (ByteCode::LoopStart as u8, Span { start: 0, end: 1 }),
                (0, Span { start: 0, end: 1 }),
                (0, Span { start: 0, end: 1 }),
            ];
            asm.extend(compile_expression(conditional)?);

            // Jump to `LoopEnd` if the condition is falsy
            let conditional_check_idx = asm.len();
            asm.push((ByteCode::JmpZ as u8, Span { start: 0, end: 0 }));
            asm.push((0, Span { start: 0, end: 0 }));
            asm.push((0, Span { start: 0, end: 0 }));

            asm.extend(compile_block(body, false)?);

            // Jump to start of loop to check the condition again
            let loop_start_offset = u16_to_u8s(asm.len() as u16 - 3);
            asm.push((ByteCode::JmpUp as u8, Span { start: 0, end: 0 }));
            asm.push((loop_start_offset[0], Span { start: 0, end: 0 }));
            asm.push((loop_start_offset[1], Span { start: 0, end: 0 }));

            // This is the offset from conditional_check_idx that will get us
            // out of the loop
            let pc_offset = u16_to_u8s(asm.len() as u16 - conditional_check_idx as u16);
            asm[conditional_check_idx + 1].0 = pc_offset[0];
            asm[conditional_check_idx + 2].0 = pc_offset[1];

            let loop_end = u16_to_u8s(asm.len() as u16);
            asm[1].0 = loop_end[0];
            asm[2].0 = loop_end[1];

            asm.push((ByteCode::LoopEnd as u8, Span { start: 0, end: 0 }));

            Ok(asm)
        }
        Statement::Break => Ok(vec![(ByteCode::Break as u8, Span { start: 0, end: 1 })]),
        Statement::Continue => Ok(vec![(ByteCode::Continue as u8, Span { start: 0, end: 1 })]),
        Statement::Return(expr) => compile_return(&expr.as_ref()),
        Statement::If(conditional, if_body, else_body) => {
            compile_if(conditional, if_body, else_body, false)
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

pub fn compile_block_inner(block: &Block, is_expr: bool, asm: &mut Vec<(u8, Span)>) -> Result<(), String> {
    for stmt in block.stmts.iter() {
        asm.extend(compile_statement(&stmt)?);
    }
    if let Some(last_expr) = &block.last_expr {
        asm.extend(compile_expression(&last_expr)?);
    } else {
        asm.push((ByteCode::LiteralNone as u8, Span::start()));
    }
    if !is_expr {
        asm.push((ByteCode::Pop as u8, Span::start()));
    }

    Ok(())
}

pub fn compile_block(block: &Block, is_expr: bool) -> Result<Vec<(u8, Span)>, String> {
    let mut asm = Vec::new();

    asm.push((ByteCode::StartScope as u8, Span { start: 0, end: 1 }));
    compile_block_inner(block, is_expr, &mut asm)?;
    asm.push((ByteCode::EndScope as u8, Span { start: 0, end: 1 }));

    Ok(asm)
}

pub fn compile_if(
    conditional: &ExprNode,
    if_body: &Block,
    else_body: &Block,
    is_expr: bool,
) -> Result<Vec<(u8, Span)>, String> {
    let mut asm = compile_expression(conditional)?;
    let conditional_check_idx = asm.len();
    asm.push((ByteCode::JmpZ as u8, Span { start: 0, end: 0 }));
    asm.push((0, Span { start: 0, end: 0 }));
    asm.push((0, Span { start: 0, end: 0 }));
    asm.extend(compile_block(if_body, is_expr)?);
    asm.push((ByteCode::Jmp as u8, Span { start: 0, end: 0 }));
    asm.push((0, Span { start: 0, end: 0 }));
    asm.push((0, Span { start: 0, end: 0 }));
    // We jump to here when the condition is false
    let else_case_start_idx = asm.len();

    asm.extend(compile_block(else_body, is_expr)?);

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
            asm.push((ByteCode::Copy as u8, Span { start: 0, end: 0 }));

            let short_circuit_idx = asm.len();
            asm.push((ByteCode::JmpZ as u8, Span { start: 0, end: 0 }));
            asm.push((0, Span { start: 0, end: 0 }));
            asm.push((0, Span { start: 0, end: 0 }));

            // Since the result of a is truthy we get rid of it and return the result of b
            asm.push((ByteCode::Pop as u8, Span { start: 0, end: 0 }));

            asm.extend(compile_expression(&b)?);

            let short_circuit_offset = u16_to_u8s(asm.len() as u16 - short_circuit_idx as u16);
            asm[short_circuit_idx + 1].0 = short_circuit_offset[0];
            asm[short_circuit_idx + 2].0 = short_circuit_offset[1];

            Ok(asm)
        }
        Expression::BooleanOp(BooleanOp::Or(a, b)) => {
            let mut asm = compile_expression(&a)?;
            asm.push((ByteCode::Copy as u8, Span { start: 0, end: 0 }));

            let short_circuit_idx = asm.len();
            asm.push((ByteCode::JmpNZ as u8, Span { start: 0, end: 0 }));
            asm.push((0, Span { start: 0, end: 0 }));
            asm.push((0, Span { start: 0, end: 0 }));

            // Since the result of a is falsy we get rid of it and return the result of b
            asm.push((ByteCode::Pop as u8, Span { start: 0, end: 0 }));

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
        Expression::Block(block) => compile_block(block, true),
        Expression::If(conditional, if_body, else_body) => {
            compile_if(conditional, if_body, else_body, true)
        },
        Expression::Fn(func) => {
            compile_fn_expression(
                &func.pos_args_required,
                &func.pos_args_optional,
                &func.body,
            )
        },
    }
}

pub struct Bitmask {
    data: Vec<u64>,
    size: usize,
}

impl Bitmask {
    pub fn new(num_bits: usize) -> Self {
        // Round up if we don't have a number of bits that's cleanly divisible by 64
        let blocks = num_bits.div_ceil(64);

        Bitmask {
            data: vec![0; blocks],
            size: num_bits,
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
        let mut ranges = Vec::new();
        let mut start_of_run: Option<usize> = None;

        for i in 0..self.size {
            let is_zero = !self.is_set(i);

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
        }

        if let Some(start) = start_of_run {
            ranges.push(start..self.size);
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
        let mask = Bitmask::new(last_block_start.into());
        Self {
            mem,
            mask,
        }
    }

    fn mark(&mut self, mut vals: Vec<ShimValue>) {
        unsafe {
            while !vals.is_empty() {
                match vals.pop().unwrap() {
                    ShimValue::Integer(_) | ShimValue::Float(_) | ShimValue::Bool(_) | ShimValue::Unit | ShimValue::None | ShimValue::Fn(_) | ShimValue::Uninitialized => (),
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
                    ShimValue::String(pos) => {
                        let pos: usize = pos.into();
                        let len = self.mem.mem[pos] as usize;
                        // TODO: check this...
                        for idx in pos..(pos + 1 + len.div_ceil(8)) {
                            self.mask.set(idx);
                        }
                    },
                    ShimValue::Dict(pos) => {
                        let pos: usize = pos.into();
                        let dict: &NewShimDict = std::mem::transmute(&self.mem.mem[pos]);
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
                        for idx in pos..(pos + (std::mem::size_of::<NewShimDict>()/8)) {
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
                    ShimValue::BoundMethod(pos, _pc) => {
                        let val = ShimValue::Struct(pos);
                        vals.push(val);
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
                }
            }
        }
    }

    fn sweep(&mut self) {
        for block in self.mask.find_zeros() {
            self.mem.free(block.start.into(), (block.end-block.start).into());
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
        for (idx, scope) in env.env_chain.iter().enumerate() {
            println!("Scope {idx}");
            for (ident, bytes) in scope.iter() {
                let val = unsafe { ShimValue::from_u64(*bytes) };
                println!("{:>12}: {:?}", debug_u8s(ident), val);
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
                                    StructAttribute::MethodDefPC(_) => (),
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
                                    StructAttribute::MethodDefPC(_) => {
                                        println!("                - {}()", debug_u8s(&attr));
                                    }
                                };
                            }
                        }
                    },
                    _ => (),
                }
            }
            println!();
        }
    }

    pub fn gc(&mut self, env: &Environment) {
        self.print_mem();
        self.print_env(env);
        let mut gc = GC::new(&mut self.mem);
        let mut roots: Vec<ShimValue> = Vec::new();
        for scope in env.env_chain.iter() {
            for (_ident, bytes) in scope.iter() {
                roots.push(unsafe { ShimValue::from_u64(*bytes) });
            }
        }
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
        let mut pc = *mod_pc;
        // These are values that are operated on. Expressions push and pop to
        // this stack, return values go on this stack etc.
        let mut stack: Vec<ShimValue> = Vec::new();


        // This is the (PC, loop_info, scope_count, fn_optional_param_names,
        // fn_optional_param_name_idx) call stack
        let mut stack_frame: Vec<(
            // PC
            usize,
            // loop_info
            Vec<(usize, usize, usize)>,
            // scope_count
            usize,
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
                        CallResult::PC(new_pc) => {
                            stack_frame.push((
                                pc + 1,
                                loop_info.clone(),
                                env.env_chain.len(),
                                fn_optional_param_names.clone(),
                                fn_optional_param_name_idx,
                            ));
                            loop_info = Vec::new();
                            env.push_scope();
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
                    loop_info.push((pc + 3, loop_end, env.env_chain.len()));
                    pc += 2;
                }
                val if val == ByteCode::LoopEnd as u8 => {
                    loop_info.pop().expect("loop end should have loop info");
                }
                val if val == ByteCode::Break as u8 => {
                    let (_, end_pc, scope_count) =
                        loop_info.last().expect("break should have loop info");
                    while env.env_chain.len() > *scope_count {
                        env.pop_scope().unwrap();
                    }
                    pc = *end_pc;
                    continue;
                }
                val if val == ByteCode::Continue as u8 => {
                    let (start_pc, _, scope_count) =
                        loop_info.last().expect("continue should have loop info");
                    while env.env_chain.len() > *scope_count {
                        env.pop_scope().unwrap();
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
                            env.insert_new(param_name.to_vec(), val);
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
                            env.insert_new(param_name.to_vec(), val);
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

                    match env.get(optional_param_name) {
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
                    env.update(optional_param_name, stack.pop().unwrap())?;
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
                    env.insert_new(ident.to_vec(), val);
                    pc += 1 + ident_len;
                }
                val if val == ByteCode::Assignment as u8 => {
                    let val = stack.pop().expect("Value for assignment");
                    let ident_len = bytes[pc + 1] as usize;
                    let ident = &bytes[pc + 2..pc + 2 + ident_len as usize];

                    if !env.contains_key(ident) {
                        return Err(format_script_err(
                            self.program.spans[pc],
                            &self.program.script,
                            &format!("Identifier {:?} not found", ident),
                        ));
                    }
                    env.update(ident, val)?;

                    pc += 1 + ident_len;
                }
                val if val == ByteCode::VariableLoad as u8 => {
                    let ident_len = bytes[pc + 1] as usize;
                    let ident = &bytes[pc + 2..pc + 2 + ident_len as usize];
                    if let Some(value) = env.get(ident) {
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
                            val @ ShimValue::String(_position) => {
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
                        CallResult::PC(new_pc) => {
                            stack_frame.push((
                                pc + 3,
                                loop_info.clone(),
                                env.env_chain.len(),
                                fn_optional_param_names.clone(),
                                fn_optional_param_name_idx,
                            ));
                            loop_info = Vec::new();
                            env.push_scope();
                            pc = new_pc as usize;
                            continue;
                        }
                    }
                    pc += 2;
                }
                val if val == ByteCode::StartScope as u8 => {
                    env.push_scope();
                }
                val if val == ByteCode::EndScope as u8 => {
                    env.pop_scope()?;
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
                    (
                        pc,
                        loop_info,
                        scope_count,
                        fn_optional_param_names,
                        fn_optional_param_name_idx,
                    ) = stack_frame.pop().expect("stack frame to return to");
                    while env.env_chain.len() > scope_count {
                        env.pop_scope().unwrap();
                    }
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
                    stack.push(ShimValue::Fn(pc as u32 - instruction_offset));
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
                        struct_table.push((
                            ident.to_vec(),
                            StructAttribute::MethodDefPC(method_pc as u32),
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
                            name: name,
                            member_count: member_count,
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
        out.push_str(&format!("{idx:4}:  {b:3}  "));

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
            out.push_str(&format!("no-op"));
        } else if *b == ByteCode::Pop as u8 {
            out.push_str(&format!("pop"));
        } else if *b == ByteCode::Assignment as u8 {
            out.push_str(&format!("assignment"));
        } else if *b == ByteCode::Call as u8 {
            let arg_count = bytes[idx+1] as usize;
            let kwarg_count = bytes[idx+2] as usize;
            out.push_str(&format!("call args={}  kwargs={}", arg_count, kwarg_count));
            idx += 2;
        } else if *b == ByteCode::Not as u8 {
            out.push_str(&format!("Not"));
        } else if *b == ByteCode::GT as u8 {
            out.push_str(&format!("GT"));
        } else if *b == ByteCode::GTE as u8 {
            out.push_str(&format!("GTE"));
        } else if *b == ByteCode::LT as u8 {
            out.push_str(&format!("LT"));
        } else if *b == ByteCode::LTE as u8 {
            out.push_str(&format!("LTE"));
        } else if *b == ByteCode::Index as u8 {
            out.push_str(&format!("index"));
        } else if *b == ByteCode::Add as u8 {
            out.push_str(&format!("add"));
        } else if *b == ByteCode::CreateFn as u8 {
            out.push_str(&format!("fn"));
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
            out.push_str(&format!("assign arg"));
        } else if *b == ByteCode::CreateFn as u8 {
            out.push_str(&format!("fn"));
        } else if *b == ByteCode::CreateStruct as u8 {
            out.push_str(&format!("create struct"));
        } else if *b == ByteCode::GetAttr as u8 {
            let len = bytes[idx + 1] as usize;
            let slice = &bytes[idx + 2..idx + 2 + len];
            out.push_str(&format!(r#"get .{}"#, debug_u8s(slice)));
            idx += len + 1;
        } else if *b == ByteCode::SetAttr as u8 {
            let len = bytes[idx + 1] as usize;
            let slice = &bytes[idx + 2..idx + 2 + len];
            out.push_str(&format!(r#"set .{}"#, debug_u8s(slice)));
            idx += len + 1;
        } else if *b == ByteCode::VariableLoad as u8 {
            let len = bytes[idx+1] as usize;
            let slice = &bytes[idx+2..idx+2+len];
            out.push_str(&format!(r#"load "{}""#, debug_u8s(slice)));
            idx += len + 1;
        } else if *b == ByteCode::Break as u8 {
            out.push_str(&format!("break"));
        } else if *b == ByteCode::Continue as u8 {
            out.push_str(&format!("continue"));
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
            out.push_str(&format!("None"));
        } else if *b == ByteCode::CreateList as u8 {
            out.push_str(&format!("CreateList"));
        } else if *b == ByteCode::Copy as u8 {
            out.push_str(&format!("Copy"));
        } else if *b == ByteCode::LoopStart as u8 {
            let offset = ((bytes[idx + 1] as usize) << 8) + bytes[idx + 2] as usize;
            let target = idx + offset;
            out.push_str(&format!("Loop Start -> {}", target));
            idx += 2;
        } else if *b == ByteCode::LoopEnd as u8 {
            out.push_str(&format!("Loop End"));
        } else if *b == ByteCode::StartScope as u8 {
            out.push_str(&format!("start_scope"));
        } else if *b == ByteCode::EndScope as u8 {
            out.push_str(&format!("end_scope"));
        } else if *b == ByteCode::Pad0 as u8 {
            out.push_str(&format!(""));
        } else if *b == ByteCode::Pad1 as u8 {
            out.push_str(&format!(""));
        } else if *b == ByteCode::Pad2 as u8 {
            out.push_str(&format!(""));
        } else if *b == ByteCode::Pad3 as u8 {
            out.push_str(&format!(""));
        } else if *b == ByteCode::Pad4 as u8 {
            out.push_str(&format!(""));
        } else if *b == ByteCode::Pad5 as u8 {
            out.push_str(&format!(""));
        } else if *b == ByteCode::Pad6 as u8 {
            out.push_str(&format!(""));
        } else if *b == ByteCode::Pad7 as u8 {
            out.push_str(&format!(""));
        } else if *b == ByteCode::Pad8 as u8 {
            out.push_str(&format!(""));
        } else if *b == ByteCode::Pad9 as u8 {
            out.push_str(&format!(""));
        } else if *b == ByteCode::Return as u8 {
            out.push_str(&format!("return"));
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
