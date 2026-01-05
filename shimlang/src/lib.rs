use std::collections::HashMap;

#[cfg(feature = "facet")]
use facet::Facet;

use std::ops::{Add, Sub};
use std::ops::{AddAssign, SubAssign};

#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    fn start() -> Self {
        Self {
            start: 0,
            end: 1,
        }
    }
}

#[derive(Debug)]
pub struct Node<T> {
    pub data: T,
    pub span: Span,
}

// Now redefine your types using the wrapper
pub type ExprNode = Node<Expression>;

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
    Call(Box<ExprNode>, Vec<ExprNode>),
    Attribute(Box<ExprNode>, Vec<u8>),
    Block(Block),
    If(Box<ExprNode>, Block, Block),
}

#[derive(Debug)]
pub enum Statement {
    Let(Vec<u8>, ExprNode),
    Assignment(Vec<u8>, ExprNode),
    AttributeAssignment(ExprNode, Vec<u8>, ExprNode),
    If(ExprNode, Block, Block),
    For(Vec<u8>, ExprNode, Block),
    While(ExprNode, Block),
    Break,
    Continue,
    Fn(Vec<u8>, Vec<Vec<u8>>, Block),
    Struct(
        Vec<u8>,
        Vec<Vec<u8>>,
        /** These are methods */
        Vec<Statement>
    ),
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
            line_info.push(
                LineInfo { start_idx: line_start_idx, end_idx: idx as u32 }
            );
            line_start_idx = idx as u32 + 1;
        }
    }

    // If the last character was not a newline
    if line_start_idx != script.len() as u32 {
        line_info.push(
            LineInfo { start_idx: line_start_idx, end_idx: script.len() as u32 -1 }
        );
    }

    line_info
}

fn format_script_err(span: Span, script: &[u8], msg: &str) -> String {
    let script_lines = script_lines(script);
    let mut out = "".to_string();

    // The `gutter_size` includes everything up to the first character of the line
    let gutter_size = script_lines.len().to_string().len() + 4;
    for (lineno, line_info) in script_lines.iter().enumerate() {
        let lineno = lineno + 1;

        let line: String = unsafe { std::str::from_utf8_unchecked(&script[line_info.start_idx as usize ..= line_info.end_idx as usize]).to_string() };
        out.push_str(
            &format!(" {:lineno_size$} | {}", lineno, line, lineno_size = gutter_size - 4)
        );
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
            Err(
                self.format_peek_err("End of token stream")
            )
        } else {
            Ok(&self.tokens[self.idx])
        }
    }

    fn peek_span(&self) -> Result<Span, String> {
        if self.is_empty() {
            Err(
                self.format_peek_err("End of token stream")
            )
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
            Err("End of token stream".to_string())
        }
    }

    fn consume(&mut self, expected: Token) -> Result<(), String> {
        let value = self.pop()?;
        if value == expected {
            Ok(())
        } else {
            self.unadvance()?;
            Err(
                self.format_peek_err(
                    &format!("Expected token {:?} but found {:?}", expected, value)
                )
            )
        }
    }

    fn advance(&mut self) -> Result<(), String> {
        self.pop()?;
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

    fn script_lines(&self) -> Vec<LineInfo> {
        script_lines(&self.script)
    }

    fn idx_to_line_col(&self, _idx: u32) -> (u32, u32) {
        todo!();
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
        format_script_err(
            span,
            &self.script,
            msg
        )
    }

    fn format_err(&self, msg: &str) -> String {
        let script = unsafe { std::str::from_utf8_unchecked(&self.script) };

        format!(
            "Script:\n{}\n\nMessage:\n{}",
            script,
            msg
        )
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
        Token::None => {
            Expression::Primary(Primary::None)
        },
        Token::Integer(i) => {
            Expression::Primary(Primary::Integer(i))
        },
        Token::Float(f) => {
            Expression::Primary(Primary::Float(f))
        },
        Token::String(s) => {
            Expression::Primary(Primary::String(s))
        },
        Token::Bool(b) => {
            Expression::Primary(Primary::Bool(b))
        },
        Token::Identifier(s) => {
            Expression::Primary(Primary::Identifier(s))
        },
        Token::LCurly => {
            tokens.unadvance()?;
            let block = parse_block(tokens)?;
            Expression::Block(block)
        },
        Token::LBracket => {
            let expr = parse_expression(tokens)?;
            tokens.consume(Token::RBracket)?;
            return Ok(expr);
        },
        Token::LSquare => {
            let items = parse_arguments(tokens, Token::RSquare)?;
            // TODO: fix span here
            Expression::Primary(Primary::List(items))
        },
        token => {
            tokens.unadvance()?;
            return Err(tokens.format_peek_err(
                &format!(
                    "Unexpected `{:?}` in parse_primary",
                    // TODO: should display the exact character like `[` rather than name line `SemiColon`
                    token
                )
            ));
        },
    };
    Ok(
        Node {
            data: expr,
            span: span,
        }
    )
}

pub fn parse_arguments(tokens: &mut TokenStream, closing_token: Token) -> Result<Vec<ExprNode>, String> {
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
            token => return Err(
                tokens.format_peek_err(&format!("Expected comma or closing bracket, found {:?}", token))
            ),
        }
    }
    Ok(args)
}

pub fn parse_call(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_primary(tokens)?;
    while !tokens.is_empty() {
        let span = tokens.peek_span()?;
        match *tokens.peek()? {
            Token::LBracket => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::Call(Box::new(expr), parse_arguments(tokens, Token::RBracket)?),
                    span: span,
                };
            },
            Token::Dot => {
                tokens.advance()?;
                let ident = match tokens.pop()? {
                    Token::Identifier(ident) => {
                        ident.clone()
                    },
                    token => return Err(
                        tokens.format_peek_err(&format!("Expected ident after dot, found {:?}", token))
                    )
                };
                expr = Node {
                    data: Expression::Attribute(Box::new(expr), ident),
                    span: span,
                };
            },
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
                    data: Expression::BooleanOp(BooleanOp::Or(Box::new(expr), Box::new(parse_expression(tokens)?))),
                    span: span,
                };
            },
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
                    data: Expression::BooleanOp(BooleanOp::And(Box::new(expr), Box::new(parse_expression(tokens)?))),
                    span: span,
                };
            },
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
                    data: Expression::BinaryOp(BinaryOp::GT(Box::new(expr), Box::new(parse_expression(tokens)?))),
                    span: span,
                };
            },
            Token::GTE => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::GTE(Box::new(expr), Box::new(parse_expression(tokens)?))),
                    span: span,
                };
            },
            Token::LT => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::LT(Box::new(expr), Box::new(parse_expression(tokens)?))),
                    span: span,
                };
            },
            Token::LTE => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::LTE(Box::new(expr), Box::new(parse_expression(tokens)?))),
                    span: span,
                };
            },
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
                    data: Expression::BinaryOp(BinaryOp::Multiply(Box::new(expr), Box::new(parse_expression(tokens)?))),
                    span: span,
                };
            },
            Token::Slash => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::Divide(Box::new(expr), Box::new(parse_expression(tokens)?))),
                    span: span,
                };
            },
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
                    data: Expression::BinaryOp(BinaryOp::Equal(Box::new(expr), Box::new(parse_expression(tokens)?))),
                    span: span,
                };
            },
            Token::BangEqual => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::NotEqual(Box::new(expr), Box::new(parse_expression(tokens)?))),
                    span: span,
                };
            },
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
                    data: Expression::BinaryOp(BinaryOp::Add(Box::new(expr), Box::new(parse_expression(tokens)?))),
                    span: span,
                };
            },
            Token::Minus => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::Subtract(Box::new(expr), Box::new(parse_expression(tokens)?))),
                    span: span,
                };
            },
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
        },
        Token::Minus => {
            tokens.advance()?;
            let expr = parse_unary(tokens)?;
            Ok(Node {
                data: Expression::UnaryOp(UnaryOp::Not(Box::new(expr))),
                span: span,
            })
        },
        _ => parse_call(tokens),
    }
}


struct Conditional {
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
            Ok(
                ExprNode {
                    data: Expression::If(Box::new(cond.conditional), cond.if_body, cond.else_body),
                    span: Span::start(),
                }
            )
        },
        _ => parse_logical_or(tokens)
    }
}

pub fn parse_ast(tokens: &mut TokenStream) -> Result<Ast, String> {
    let block = parse_block_inner(tokens)?;
    Ok(Ast { block: block, script: tokens.script.clone() })
}

pub fn parse_function(tokens: &mut TokenStream) -> Result<Statement, String> {
    tokens.consume(Token::Fn)?;
    let ident = match tokens.pop()? {
        Token::Identifier(ident) => {
            ident.clone()
        },
        token => return Err(
            tokens.format_peek_err(&format!("Expected ident after fn, found {:?}", token))
        )
    };
    tokens.consume(Token::LBracket)?;

    let mut params = Vec::new();
    while !tokens.is_empty() {
        match tokens.peek()? {
            Token::Identifier(ident) => {
                params.push(ident.clone())
            }
            Token::RBracket => break,
            token => return Err(
                tokens.format_peek_err(&format!("Expected ident for fn params, found {:?}", token))
            )
        }
        tokens.advance()?;

        if *tokens.peek()? != Token::Comma {
            break;
        }
        tokens.advance()?;
    }
    tokens.consume(Token::RBracket)?;

    let mut body = parse_block(tokens)?;

    Ok(Statement::Fn(ident, params, body))
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
                Token::Identifier(ident) => {
                    ident.clone()
                },
                token => return Err(format!("Expected ident after let, found {:?}", token))
            };

            match tokens.pop()? {
                Token::Equal => (),
                token => return Err(format!("Expected = after `let ident`, found {:?}", token))
            }

            let expr = parse_expression(tokens)?;
            match tokens.pop()? {
                Token::Semicolon => (),
                token => {
                    tokens.unadvance()?;
                    return Err(
                        tokens.format_peek_err(
                            &format!("Expected semicolon after `let <ident> = <expr>`, found {:?}", token)
                        )
                    );
                }
            }

            Statement::Let(ident, expr)
        } else if *tokens.peek()? == Token::Fn {
            parse_function(tokens)?
        } else if *tokens.peek()? == Token::If {
            let cond = parse_conditional(tokens)?;

            // Do we treat this as an expression or statement?
            if *tokens.peek()? == Token::RCurly {
                let expr = Expression::If(Box::new(cond.conditional), cond.if_body, cond.else_body);
                last_expr = Some(
                    Box::new(
                        ExprNode {
                            data: expr,
                            span: Span::start(),
                        }
                    )
                );
                break;
            } else {
                Statement::If(cond.conditional, cond.if_body, cond.else_body)
            }
        } else if *tokens.peek()? == Token::For {
            tokens.advance()?;
            let ident = match tokens.pop()? {
                Token::Identifier(ident) => {
                    ident
                },
                token => {
                    tokens.unadvance()?;
                    return Err(
                        tokens.format_peek_err(&format!("Expected ident after for, found {:?}", token))
                    );
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
                Token::Identifier(ident) => {
                    ident.clone()
                },
                token => {
                    tokens.unadvance()?;
                    return Err(
                        tokens.format_peek_err(&format!("Expected ident after struct, found {:?}", token))
                    );
                }
            };
            tokens.consume(Token::LCurly)?;

            let mut members = Vec::new();
            while !tokens.is_empty() {
                match tokens.pop()? {
                    Token::Identifier(ident) => {
                        members.push(ident.clone());
                        if *tokens.peek()? != Token::Comma {
                            break;
                        }
                        tokens.advance()?;
                    },
                    Token::RCurly | Token::Fn => {
                        tokens.unadvance()?;
                        break;
                    },
                    token => return Err(format!("Expected member list after struct, found {:?}", token))
                };
            }

            let mut methods = Vec::new();
            while !tokens.is_empty() {
                match tokens.peek()? {
                    Token::Fn => {
                        methods.push(parse_function(tokens)?);
                    },
                    Token::RCurly => {
                        break;
                    }
                    token => return Err(format!("Unexpected token during method parsing {:?}", token))
                }
            }
            tokens.consume(Token::RCurly)?;

            Statement::Struct(ident, members, methods)
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
                        return Err(
                            tokens.format_peek_err(
                                &format!("Expected semicolon after `return <expr>`, found {:?}", token)
                            )
                        );
                    }
                }
                Statement::Return(Some(expr))
            }
        } else {
            let expr = parse_expression(tokens)?;

            match tokens.peek()? {
                Token::RCurly => {
                    last_expr = Some(Box::new(expr));
                    break;
                },
                Token::Semicolon => {
                    tokens.pop()?;
                    Statement::Expression(expr)
                },
                Token::Equal => {
                    tokens.pop()?;
                    match expr.data {
                        Expression::Primary(Primary::Identifier(ident)) => {
                            let expr_to_assign = parse_expression(tokens)?;
                            tokens.consume(Token::Semicolon)?;
                            Statement::Assignment(ident.clone(), expr_to_assign)
                        },
                        Expression::Attribute(expr, ident) => {
                            let expr_to_assign = parse_expression(tokens)?;
                            tokens.consume(Token::Semicolon)?;
                            Statement::AttributeAssignment(*expr, ident.clone(), expr_to_assign)
                        },
                        expr_data => {
                            return Err(
                                format_script_err(
                                    expr.span,
                                    &tokens.script,
                                    &format!("Can't assign to {:?}", expr_data),
                                )
                            );
                        }
                    }
                },
                token => return Err(
                    tokens.format_peek_err(
                        &format!(
                            "Expected semicolon after expression statement, found {:?}",
                            token
                        )
                    )
                )
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
            b'a' ..= b'z' | b'A' ..= b'Z' | b'0' ..= b'9' | b'_' => continue,
            _ => {
                let ident = text[0..idx].to_vec();
                *text = &text[(idx-1)..];
                return Ok(ident);
            }
        }
    }
    Ok(text.to_vec())
}

pub fn lex_string(text: &mut &[u8]) -> Result<Vec<u8>, String> {
    let enclosing_char = text[0];
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
                b => return Err(format!("Could not escape {:?}", printable_byte(*b)))
            }
            escape_next = false;
            continue
        }
        if *c == enclosing_char {
            *text = &text[idx..];
            return Ok(out);
        }
        match c {
            b'\\' => {
                escape_next = true;
                continue;
            },
            _ => {
                out.push(*c);
            }
        }
    }
    Err(format!("No closing quote {:?} found", printable_byte(enclosing_char)))
}

pub fn lex_number(text: &mut &[u8]) -> Result<Token, String> {
    let mut found_decimal = false;
    for (idx, c) in text.iter().enumerate() {
        match c {
            b'0' ..= b'9' => continue,
            b'.' => {
                if found_decimal {
                    return Err(format!("Found multiple decimals in number"));
                }
                found_decimal = true;
            }
            _ => {
                let token = if found_decimal {
                    Token::Float(
                        unsafe {
                            let slice = &text[..idx];
                            std::str::from_utf8_unchecked(slice).parse().map_err(|e| 
                                {
                                    let string_slice = match std::str::from_utf8(slice) {
                                        Ok(s) => s,
                                        Err(e) => return format!("Not utf-8 {:?}", e)
                                    };
                                    format!("Could not tokenize number '{}' {:?}", string_slice, e)
                                }
                            )?
                        }
                    )
                } else {
                    Token::Integer(
                        unsafe {
                            let slice = &text[..idx];
                            std::str::from_utf8_unchecked(slice).parse().map_err(|e| 
                                {
                                    let string_slice = match std::str::from_utf8(slice) {
                                        Ok(s) => s,
                                        Err(e) => return format!("Not utf-8 {:?}", e)
                                    };
                                    format!("Could not tokenize number '{}' {:?}", string_slice, e)
                                }
                            )?
                        }
                    )
                };
                *text = &text[(idx-1)..];
                return Ok(token);
            }
        }
    }
    let token = Token::Integer(
        unsafe {
            std::str::from_utf8_unchecked(text).parse().map_err(|e| format!("{:?}", e))?
        }
    );
    Ok(token)
}

pub fn lex(text: &[u8]) -> Result<TokenStream, String> {
    let starting_len = text.len();
    let starting_text = text;
    let mut text = text;
    let mut spans = Vec::new();
    let mut tokens = Vec::new();

    while !text.is_empty() {
        let c = text[0];
        let token_start_len = text.len();
        match c {
            b'a' ..= b'z' | b'A' ..= b'Z' | b'_' => {
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
            },
            b'0' ..= b'9' => tokens.push(lex_number(&mut text)?),
            b'"' => tokens.push(
                Token::String(lex_string(&mut text)?)
            ),
            b'{' => tokens.push(Token::LCurly),
            b'}' => tokens.push(Token::RCurly),
            b',' => tokens.push(Token::Comma),
            b'(' => tokens.push(Token::LBracket),
            b')' => tokens.push(Token::RBracket),
            b'+' => tokens.push(Token::Plus),
            b'-' => tokens.push(Token::Minus),
            b'=' => {
                match text[1] {
                    b'=' => {
                        text = &text[1..];
                        tokens.push(Token::DEqual);
                    },
                    _ => tokens.push(Token::Equal),
                }
            },
            b'>' => {
                match text[1] {
                    b'=' => {
                        text = &text[1..];
                        tokens.push(Token::GTE);
                    },
                    _ => tokens.push(Token::LT),
                }
            },
            b'<' => {
                match text[1] {
                    b'=' => {
                        text = &text[1..];
                        tokens.push(Token::LTE);
                    },
                    _ => tokens.push(Token::LT),
                }
            },
            b';' => tokens.push(Token::Semicolon),
            b'\n' => (),
            b'[' => tokens.push(Token::LSquare),
            b']' => tokens.push(Token::RSquare),
            b':' => tokens.push(Token::Colon),
            b'!' => tokens.push(Token::Bang),
            b'.' => tokens.push(Token::Dot),
            b' ' => (),
            _ => return Err(format!("Unknown character '{}'", printable_byte(c)))
        }
        text = &text[1..];
        let token_end_len = text.len();
        if tokens.len() > spans.len() {
            spans.push(
                Span {
                    start: (starting_len - token_start_len) as u32,
                    end: (starting_len - token_end_len) as u32,
                }
            );
        }
    }
    Ok(
        TokenStream {
            idx: 0,
            tokens: tokens,
            token_spans: spans,
            script: starting_text.to_vec(),
        }
    )
}

pub fn ast_from_text(text: &[u8]) -> Result<Ast, String> {
    let mut tokens = lex(text)?;
    parse_ast(&mut tokens)
}

pub struct Config {
    // There are max 2^24 addressable values, each 8 bytes large
    // This value can be up to 2^32.
    memory_space_bytes: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            memory_space_bytes: 2u32.pow(22) // 4 MB
        }
    }
}

#[derive(Hash, Eq, PartialOrd, Ord, Copy, Clone, Debug, PartialEq)]
#[repr(packed)]
struct u24([u8; 3]);

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
        Self {
            pos,
            size,
        }
    }
}

#[cfg_attr(feature = "facet", derive(Facet))]
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

impl MMU {
    fn with_capacity(word_count: Word) -> Self {
        let mem = vec![0; usize::from(word_count.0)];
        let free_list = vec![FreeBlock::new(Word(0.into()), word_count)];
        Self {
            mem: mem,
            free_list: free_list,
        }
    }

    fn compact_free_list() {
        todo!("compact_free_list not implemented");
    }

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

    /**
     * Returns the position in `self.mem` of the block allocted
     */
    fn alloc(&mut self, words: Word) -> Word {
        for block in self.free_list.iter_mut() {
            if block.size >= words {
                let returned_pos: Word = block.pos;
                block.pos += words;
                block.size -= words;

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
            } else {
                eprintln!("{:?} not larger than {:?}", block.size, words);
            }
        }
        panic!("Could not allocate {:?} words from free list {:#?}", words, self.free_list);
    }

    fn free(&mut self, _words: u32, _ptr: *const u64) {
    }
}

struct Environment {
    env_chain: Vec<HashMap<Vec<u8>, u64>>
}

impl Environment {
    fn new() -> Self {
        Self {
            env_chain: vec![HashMap::new()]
        }
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
                return true
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

// TODO: uncomment #[derive(Facet)]
pub struct Interpreter {
    pub mem: MMU,
    pub source: HashMap<String, String>,
    pub env: Environment,
}

#[derive(Copy, Clone, Debug)]
pub enum ShimValue {
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
    // A function pointer doesn't fit in the ShimValue, so we need to store the
    // function pointer in interpreter memory
    NativeFn(Word),
    // TODO: it seems like this should point to a more generic reference-counted
    // object type that all non-value types share
    String(Word),
    List(Word),
    StructDef(Word),
    Struct(Word),
}

use std::mem::{size_of};
const _: () = {
    assert!(std::mem::size_of::<ShimValue>() <= 8);
};

type NativeFn = fn(&mut Interpreter, &Vec<ShimValue>) -> Result<ShimValue, String>;
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

#[derive(Debug)]
enum StructAttribute {
    MemberInstanceOffset(u8),
    MethodDefPC(u32),
}

#[derive(Debug)]
struct StructDef {
    member_count: u8,
    method_count: u8,
    lookup: Vec<(Vec<u8>, StructAttribute)>,
}

#[derive(Debug)]
enum CallResult {
    ReturnValue(ShimValue),
    PC(u32),
}

fn shim_print(interpreter: &mut Interpreter, args: &Vec<ShimValue>) -> Result<ShimValue, String> {
    for (idx, arg) in args.iter().enumerate() {
        if idx != 0 {
            print!(" ");
        }
        print!("{}", arg.to_string(interpreter));
    }

    println!();
    Ok(ShimValue::None)
}

fn shim_panic(interpreter: &mut Interpreter, args: &Vec<ShimValue>) -> Result<ShimValue, String> {
    let mut out = String::new();
    for (idx, arg) in args.iter().enumerate() {
        if idx != 0 {
            out.push(' ');
        }
        out.push_str(&format!("{}", arg.to_string(interpreter)));
    }

    out.push('\n');
    Err(out)
}

impl ShimValue {
    fn call(&self, interpreter: &mut Interpreter, stack: &mut Vec<ShimValue>) -> Result<CallResult, String> {
        let arg_pos: Word = match stack[stack.len()-1] {
            ShimValue::List(arg_pos) => arg_pos,
            args => return Err(format!("Can't call print with non-list args {:?}", args)),
        };

        let args: &mut Vec<ShimValue> = unsafe {
            let ptr: *mut Vec<ShimValue> = std::mem::transmute(&mut interpreter.mem.mem[usize::from(arg_pos.0)]);
            &mut *ptr
        };
        match self {
            ShimValue::None => Err(format!("Can't call None as a function")),
            ShimValue::Fn(pc) => {
                Ok(CallResult::PC(*pc))
            }
            ShimValue::BoundMethod(pos, pc) => {
                // push struct pos to start of arg list then return the pc of the method
                args.insert(0, ShimValue::Struct(*pos));
                Ok(CallResult::PC(*pc))
            }
            ShimValue::StructDef(struct_def_pos) => {
                let struct_def: &StructDef = unsafe { interpreter.mem.get(*struct_def_pos) };
                if struct_def.member_count as usize != args.len() {
                    return Err(
                        format!(
                            "Expected {} arguments in initializer, got {}",
                            struct_def.member_count,
                            args.len(),
                        )
                    );
                }

                // Allocate space for each member, plus the header
                let word_count = Word((struct_def.member_count as u32 + 1).into());
                let new_pos = interpreter.mem.alloc(word_count);

                // The first word points to the StructDef
                interpreter.mem.mem[usize::from(new_pos.0)] = u64::from(struct_def_pos.0);

                // The remaining words get copies of the arguments to the initializer
                for (idx, arg) in args.into_iter().enumerate() {
                    interpreter.mem.mem[usize::from(new_pos.0)+1+idx] = arg.to_u64();
                }

                Ok(CallResult::ReturnValue(ShimValue::Struct(new_pos)))
            }
            ShimValue::NativeFn(pos) => {
                stack.pop();
                let native_fn: &NativeFn = unsafe { interpreter.mem.get(*pos) };
                Ok(CallResult::ReturnValue(native_fn(interpreter, args)?))
            }
            other => Err(format!("Can't call value {:?} as a function", other.to_string(interpreter))),
        }
    }

    fn to_string(&self, interpreter: &mut Interpreter) -> String {
        match self {
            ShimValue::Integer(i) => i.to_string(),
            ShimValue::Float(f) => format_float(*f),
            ShimValue::Bool(false) => "false".to_string(),
            ShimValue::Bool(true) => "true".to_string(),
            ShimValue::String(position) => {
                unsafe {
                    let ptr: *mut Vec<u8> = std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)]);
                    String::from_utf8((*ptr).clone()).expect("valid utf-8 string stored")
                }
            },
            ShimValue::List(position) => {
                let mut out = "[".to_string();
                unsafe {
                    let ptr: *mut Vec<ShimValue> = std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)]);
                    for (idx, item) in (*ptr).iter().enumerate() {
                        if idx != 0 {
                            out.push_str(",");
                            out.push_str(" ");
                        }
                        out.push_str(&item.to_string(interpreter));
                    }
                }

                out.push_str("]");

                out
            },
            value => format!("{:?}", value),
        }
    }

    fn is_truthy(&self, interpreter: &mut Interpreter) -> Result<bool, String> {
        match self {
            ShimValue::Integer(i) => Ok(*i != 0),
            ShimValue::Float(f) => Ok(*f != 0.0),
            ShimValue::Bool(false) => Ok(false),
            ShimValue::Bool(true) => Ok(true),
            ShimValue::String(position) => {
                unsafe {
                    let ptr: *mut Vec<u8> = std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)]);
                    Ok(!(*ptr).is_empty())
                }
            },
            ShimValue::List(position) => {
                unsafe {
                    let ptr: *mut Vec<ShimValue> = std::mem::transmute(&mut interpreter.mem.mem[usize::from(position.0)]);
                    Ok((*ptr).len() != 0)
                }
            },
            _ => Ok(true),
        }
    }

    fn add(&self, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Integer(a), ShimValue::Integer(b)) => {
                Ok(ShimValue::Integer(a + b))
            },
            (a, b) => Err(format!("Can't add {:?} and {:?}", a, b))
        }
    }

    fn sub(&self, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Integer(a), ShimValue::Integer(b)) => {
                Ok(ShimValue::Integer(a - b))
            },
            (a, b) => Err(format!("Can't sub {:?} and {:?}", a, b))
        }
    }

    fn equal(&self, _interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => Ok(ShimValue::Bool(a == b)),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Bool(a == b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(a == b)),
            (ShimValue::None, ShimValue::None) => Ok(ShimValue::Bool(true)),
            _ => Ok(ShimValue::Bool(false)),
        }
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
        match (self, other) {
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Float(a * b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Integer(a * b)),
            (a, b) => Err(format!("Can't Multiply {:?} and {:?}", a, b))
        }
    }

    fn div(&self, _interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Float(a / b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Integer(a / b)),
            (a, b) => Err(format!("Can't Divide {:?} and {:?}", a, b))
        }
    }

    fn gt(&self, _interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => Ok(ShimValue::Bool(*a == true && *b == false)),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Bool(a > b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(a > b)),
            (ShimValue::None, ShimValue::None) => Ok(ShimValue::Bool(false)),
            (a, b) => Err(format!("Can't GT {:?} and {:?}", a, b))
        }
    }

    fn gte(&self, _interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => Ok(ShimValue::Bool(a == b)),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Bool(a >= b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(a >= b)),
            (ShimValue::None, ShimValue::None) => Ok(ShimValue::Bool(true)),
            (a, b) => Err(format!("Can't GTE {:?} and {:?}", a, b))
        }
    }

    fn lt(&self, _interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => Ok(ShimValue::Bool(*a == false && *b == true)),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Bool(a < b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(a < b)),
            (ShimValue::None, ShimValue::None) => Ok(ShimValue::Bool(false)),
            (a, b) => Err(format!("Can't LT {:?} and {:?}", a, b))
        }
    }

    fn lte(&self, _interpreter: &mut Interpreter, other: &Self) -> Result<ShimValue, String> {
        match (self, other) {
            (ShimValue::Bool(a), ShimValue::Bool(b)) => Ok(ShimValue::Bool(a == b)),
            (ShimValue::Float(a), ShimValue::Float(b)) => Ok(ShimValue::Bool(a <= b)),
            (ShimValue::Integer(a), ShimValue::Integer(b)) => Ok(ShimValue::Bool(a <= b)),
            (ShimValue::None, ShimValue::None) => Ok(ShimValue::Bool(true)),
            (a, b) => Err(format!("Can't LTE {:?} and {:?}", a, b))
        }
    }

    fn not(&self, _interpreter: &mut Interpreter) -> Result<ShimValue, String> {
        match self {
            ShimValue::Bool(a) => Ok(ShimValue::Bool(!a)),
            ShimValue::Float(a) => Ok(ShimValue::Bool(*a == 0.0)),
            ShimValue::Integer(a) => Ok(ShimValue::Bool(*a == 0)),
            ShimValue::None => Ok(ShimValue::Bool(true)),
            other => Ok(ShimValue::Bool(false)),
        }
    }

    fn neg(&self, _interpreter: &mut Interpreter) -> Result<ShimValue, String> {
        match self {
            ShimValue::Float(a) => Ok(ShimValue::Float(-a)),
            ShimValue::Integer(a) => Ok(ShimValue::Integer(-a)),
            _ => Err(format!("Can't Negate {:?}", self))
        }
    }

    fn get_attr(&self, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
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
                                    Ok(
                                        *interpreter.mem.get(*pos + *offset as u32 + 1)
                                    )
                                }
                                StructAttribute::MethodDefPC(pc) => {
                                    Ok(
                                        ShimValue::BoundMethod(
                                            *pos,
                                            *pc
                                        )
                                    )
                                }
                            };
                        }
                    }
                }
                Err(format!("Ident {:?} not found for {:?}", ident, self))
            },
            ShimValue::StructDef(def_pos) => {
                unsafe {
                    let def: &StructDef = interpreter.mem.get(*def_pos);
                    for (attr, loc) in def.lookup.iter() {
                        if ident == attr {
                            return match loc {
                                StructAttribute::MemberInstanceOffset(offset) => {
                                    Err(format!("Can't access member {:?} on StructDef {:?}", ident, self))
                                }
                                StructAttribute::MethodDefPC(pc) => {
                                    // Return the method
                                    Ok(ShimValue::Fn(*pc))
                                }
                            };
                        }
                    }
                }
                Err(format!("Ident {:?} not found for {:?}", ident, self))
            },
            val => Err(format!("Ident {:?} not available on {:?}", ident, val))
        }
    }

    fn set_attr(&self, interpreter: &mut Interpreter, ident: &[u8], val: ShimValue) -> Result<(), String> {
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
                                    let slot: &mut ShimValue = interpreter.mem.get_mut(*pos + *offset as u32 + 1);
                                    *slot = val;
                                    Ok(())
                                }
                                StructAttribute::MethodDefPC(_) => {
                                    Err(
                                        format!("Can't assign to struct method {:?} for {:?}", ident, self)
                                    )
                                }
                            };
                        }
                    }
                }
                Err(format!("Ident {:?} not found for {:?}", ident, self))
            },
            val => Err(format!("Ident {:?} not available on {:?}", ident, val))
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
    AssertLen = 128,
    Splat,
    Pop,
    Add,
    Sub,
    Equal,
    NotEqual,
    Multiply,
    Divide,
    GT,
    GTE,
    LT,
    LTE,
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
    Break,
    Continue,
    Call,
    Return,
    Jmp,
    JmpUp,
    JmpNZ,
    JmpZ,
}

pub struct Program {
    pub bytecode: Vec<u8>,
    spans: Vec<Span>,
    script: Vec<u8>,
}

pub fn compile_ast(ast: &Ast) -> Result<Program, String> {
    let mut program = compile_block(&ast.block, false)?;
    let (bytecode, spans): (Vec<u8>, Vec<Span>) = program.into_iter().unzip();
    Ok(
        Program {
            bytecode: bytecode,
            spans: spans,
            script: ast.script.clone(),
        }
    )
}

pub fn u16_to_u8s(val: u16) -> [u8; 2] {
    [
        (val >> 8) as u8,
        (val & 0xff) as u8
    ]
}

pub fn u8s_to_u16(val: [u8; 2]) -> u16 {
    ((val[0] as u16) << 8) + val[1] as u16
}

pub fn compile_function_body(params: &Vec<Vec<u8>>, body: &Block) -> Result<Vec<(u8, Span)>, String> {
    let mut asm = Vec::new();
    asm.push((ByteCode::AssertLen as u8, Span {start:0, end:1}));
    asm.push((params.len() as u8, Span {start:0, end:1}));
    asm.push((ByteCode::Splat as u8, Span {start:0, end:1}));
    for param in params.iter().rev() {
        asm.push((ByteCode::VariableDeclaration as u8, Span {start:0, end:1}));
        asm.push((param.len().try_into().expect("Param len should into u8"), Span {start:0, end:1}));
        for b in param {
            asm.push((*b, Span {start:0, end:1}));
        }
    }

    for stmt in body.stmts.iter() {
        asm.extend(compile_statement(&stmt)?);
    }

    if let Some(expr) = &body.last_expr {
        let val: Option<&ExprNode> = Some(expr);
        asm.extend(
            compile_return(&val)?
        );
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
            let val: Option<&ExprNode> = Some(
                &ExprNode {
                    data: Expression::Primary(Primary::None),
                    span: Span::start(),
                }
            );
            asm.extend(
                compile_return(&val)?
            );
        }
    }

    if asm.len() > u16::MAX as usize {
        return Err(format!("Function has more than {} instructions", u16::MAX));
    }
    Ok(asm)
}

pub fn compile_return(expr: &Option<&ExprNode>) -> Result<Vec<(u8, Span)>, String> {
    let mut res = Vec::new();
    if let Some(expr) = expr {
        res.extend(compile_expression(expr)?);
    } else {
        res.push((ByteCode::LiteralNone as u8, Span { start: 0, end: 1}));
    }
    res.push((ByteCode::Return as u8, Span { start: 0, end: 1}));
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
            expr_asm.push((ident.len().try_into().expect("Ident len should into u8"), expr.span));
            for b in ident.into_iter() {
                expr_asm.push((*b, expr.span));
            }

            Ok(expr_asm)
        },
        Statement::Assignment(ident, expr) => {
            let mut expr_asm = compile_expression(expr)?;
            expr_asm.push((ByteCode::Assignment as u8, expr.span));
            expr_asm.push((ident.len().try_into().expect("Ident len should into u8"), expr.span));
            for b in ident.into_iter() {
                expr_asm.push((*b, expr.span));
            }

            Ok(expr_asm)
        },
        Statement::AttributeAssignment(obj_expr, ident, expr) => {
            let mut expr_asm = compile_expression(obj_expr)?;
            expr_asm.extend(compile_expression(expr)?);
            expr_asm.push((ByteCode::SetAttr as u8, expr.span));
            expr_asm.push((ident.len().try_into().expect("Ident len should into u8"), expr.span));
            for b in ident.into_iter() {
                expr_asm.push((*b, expr.span));
            }

            Ok(expr_asm)
        },
        Statement::Fn(ident, params, body) => {
            // This will be replaced with a relative jump to after the function
            // declaration
            let mut asm = vec![
                (ByteCode::Jmp as u8, Span {start: 0, end: 1}),
                (0, Span {start: 0, end: 1}),
                (0, Span {start: 0, end: 1}),
            ];
            asm.extend(compile_function_body(params, body)?);

            // Fix the jump offset at the function declaration now that we know
            // the size of the body
            let pc_offset = asm.len() as u16;
            asm[1].0 = (pc_offset >> 8) as u8;
            asm[2].0 = (pc_offset & 0xff) as u8;

            // Assign the value to the ident
            let pc_offset = asm.len() as u16 - 3;
            asm.push((ByteCode::CreateFn as u8, Span {start:0, end:1}));
            asm.push(((pc_offset >> 8) as u8, Span {start:0, end:1}));
            asm.push(((pc_offset & 0xff) as u8, Span {start:0, end:1}));

            asm.push((ByteCode::VariableDeclaration as u8, Span {start:0, end:1}));
            asm.push((ident.len().try_into().expect("Ident len should into u8"), Span {start:0, end:1}));
            for b in ident.into_iter() {
                asm.push((*b, Span {start:0, end:1}));
            }

            Ok(asm)
        },
        Statement::Struct(ident, members, methods) => {
            let mut asm = vec![
                (ByteCode::CreateStruct as u8, Span {start: 0, end: 1}),
                (0, Span {start: 0, end: 1}),
                (0, Span {start: 0, end: 1}),
            ];
            asm.push((members.len() as u8, Span {start: 0, end: 1}));
            asm.push((methods.len() as u8, Span {start: 0, end: 1}));

            for member in members.iter() {
                asm.push((member.len().try_into().expect("Member ident len should into u8"), Span {start:0, end:1}));
                for b in member.into_iter() {
                    asm.push((*b, Span {start:0, end:1}));
                }
            }

            let mut method_defs = Vec::new();
            for method in methods {
                match method {
                    Statement::Fn(ident, params, body) => {
                        method_defs.push((ident, compile_function_body(params, body)?));
                    },
                    other => return Err(format!("Unexpected statement as struct method {:?}", other)),
                }
            }

            let mut jump_asm_idx = Vec::new();
            for (ident, _method_def) in method_defs.iter() {
                jump_asm_idx.push(asm.len());
                asm.push((0, Span {start: 0, end: 1}));
                asm.push((0, Span {start: 0, end: 1}));
                asm.push((ident.len().try_into().expect("Method ident len should into u8"), Span {start:0, end:1}));
                for b in ident.into_iter() {
                    asm.push((*b, Span {start:0, end:1}));
                }
            }

            for (method_idx, (_, method_def)) in method_defs.into_iter().enumerate() {
                let jump_idx = jump_asm_idx[method_idx];
                let pc_offset = asm.len() as u16;
                asm[jump_idx].0 = (pc_offset >> 8) as u8;
                asm[jump_idx+1].0 = (pc_offset & 0xff) as u8;
                asm.extend(method_def);
            }

            let pc_offset = asm.len() as u16;
            asm[1].0 = (pc_offset >> 8) as u8;
            asm[2].0 = (pc_offset & 0xff) as u8;

            asm.push((ByteCode::VariableDeclaration as u8, Span {start:0, end:1}));
            asm.push((ident.len().try_into().expect("Ident len should into u8"), Span {start:0, end:1}));

            for b in ident.into_iter() {
                asm.push((*b, Span {start:0, end:1}));
            }

            Ok(asm)
        },
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

                // Create empty args and call
                asm.push((ByteCode::CreateList as u8, expr.span));
                asm.push((0, expr.span));
                asm.push((0, expr.span));
                asm.push((ByteCode::Call as u8, expr.span));

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
            asm.extend(
                vec![
                    (ByteCode::LoopStart as u8, Span {start: 0, end: 1}),
                    (0, Span {start: 0, end: 1}),
                    (0, Span {start: 0, end: 1}),
                ]
            );

            // Copy the .next bound method and call it
            asm.push((ByteCode::Copy as u8, expr.span));
            asm.push((ByteCode::CreateList as u8, expr.span));
            asm.push((0, expr.span));
            asm.push((0, expr.span));
            asm.push((ByteCode::Call as u8, expr.span));

            // Copy the result of .next() so we can later check if it's None
            asm.push((ByteCode::Copy as u8, expr.span));
            asm.push((ByteCode::VariableDeclaration as u8, Span {start:0, end:1}));
            asm.push((ident.len().try_into().expect("For loop ident len should into u8"), Span {start:0, end:1}));
            for b in ident {
                asm.push((*b, Span {start:0, end:1}));
            }

            asm.push((ByteCode::LiteralNone as u8, expr.span));
            asm.push((ByteCode::Equal as u8, expr.span));

            // Jump to `LoopEnd` if calling .next() returns None
            let none_check_idx = asm.len();
            asm.push((ByteCode::JmpNZ as u8, Span { start:0, end:0}));
            asm.push((0, Span { start:0, end:0}));
            asm.push((0, Span { start:0, end:0}));

            asm.extend(compile_block(body, false)?);

            // Jump to start of loop to check the condition again
            let loop_start_offset = u16_to_u8s(
                asm.len() as u16 - loop_start_idx as u16 - 3
            );
            asm.push((ByteCode::JmpUp as u8, Span { start:0, end:0}));
            asm.push((loop_start_offset[0], Span { start:0, end:0}));
            asm.push((loop_start_offset[1], Span { start:0, end:0}));

            // This is the offset from none_check_idx that will get us
            // out of the loop
            let pc_offset = u16_to_u8s(
                asm.len() as u16 - none_check_idx as u16
            );
            asm[none_check_idx+1].0 = pc_offset[0];
            asm[none_check_idx+2].0 = pc_offset[1];

            let loop_end = u16_to_u8s(asm.len() as u16 - loop_start_idx as u16);
            asm[loop_start_idx + 1].0 = loop_end[0];
            asm[loop_start_idx + 2].0 = loop_end[1];

            asm.push((ByteCode::LoopEnd as u8, Span { start:0, end:0}));

            Ok(asm)
        },
        Statement::While(conditional, body) => {
            let mut asm = vec![
                (ByteCode::LoopStart as u8, Span {start: 0, end: 1}),
                (0, Span {start: 0, end: 1}),
                (0, Span {start: 0, end: 1}),
            ];
            asm.extend(compile_expression(conditional)?);

            // Jump to `LoopEnd` if the condition is falsy
            let conditional_check_idx = asm.len();
            asm.push((ByteCode::JmpZ as u8, Span { start:0, end:0}));
            asm.push((0, Span { start:0, end:0}));
            asm.push((0, Span { start:0, end:0}));

            asm.extend(compile_block(body, false)?);

            // Jump to start of loop to check the condition again
            let loop_start_offset = u16_to_u8s(
                asm.len() as u16 - 3
            );
            asm.push((ByteCode::JmpUp as u8, Span { start:0, end:0}));
            asm.push((loop_start_offset[0], Span { start:0, end:0}));
            asm.push((loop_start_offset[1], Span { start:0, end:0}));

            // This is the offset from conditional_check_idx that will get us
            // out of the loop
            let pc_offset = u16_to_u8s(
                asm.len() as u16 - conditional_check_idx as u16
            );
            asm[conditional_check_idx+1].0 = pc_offset[0];
            asm[conditional_check_idx+2].0 = pc_offset[1];

            let loop_end = u16_to_u8s(asm.len() as u16);
            asm[1].0 = loop_end[0];
            asm[2].0 = loop_end[1];

            asm.push((ByteCode::LoopEnd as u8, Span { start:0, end:0}));

            Ok(asm)
        },
        Statement::Break => {
            Ok(
                vec![
                    (ByteCode::Break as u8, Span {start: 0, end: 1}),
                ]
            )
        },
        Statement::Continue => {
            Ok(
                vec![
                    (ByteCode::Continue as u8, Span {start: 0, end: 1}),
                ]
            )
        },
        Statement::Return(expr) => {
            compile_return(&expr.as_ref())
        },
        Statement::If(conditional, if_body, else_body) => {
            compile_if(conditional, if_body, else_body, false)
        },
        Statement::Expression(expr) => {
            // Expression evaluates to a value that's on the top of the stack
            let mut expr_asm = compile_expression(expr)?;
            // Pop the value since it's not used
            expr_asm.push((ByteCode::Pop as u8, expr.span));

            Ok(expr_asm)
        },
    }
}

pub fn compile_block(block: &Block, is_expr: bool) -> Result<Vec<(u8, Span)>, String> {
    let mut asm = Vec::new();

    asm.push((ByteCode::StartScope as u8, Span { start: 0, end: 1}));
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
    asm.push((ByteCode::EndScope as u8, Span { start: 0, end: 1}));

    Ok(asm)
}

pub fn compile_if(conditional: &ExprNode, if_body: &Block, else_body: &Block, is_expr: bool) -> Result<Vec<(u8, Span)>, String> {
    let mut asm = compile_expression(conditional)?;
    let conditional_check_idx = asm.len();
    asm.push((ByteCode::JmpZ as u8, Span { start:0, end:0}));
    asm.push((0, Span { start:0, end:0}));
    asm.push((0, Span { start:0, end:0}));
    asm.extend(compile_block(if_body, is_expr)?);
    asm.push((ByteCode::Jmp as u8, Span { start:0, end:0}));
    asm.push((0, Span { start:0, end:0}));
    asm.push((0, Span { start:0, end:0}));
    // We jump to here when the condition is false
    let else_case_start_idx = asm.len();

    asm.extend(compile_block(else_body, is_expr)?);

    // Offset from conditional to the else branch
    let else_jump_offset = u16_to_u8s(
        else_case_start_idx as u16 - conditional_check_idx as u16
    );
    asm[conditional_check_idx + 1].0 = else_jump_offset[0];
    asm[conditional_check_idx + 2].0 = else_jump_offset[1];

    // Offset from the end of the if branch to after the else branch
    let if_jump_offset = u16_to_u8s(
        asm.len() as u16 - else_case_start_idx as u16 + 3
    );

    asm[else_case_start_idx - 2].0 = if_jump_offset[0];
    asm[else_case_start_idx - 1].0 = if_jump_offset[1];

    Ok(asm)
}

pub fn compile_expression(expr: &ExprNode) -> Result<Vec<(u8, Span)>, String> {
    match &expr.data {
        Expression::Primary(Primary::None) => {
            let val = ShimValue::None;
            let mut res = vec![(ByteCode::LiteralShimValue as u8, expr.span)];
            for b in val.to_bytes().into_iter() {
                res.push((b, expr.span));
            }
            Ok(res)
        },
        Expression::Primary(Primary::Bool(b)) => {
            let val = ShimValue::Bool(*b);
            let mut res = vec![(ByteCode::LiteralShimValue as u8, expr.span)];
            for b in val.to_bytes().into_iter() {
                res.push((b, expr.span));
            }
            Ok(res)
        },
        Expression::Primary(Primary::Integer(i)) => {
            let val = ShimValue::Integer(*i);
            let mut res = vec![(ByteCode::LiteralShimValue as u8, expr.span)];
            for b in val.to_bytes().into_iter() {
                res.push((b, expr.span));
            }
            Ok(res)
        },
        Expression::Primary(Primary::Float(f)) => {
            let val = ShimValue::Float(*f);
            let mut res = vec![(ByteCode::LiteralShimValue as u8, expr.span)];
            for b in val.to_bytes().into_iter() {
                res.push((b, expr.span));
            }
            Ok(res)
        },
        Expression::Primary(Primary::Identifier(ident)) => {
            let mut res = Vec::new();
            res.push((ByteCode::VariableLoad as u8, expr.span));
            res.push((ident.len().try_into().expect("Ident len should into u8"), expr.span));
            for b in ident.into_iter() {
                res.push((*b, expr.span));
            }
            Ok(res)
        },
        Expression::Primary(Primary::String(s)) => {
            let mut res = Vec::new();
            res.push((ByteCode::LiteralString as u8, expr.span));
            res.push((s.len().try_into().expect("Ident should into u8"), expr.span));
            for b in s.into_iter() {
                res.push((*b, expr.span));
            }
            Ok(res)
        },
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
        },
        Expression::Primary(Primary::Expression(expr)) => {
            compile_expression(&expr)
        },
        Expression::BooleanOp(BooleanOp::And(a, b)) => {
            let mut asm = compile_expression(&a)?;
            asm.push((ByteCode::Copy as u8, Span { start:0, end:0}));

            let short_circuit_idx = asm.len();
            asm.push((ByteCode::JmpZ as u8, Span { start:0, end:0}));
            asm.push((0, Span { start:0, end:0}));
            asm.push((0, Span { start:0, end:0}));

            // Since the result of a is truthy we get rid of it and return the result of b
            asm.push((ByteCode::Pop as u8, Span { start:0, end:0}));

            asm.extend(compile_expression(&b)?);

            let short_circuit_offset = u16_to_u8s(
                asm.len() as u16 - short_circuit_idx as u16
            );
            asm[short_circuit_idx+1].0 = short_circuit_offset[0];
            asm[short_circuit_idx+2].0 = short_circuit_offset[1];

            Ok(asm)
        },
        Expression::BooleanOp(BooleanOp::Or(a, b)) => {
            let mut asm = compile_expression(&a)?;
            asm.push((ByteCode::Copy as u8, Span { start:0, end:0}));

            let short_circuit_idx = asm.len();
            asm.push((ByteCode::JmpNZ as u8, Span { start:0, end:0}));
            asm.push((0, Span { start:0, end:0}));
            asm.push((0, Span { start:0, end:0}));

            // Since the result of a is falsy we get rid of it and return the result of b
            asm.push((ByteCode::Pop as u8, Span { start:0, end:0}));

            asm.extend(compile_expression(&b)?);

            let short_circuit_offset = u16_to_u8s(
                asm.len() as u16 - short_circuit_idx as u16
            );
            asm[short_circuit_idx+1].0 = short_circuit_offset[0];
            asm[short_circuit_idx+2].0 = short_circuit_offset[1];

            Ok(asm)
        },
        Expression::BinaryOp(op) => {
            let (opcode, a, b) = match op {
                BinaryOp::Add(a, b) => (ByteCode::Add, a, b),
                BinaryOp::Subtract(a, b) => (ByteCode::Sub, a, b),
                BinaryOp::Equal(a, b) => (ByteCode::Equal, a, b),
                BinaryOp::NotEqual(a, b) => (ByteCode::NotEqual, a, b),
                BinaryOp::Multiply(a, b) => (ByteCode::Multiply, a, b),
                BinaryOp::Divide(a, b) => (ByteCode::Divide, a, b),
                BinaryOp::GT(a, b) => (ByteCode::GT, a, b),
                BinaryOp::GTE(a, b) => (ByteCode::GTE, a, b),
                BinaryOp::LT(a, b) => (ByteCode::LT, a, b),
                BinaryOp::LTE(a, b) => (ByteCode::LTE, a, b),
            };
            let mut res = compile_expression(&a)?;
            res.extend(compile_expression(&b)?);
            res.push(
                (opcode as u8, expr.span)
            );
            Ok(res)
        },
        Expression::UnaryOp(op) => {
            let (opcode, a) = match op {
                UnaryOp::Not(a) => (ByteCode::Not, a),
                UnaryOp::Negate(a) => (ByteCode::Negate, a),
            };
            let mut res = compile_expression(&a)?;
            res.push(
                (opcode as u8, expr.span)
            );
            Ok(res)
        },
        Expression::Call(expr, args) => {
            // First we evaluate the thing that needs to be called
            let mut res = compile_expression(&expr)?;

            // Then we evaluate each argument
            for arg_expr in args.iter() {
                res.extend(compile_expression(arg_expr)?);
            }
            // And the args become a list to be passed to the callable
            res.push((ByteCode::CreateList as u8, expr.span));
            res.push((0, expr.span));
            res.push((args.len() as u8, expr.span));

            res.push((ByteCode::Call as u8, expr.span));
            Ok(res)
        },
        Expression::Attribute(expr, ident) => {
            let mut res = compile_expression(&expr)?;
            res.push((ByteCode::GetAttr as u8, expr.span));
            res.push((ident.len().try_into().expect("Ident len should into u8"), expr.span));
            for b in ident.into_iter() {
                res.push((*b, expr.span));
            }
            Ok(res)
        },
        Expression::Block(block) => {
            compile_block(block, true)
        },
        Expression::If(conditional, if_body, else_body) => {
            compile_if(conditional, if_body, else_body, true)
        },
    }
}

impl Interpreter {
    pub fn create(config: &Config) -> Self {
        let mut mmu = MMU::with_capacity(Word((config.memory_space_bytes / 8).into()));
        let mut env = Environment::new();

        let builtins: &[(&[u8], Box<NativeFn>)] = &[
            (b"print", Box::new(shim_print)),
            (b"panic", Box::new(shim_panic)),
        ];

        for (name, func) in builtins {
            let position = mmu.alloc(Word(1.into()));
            unsafe {
                let ptr: *mut NativeFn = std::mem::transmute(
                    &mut mmu.mem[usize::from(position.0)]
                );
                *ptr = **func;
            }

            env.insert_new(
                name.to_vec(),
                ShimValue::NativeFn(position)
            );
        }

        Self {
            mem: mmu,
            source: HashMap::new(),
            env,
        }
    }

    pub fn execute_bytecode(&mut self, program: &Program) -> Result<(), String> {
        let mut pc = 0;

        // These are values that are operated on. Expressions push and pop to
        // this stack, return values go on this stack etc.
        let mut stack: Vec<ShimValue> = Vec::new();

        // This is the (PC, loop_info, scope_count) call stack
        let mut stack_frame: Vec<(usize, Vec<(usize, usize, usize)>, usize)> = Vec::new();

        // This is the PC of the (start, end, scope_count) of the current loop for the
        // current function
        let mut loop_info: Vec<(usize, usize, usize)> = Vec::new();

        // TODO: need stack depth to reset to on break
        // TODO: need stack depth to reset to on continue
        // TODO: need stack depth to reset to on return

        let bytes = &program.bytecode;
        while pc < bytes.len() {
            match bytes[pc] {
                val if val == ByteCode::Pop as u8 => {
                    stack.pop();
                },
                val if val == ByteCode::Add as u8 => {
                    let b = stack.pop().expect("Operand for add");
                    let a = stack.pop().expect("Operand for add");
                    stack.push(
                        a.add(&b)
                            .map_err(
                                |err_str| format_script_err(
                                    program.spans[pc],
                                    &program.script,
                                    &err_str
                                )
                            )?
                    );
                },
                val if val == ByteCode::Sub as u8 => {
                    let b = stack.pop().expect("Operand for add");
                    let a = stack.pop().expect("Operand for add");
                    stack.push(a.sub(&b)?);
                },
                val if val == ByteCode::Equal as u8 => {
                    let b = stack.pop().expect("Operand for add");
                    let a = stack.pop().expect("Operand for add");
                    stack.push(a.equal(self, &b)?);
                },
                val if val == ByteCode::NotEqual as u8 => {
                    let a = stack.pop().expect("Operand for ByteCode::NotEqual");
                    let b = stack.pop().expect("Operand for ByteCode::NotEqual");
                    stack.push(a.not_equal(self, &b)?);
                },
                val if val == ByteCode::Multiply as u8 => {
                    let a = stack.pop().expect("Operand for ByteCode::Multiply");
                    let b = stack.pop().expect("Operand for ByteCode::Multiply");
                    stack.push(a.mul(self, &b)?);
                },
                val if val == ByteCode::Divide as u8 => {
                    let a = stack.pop().expect("Operand for ByteCode::Divide");
                    let b = stack.pop().expect("Operand for ByteCode::Divide");
                    stack.push(a.div(self, &b)?);
                },
                val if val == ByteCode::GT as u8 => {
                    let a = stack.pop().expect("Operand for ByteCode::GT");
                    let b = stack.pop().expect("Operand for ByteCode::GT");
                    stack.push(a.gt(self, &b)?);
                },
                val if val == ByteCode::GTE as u8 => {
                    let a = stack.pop().expect("Operand for ByteCode::GTE");
                    let b = stack.pop().expect("Operand for ByteCode::GTE");
                    stack.push(a.gte(self, &b)?);
                },
                val if val == ByteCode::LT as u8 => {
                    let a = stack.pop().expect("Operand for ByteCode::LT");
                    let b = stack.pop().expect("Operand for ByteCode::LT");
                    stack.push(a.lt(self, &b)?);
                },
                val if val == ByteCode::LTE as u8 => {
                    let a = stack.pop().expect("Operand for ByteCode::LTE");
                    let b = stack.pop().expect("Operand for ByteCode::LTE");
                    stack.push(a.lte(self, &b)?);
                },
                val if val == ByteCode::Not as u8 => {
                    let a = stack.pop().expect("Operand for ByteCode::Not");
                    stack.push(a.not(self)?);
                },
                val if val == ByteCode::Negate as u8 => {
                    let a = stack.pop().expect("Operand for ByteCode::Negate");
                    stack.push(a.neg(self)?);
                },
                val if val == ByteCode::LiteralNone as u8 => {
                    stack.push(ShimValue::None);
                },
                val if val == ByteCode::Copy as u8 => {
                    stack.push(*stack.last().expect("non-empty stack"));
                },
                val if val == ByteCode::LoopStart as u8 => {
                    let loop_end = pc + (
                        ((bytes[pc+1] as usize) << 8) +
                        bytes[pc+2] as usize
                    );
                    loop_info.push(
                        (
                            pc + 3,
                            loop_end,
                            self.env.env_chain.len(),
                        )
                    );
                    pc += 2;
                },
                val if val == ByteCode::LoopEnd as u8 => {
                    loop_info.pop().expect("loop end should have loop info");
                },
                val if val == ByteCode::Break as u8 => {
                    let (_, end_pc, scope_count) = loop_info.last().expect("break should have loop info");
                    while self.env.env_chain.len() > *scope_count {
                        self.env.pop_scope().unwrap();
                    }
                    pc = *end_pc;
                    continue;
                },
                val if val == ByteCode::Continue as u8 => {
                    let (start_pc, _, scope_count) = loop_info.last().expect("continue should have loop info");
                    while self.env.env_chain.len() > *scope_count {
                        self.env.pop_scope().unwrap();
                    }
                    pc = *start_pc;
                    continue;
                },
                val if val == ByteCode::AssertLen as u8 => {
                    let len = bytes[pc+1] as usize;
                    if stack.is_empty() {
                        return Err(format!("stack is empty!"));
                    }
                    match stack[stack.len()-1] {
                        ShimValue::List(pos) => {
                            unsafe {
                                let ptr: *mut Vec<ShimValue> = std::mem::transmute(&mut self.mem.mem[usize::from(pos.0)]);
                                if (*ptr).len() != len {
                                    return Err(
                                        format_script_err(
                                            program.spans[
                                                stack_frame[stack_frame.len()-1].0
                                            ],
                                            &program.script,
                                            &format!("Function expects {} arguments, but got {}", len, (*ptr).len())
                                        )
                                    );
                                }
                            }
                        },
                        other => return Err(format!("Can't assert len on non-list {:?}", other)),
                    }
                    pc += 1;
                },
                val if val == ByteCode::Splat as u8 => {
                    if stack.is_empty() {
                        return Err(format!("stack is empty!"));
                    }
                    match stack.pop() {
                        Some(ShimValue::List(pos)) => {
                            unsafe {
                                let ptr: *mut Vec<ShimValue> = std::mem::transmute(&mut self.mem.mem[usize::from(pos.0)]);
                                for item in (*ptr).iter() {
                                    stack.push(*item);
                                }
                            }
                        },
                        other => return Err(format!("Can't assert len on non-list {:?}", other)),
                    }
                },
                val if val == ByteCode::LiteralShimValue as u8 => {
                    let bytes = [
                        bytes[pc+1],
                        bytes[pc+2],
                        bytes[pc+3],
                        bytes[pc+4],
                        bytes[pc+5],
                        bytes[pc+6],
                        bytes[pc+7],
                        bytes[pc+8],
                    ];
                    stack.push(ShimValue::from_bytes(bytes));
                    pc += 8;
                },
                val if val == ByteCode::LiteralString as u8 => {
                    let str_len = bytes[pc+1] as usize;
                    let contents = &bytes[pc+2..pc+2+str_len as usize];


                    const _: () = {
                        assert!(std::mem::size_of::<Vec<u8>>() == 24);
                    };
                    let word_count = Word(3.into());
                    let position = self.mem.alloc(word_count);
                    unsafe {
                        let ptr: *mut Vec<u8> = std::mem::transmute(&mut self.mem.mem[usize::from(position.0)]);
                        *ptr = contents.to_vec();
                    }

                    stack.push(ShimValue::String(position));
                    pc += 1 + str_len;
                },
                val if val == ByteCode::VariableDeclaration as u8 => {
                    let val = stack.pop().expect("Value for declaration");
                    let ident_len = bytes[pc+1] as usize;
                    let ident = &bytes[pc+2..pc+2+ident_len as usize];
                    self.env.insert_new(ident.to_vec(), val);
                    pc += 1 + ident_len;
                },
                val if val == ByteCode::Assignment as u8 => {
                    let val = stack.pop().expect("Value for assignment");
                    let ident_len = bytes[pc+1] as usize;
                    let ident = &bytes[pc+2..pc+2+ident_len as usize];

                    if !self.env.contains_key(ident) {
                        return Err(
                            format_script_err(
                                program.spans[pc],
                                &program.script,
                                &format!("Identifier {:?} not found", ident)
                            )
                        );
                    }
                    self.env.update(ident, val)?;

                    pc += 1 + ident_len;
                },
                val if val == ByteCode::VariableLoad as u8 => {
                    let ident_len = bytes[pc+1] as usize;
                    let ident = &bytes[pc+2..pc+2+ident_len as usize];
                    if let Some(value) = self.env.get(ident) {
                        stack.push(value);
                    } else {
                        return Err(
                            format_script_err(
                                program.spans[pc],
                                &program.script,
                                &format!("Unknown identifier {:?}", ident)
                            )
                        );
                    }
                    pc += 1 + ident_len;
                },
                val if val == ByteCode::GetAttr as u8 => {
                    let ident_len = bytes[pc+1] as usize;
                    let ident = &bytes[pc+2..pc+2+ident_len as usize];

                    let obj = stack.pop().expect("val to access");
                    stack.push(obj.get_attr(self, ident)?);

                    pc += 1 + ident_len;
                },
                val if val == ByteCode::SetAttr as u8 => {
                    let ident_len = bytes[pc+1] as usize;
                    let ident = &bytes[pc+2..pc+2+ident_len as usize];

                    let val = stack.pop().expect("val to assign");
                    let obj = stack.pop().expect("obj to set");
                    obj.set_attr(self, ident, val)
                        .map_err(
                            |err_str| format_script_err(
                                program.spans[pc],
                                &program.script,
                                &err_str
                            )
                        )?;

                    pc += 1 + ident_len;
                },
                val if val == ByteCode::Call as u8 => {
                    // When Call appears the args should already be in a list at
                    // the top of the stack, followed by the callable

                    // Remove the callable from the stack
                    let callable = stack.swap_remove(stack.len() - 2);

                    match callable.call(self, &mut stack)
                        .map_err(
                            |err_str| format_script_err(
                                program.spans[pc],
                                &program.script,
                                &err_str
                            )
                        )?
                    {
                        CallResult::ReturnValue(res) => stack.push(res),
                        CallResult::PC(new_pc) => {
                            stack_frame.push(
                                (
                                    pc+1,
                                    loop_info.clone(),
                                    self.env.env_chain.len(),
                                )
                            );
                            loop_info = Vec::new();
                            self.env.push_scope();
                            pc = new_pc as usize;
                            continue;
                        }
                    }
                },
                val if val == ByteCode::StartScope as u8 => {
                    self.env.push_scope();
                }
                val if val == ByteCode::EndScope as u8 => {
                    self.env.pop_scope()?;
                }
                val if val == ByteCode::Return as u8 => {
                    // The value at the top of the stack is the return value of
                    // the function, so we just need to pop the PC
                    let scope_count;
                    (pc, loop_info, scope_count) = stack_frame.pop().expect("stack frame to return to");
                    while self.env.env_chain.len() > scope_count {
                        self.env.pop_scope().unwrap();
                    }
                    continue;
                }
                val if val == ByteCode::JmpUp as u8 => {
                    let new_pc = pc - (
                        ((bytes[pc+1] as usize) << 8) +
                        bytes[pc+2] as usize
                    );
                    pc = new_pc;
                    continue;
                }
                val if val == ByteCode::Jmp as u8 => {
                    // TODO: signed jumps
                    let new_pc = pc +
                        ((bytes[pc+1] as usize) << 8) +
                        bytes[pc+2] as usize;
                    pc = new_pc;
                    continue;
                }
                val if val == ByteCode::JmpNZ as u8 => {
                    let conditional = stack.pop().expect("JMPNZ val to check");
                    if conditional.is_truthy(self)? {
                        // TODO: signed jumps
                        let new_pc = pc +
                            ((bytes[pc+1] as usize) << 8) +
                            bytes[pc+2] as usize;
                        pc = new_pc;
                        continue;
                    }
                    pc += 2;
                }
                val if val == ByteCode::JmpZ as u8 => {
                    let conditional = stack.pop().expect("JMP val to check");
                    if !conditional.is_truthy(self)? {
                        // TODO: signed jumps
                        let new_pc = pc +
                            ((bytes[pc+1] as usize) << 8) +
                            bytes[pc+2] as usize;
                        pc = new_pc;
                        continue;
                    }
                    pc += 2;
                }
                val if val == ByteCode::CreateList as u8 => {
                    let len = ((bytes[pc+1] as usize) << 8) + bytes[pc+2] as usize;

                    let word_count = Word(3.into());
                    let position = self.mem.alloc(word_count);
                    unsafe {
                        let ptr: *mut Vec<ShimValue> = std::mem::transmute(
                            &mut self.mem.mem[usize::from(position.0)]
                        );
                        *ptr = Vec::new();
                        for item in stack.drain(stack.len()-len..) {
                            (*ptr).push(item);
                        }
                    }
                    stack.push(ShimValue::List(position));

                    pc += 2;
                }
                val if val == ByteCode::CreateFn as u8 => {
                    let instruction_offset = ((bytes[pc+1] as u32) << 8) + bytes[pc+2] as u32;
                    stack.push(ShimValue::Fn(pc as u32 - instruction_offset));
                    pc += 2;
                }
                val if val == ByteCode::CreateStruct as u8 => {
                    // Everything after the first two bytes is data for the
                    // struct definition.
                    let new_pc = pc +
                        ((bytes[pc+1] as usize) << 8) +
                        bytes[pc+2] as usize;

                    let member_count = bytes[pc+3];
                    let method_count = bytes[pc+4];

                    let mut struct_table = Vec::new();

                    let mut idx = pc+5;
                    for member_idx in 0..member_count {
                        let ident_len = bytes[idx];
                        let ident = &bytes[idx+1..idx+1+ident_len as usize];
                        struct_table.push(
                            (
                                ident.to_vec(),
                                StructAttribute::MemberInstanceOffset(member_idx),
                            )
                        );
                        idx = idx + 1 + ident_len as usize;
                    }

                    for method_idx in 0..method_count {
                        let method_pc = pc +
                            ((bytes[idx] as usize) << 8) +
                            bytes[idx+1] as usize;

                        idx += 2;

                        let ident_len = bytes[idx];
                        let ident = &bytes[idx+1..idx+1+ident_len as usize];
                        struct_table.push(
                            (
                                ident.to_vec(),
                                StructAttribute::MethodDefPC(method_pc as u32),
                            )
                        );
                        idx = idx + 1 + ident_len as usize;
                    }
                    const _: () = {
                        assert!(std::mem::size_of::<StructDef>() == 32);
                    };
                    let pos = self.mem.alloc(Word(4.into()));
                    let def: &mut StructDef = unsafe { self.mem.get_mut(pos) };

                    *def = StructDef {
                        member_count: member_count,
                        method_count: method_count,
                        lookup: struct_table,
                    };

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
        Ok(())
    }
}

pub fn print_asm(bytes: &[u8]) {
    for (idx, b) in bytes.iter().enumerate() {
        eprint!("{idx:4}:  {b:3}  ");

        let text = printable_byte(*b);
        if !text.starts_with('\\') {
            eprint!("{:4} ", text);
        }

        if *b == ByteCode::Jmp as u8 {
            eprint!("JMP");
        } else if *b == ByteCode::VariableDeclaration as u8 {
            eprint!("let");
        } else if *b == ByteCode::Assignment as u8 {
            eprint!("assignment");
        } else if *b == ByteCode::Call as u8 {
            eprint!("call");
        } else if *b == ByteCode::Add as u8 {
            eprint!("add");
        } else if *b == ByteCode::CreateFn as u8 {
            eprint!("fn");
        } else if *b == ByteCode::JmpZ as u8 {
            eprint!("JMPZ");
        } else if *b == ByteCode::JmpNZ as u8 {
            eprint!("JMPNZ");
        } else if *b == ByteCode::JmpUp as u8 {
            eprint!("JMPUP");
        } else if *b == ByteCode::AssertLen as u8 {
            eprint!("assert_len");
        } else if *b == ByteCode::Splat as u8 {
            eprint!("SPLAT");
        } else if *b == ByteCode::CreateFn as u8 {
            eprint!("fn");
        } else if *b == ByteCode::CreateStruct as u8 {
            eprint!("create struct");
        } else if *b == ByteCode::GetAttr as u8 {
            eprint!("get");
        } else if *b == ByteCode::SetAttr as u8 {
            eprint!("set");
        } else if *b == ByteCode::VariableLoad as u8 {
            eprint!("load");
        } else if *b == ByteCode::Break as u8 {
            eprint!("break");
        } else if *b == ByteCode::Continue as u8 {
            eprint!("continue");
        } else if *b == ByteCode::LiteralShimValue as u8 {
            eprint!("ShimValue");
        } else if *b == ByteCode::LiteralString as u8 {
            eprint!("String");
        } else if *b == ByteCode::LiteralNone as u8 {
            eprint!("None");
        } else if *b == ByteCode::Copy as u8 {
            eprint!("Copy");
        } else if *b == ByteCode::LoopStart as u8 {
            eprint!("Loop Start");
        } else if *b == ByteCode::LoopEnd as u8 {
            eprint!("Loop End");
        } else if *b == ByteCode::StartScope as u8 {
            eprint!("start_scope");
        } else if *b == ByteCode::EndScope as u8 {
            eprint!("end_scope");
        } else if *b == ByteCode::Pad0 as u8 {
            eprint!("");
        } else if *b == ByteCode::Pad1 as u8 {
            eprint!("");
        } else if *b == ByteCode::Pad2 as u8 {
            eprint!("");
        } else if *b == ByteCode::Pad3 as u8 {
            eprint!("");
        } else if *b == ByteCode::Pad4 as u8 {
            eprint!("");
        } else if *b == ByteCode::Pad5 as u8 {
            eprint!("");
        } else if *b == ByteCode::Pad6 as u8 {
            eprint!("");
        } else if *b == ByteCode::Pad7 as u8 {
            eprint!("");
        } else if *b == ByteCode::Pad8 as u8 {
            eprint!("");
        } else if *b == ByteCode::Pad9 as u8 {
            eprint!("");
        } else if *b == ByteCode::Return as u8 {
            eprint!("return");
        }
        eprintln!();
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        let config = Config::default();
        Self::create(&config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u24_conversion() {
        assert_eq!(
            u24::from(1u32),
            u24([0, 0, 1])
        );
        assert_eq!(
            u32::from(u24::from(1u32)),
            1u32
        );

        assert_eq!(
            u24::from(1u32).0,
            [0, 0, 1]
        );
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
const _todo: u8 = 42;
