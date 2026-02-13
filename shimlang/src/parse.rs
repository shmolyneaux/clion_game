use std::ops::Add;
use crate::lex::{Token, TokenStream, lex, format_script_err};

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
    FloorDivide(Box<ExprNode>, Box<ExprNode>),
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
    pub(crate) stmts: Vec<StatementNode>,
    pub(crate) last_expr: Option<Box<ExprNode>>,
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
    pub(crate) ident: Option<Vec<u8>>,
    pub(crate) pos_args_required: Vec<Vec<u8>>,
    pub(crate) pos_args_optional: Vec<(Vec<u8>, ExprNode)>,
    pub(crate) body: Block,
}

#[derive(Debug)]
pub struct Struct {
    pub(crate) ident: Vec<u8>,
    pub(crate) members_required: Vec<Vec<u8>>,
    pub(crate) members_optional: Vec<(Vec<u8>, ExprNode)>,
    pub(crate) methods: Vec<Fn>,
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
    pub(crate) block: Block,
    pub(crate) script: Vec<u8>,
}

pub struct Conditional {
    pub(crate) conditional: ExprNode,
    pub(crate) if_body: Block,
    pub(crate) else_body: Block,
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
                let end_span = tokens.previous_span()?;
                let start = expr.span;
                expr = Node {
                    data: Expression::Index(Box::new(expr), Box::new(index_expr)),
                    span: start + end_span,
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
            Token::SlashSlash => {
                tokens.advance()?;
                expr = Node {
                    data: Expression::BinaryOp(BinaryOp::FloorDivide(
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
                return Err(format_script_err(
                    start_span,
                    &tokens.script,
                    "No token found after let",
                ));
            }
            let ident = match tokens.pop()? {
                Token::Identifier(ident) => ident.clone(),
                token => {
                    tokens.unadvance()?;
                    return Err(tokens.format_peek_err(&format!("Expected ident after let, found {:?}", token)));
                }
            };

            match tokens.pop()? {
                Token::Equal => (),
                token => {
                    tokens.unadvance()?;
                    return Err(tokens.format_peek_err(&format!("Expected = after `let ident`, found {:?}", token)));
                }
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
                    let next_span = tokens.peek_span()?;
                    return Err(format_script_err(
                        expr.span + next_span,
                        &tokens.script,
                        &format!(
                            "Expected semicolon after expression statement, found {:?}",
                            token
                        ),
                    ));
                }
            }
        };

        stmts.push(stmt);
    }

    Ok(Block { stmts, last_expr })
}

pub fn ast_from_text(text: &[u8]) -> Result<Ast, String> {
    let mut tokens = lex(text)?;
    parse_ast(&mut tokens)
}
