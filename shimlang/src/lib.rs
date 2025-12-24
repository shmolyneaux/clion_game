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
pub enum BinaryOp {
    Add(Box<ExprNode>, Box<ExprNode>),
    Subtract(Box<ExprNode>, Box<ExprNode>),
}

#[derive(Debug)]
pub enum Expression {
    Primary(Primary),
    BinaryOp(BinaryOp),
    Call(Box<ExprNode>, Vec<ExprNode>),
}

#[derive(Debug)]
pub enum Statement {
    Let(Vec<u8>, ExprNode),
    If(ExprNode, Vec<Statement>, Vec<Statement>),
    Fn(Vec<u8>, Vec<Vec<u8>>, Vec<Statement>),
    Expression(ExprNode),
    Return(Option<ExprNode>),
}

#[derive(Debug)]
pub struct Ast {
    stmts: Vec<Statement>,
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
    Let,
    Fn,
    If,
    Else,
    Struct,
    Return,
    Equal,
    Semicolon,
    LSquare,
    RSquare,
    LAngle,
    RAngle,
    LCurly,
    RCurly,
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
            Err(format!("Expected token {:?} but found {:?}", expected, value))
        }
    }

    fn advance(&mut self) -> Result<(), String> {
        self.pop()?;
        Ok(())
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

pub fn parse_primary(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let span = tokens.peek_span()?;
    let expr: Expression = match tokens.pop()? {
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
        Token::LBracket => {
            let expr = parse_expression(tokens)?;
            tokens.consume(Token::RBracket)?;
            expr.data
        },
        Token::LSquare => {
            let items = parse_arguments(tokens, Token::RSquare)?;
            Expression::Primary(Primary::List(items))
        },
        token => {
            return Err(tokens.format_peek_err(
                &format!(
                    "Unexpected `{:?}` in parse_primary",
                    // TODO: should display the exact character like `[` rather than name line `SemiColon`
                    token
                )
            ))
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
                if !tokens.is_empty() && *tokens.peek()? == Token::RBracket {
                    // Exit when there's a trailing comma
                    break;
                }
                continue;
            }
            token => return Err(format!("Expected comma or closing bracket, found {:?}", token)),
        }
    }
    Ok(args)
}

pub fn parse_call(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_term(tokens)?;
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
            _ => return Ok(expr),
        }
    }

    Ok(expr)
}

pub fn parse_term(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    let mut expr = parse_primary(tokens)?;
    while !tokens.is_empty() {
        // TODO: this is just the operator
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

pub fn parse_expression(tokens: &mut TokenStream) -> Result<ExprNode, String> {
    parse_call(tokens)
}

pub fn parse_ast(tokens: &mut TokenStream) -> Result<Ast, String> {
    let mut stmts = Vec::new();
    while !tokens.is_empty() {
        stmts.push(parse_statement(tokens)?);
    }
    Ok(Ast { stmts: stmts, script: tokens.script.clone() })
}

pub fn parse_statement(tokens: &mut TokenStream) -> Result<Statement, String> {
    if *tokens.peek()? == Token::Let {
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
            token => return Err(format!("Expected semicolon after `let <ident> = <expr>`, found {:?}", token))
        }

        Ok(Statement::Let(ident, expr))
    } else if *tokens.peek()? == Token::Fn {
        tokens.advance()?;
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
        tokens.consume(Token::LCurly)?;

        let mut body = Vec::new();
        while !tokens.is_empty() && *tokens.peek()? != Token::RCurly {
            body.push(parse_statement(tokens)?);
        }

        // For now, last line should always be a return
        // Later when the last line doesn't have a semicolon we can do this
        // better, but that might happen at the AST-creation stage?
        match body.last() {
            Some(Statement::Return(_)) => (),
            _ => body.push(Statement::Return(None)),
        }

        tokens.consume(Token::RCurly)?;

        Ok(Statement::Fn(ident, params, body))
    } else if *tokens.peek()? == Token::If {
        tokens.advance()?;
        let conditional = parse_expression(tokens)?;

        tokens.consume(Token::LCurly)?;
        let mut if_stmts = Vec::new();
        let mut else_stmts = Vec::new();
        while *tokens.peek()? != Token::RCurly {
            if_stmts.push(parse_statement(tokens)?);
        }
        tokens.consume(Token::RCurly)?;

        if *tokens.peek()? == Token::Else {
            tokens.advance()?;
            if *tokens.peek()? == Token::If {
                else_stmts.push(parse_statement(tokens)?);
            } else {
                tokens.consume(Token::LCurly)?;
                while *tokens.peek()? != Token::RCurly {
                    else_stmts.push(parse_statement(tokens)?);
                }
                tokens.consume(Token::RCurly)?;
            }
        }

        Ok(Statement::If(conditional, if_stmts, else_stmts))
    } else if *tokens.peek()? == Token::Struct {
        tokens.advance()?;
        let _ident = match tokens.pop()? {
            Token::Identifier(ident) => {
                ident.clone()
            },
            token => return Err(format!("Expected ident after struct, found {:?}", token))
        };
        Err(
            tokens.format_peek_err("Struct declarations not implemented yet")
        )
    } else if *tokens.peek()? == Token::Return {
        tokens.advance()?;
        if *tokens.peek()? == Token::Semicolon {
            tokens.advance()?;
            Ok(Statement::Return(None))
        } else {
            let expr = parse_expression(tokens)?;
            match tokens.pop()? {
                Token::Semicolon => (),
                token => return Err(format!("Expected semicolon after `let <ident> = <expr>`, found {:?}", token))
            }
            Ok(Statement::Return(Some(expr)))
        }
    } else {
        let expr = parse_expression(tokens)?;

        match tokens.peek()? {
            Token::Semicolon => {tokens.pop()?;},
            token => return Err(
                tokens.format_peek_err(
                    &format!(
                        "Expected semicolon after expression statement, found {:?}",
                        token
                    )
                )
            )
        }

        Ok(Statement::Expression(expr))
    }
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
            b'a' ..= b'z' | b'A' ..= b'Z' | b'0' ..= b'9' => continue,
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
            b'a' ..= b'z' | b'A' ..= b'Z' => {
                let ident = lex_identifier(&mut text)?;
                if ident == b"let" {
                    tokens.push(Token::Let);
                } else if ident == b"fn" {
                    tokens.push(Token::Fn);
                } else if ident == b"if" {
                    tokens.push(Token::If);
                } else if ident == b"else" {
                    tokens.push(Token::Else);
                } else if ident == b"struct" {
                    tokens.push(Token::Struct);
                } else if ident == b"return" {
                    tokens.push(Token::Return);
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
            b'=' => tokens.push(Token::Equal),
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

/**
 * The interpreter stores memory in 8-byte words. Each `Word` is
 * an index into the interpreter memory.
 */
#[cfg_attr(feature = "facet", derive(Facet))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Word(pub u32);

impl Add<u32> for Word {
    type Output = Word;

    fn add(self, rhs: u32) -> Word {
        Word(self.0 + rhs)
    }
}

impl Sub<u32> for Word {
    type Output = Word;

    fn sub(self, rhs: u32) -> Word {
        Word(self.0 - rhs)
    }
}

impl Add<Word> for Word {
    type Output = Word;

    fn add(self, rhs: Word) -> Word {
        Word(self.0 + rhs.0)
    }
}

impl Sub<Word> for Word {
    type Output = Word;

    fn sub(self, rhs: Word) -> Word {
        Word(self.0 - rhs.0)
    }
}

impl AddAssign<u32> for Word {
    fn add_assign(&mut self, rhs: u32) {
        self.0 += rhs;
    }
}

impl SubAssign<u32> for Word {
    fn sub_assign(&mut self, rhs: u32) {
        self.0 -= rhs;
    }
}

impl AddAssign<Word> for Word {
    fn add_assign(&mut self, rhs: Word) {
        self.0 += rhs.0;
    }
}

impl SubAssign<Word> for Word {
    fn sub_assign(&mut self, rhs: Word) {
        self.0 -= rhs.0;
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

impl MMU {
    fn with_capacity(word_count: Word) -> Self {
        let mem = vec![0; word_count.0 as usize];
        let free_list = vec![FreeBlock::new(Word(0), word_count)];
        Self {
            mem: mem,
            free_list: free_list,
        }
    }

    fn compact_free_list() {
        todo!("compact_free_list not implemented");
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
            }
        }
        panic!("Could not allocate {:?} words from free list {:#?}", words, self.free_list);
    }

    fn free(&mut self, _words: u32, _ptr: *const u64) {
    }
}

// TODO: uncomment #[derive(Facet)]
pub struct Interpreter {
    pub mem: MMU,
    pub source: HashMap<String, String>,
    pub env: HashMap<Vec<u8>, u64>,
}

#[derive(Copy, Clone, Debug)]
pub enum ShimValue {
    None,
    Print,
    Integer(i32),
    Float(f32),
    Bool(bool),
    // This is a program counter
    Fn(u32),
    // TODO: it seems like this should point to a more generic reference-counted
    // object type that all non-value types share
    String(Word),
    List(Word),
}

use std::mem::{size_of};
const _: () = {
    assert!(std::mem::size_of::<ShimValue>() <= 8);
};

fn format_float(val: f32) -> String {
    let s = format!("{val}");
    if !s.contains('.') && !s.contains('e') {
        format!("{s}.0")
    } else {
        s
    }
}

enum CallResult {
    BuiltinReturnValue(ShimValue),
    PC(u32),
}

impl ShimValue {
    fn call(&self, interpreter: &mut Interpreter, stack: &mut Vec<ShimValue>) -> Result<CallResult, String> {
        match self {
            ShimValue::None => Err(format!("Can't call None as function")),
            ShimValue::Print => {
                let arg_pos: Word = match stack.pop() {
                    Some(ShimValue::List(args)) => args,
                    args => return Err(format!("Can't call print with non-list args {:?}", args)),
                };

                unsafe {
                    let ptr: *mut Vec<ShimValue> = std::mem::transmute(&mut interpreter.mem.mem[arg_pos.0 as usize]);
                    for (idx, arg) in (*ptr).iter().enumerate() {
                        if idx != 0 {
                            print!(" ");
                        }
                        print!("{}", arg.to_string(interpreter));
                    }
                }

                println!();
                Ok(CallResult::BuiltinReturnValue(ShimValue::None))
            },
            ShimValue::Fn(pc) => {
                Ok(CallResult::PC(*pc))
            }
            other => Err(format!("Can't call value {:?} as function", other)),
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
                    let ptr: *mut Vec<u8> = std::mem::transmute(&mut interpreter.mem.mem[position.0 as usize]);
                    String::from_utf8((*ptr).clone()).expect("valid utf-8 string stored")
                }
            },
            ShimValue::List(position) => {
                let mut out = "[".to_string();
                unsafe {
                    let ptr: *mut Vec<ShimValue> = std::mem::transmute(&mut interpreter.mem.mem[position.0 as usize]);
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
                    let ptr: *mut Vec<u8> = std::mem::transmute(&mut interpreter.mem.mem[position.0 as usize]);
                    Ok(!(*ptr).is_empty())
                }
            },
            ShimValue::List(position) => {
                unsafe {
                    let ptr: *mut Vec<ShimValue> = std::mem::transmute(&mut interpreter.mem.mem[position.0 as usize]);
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
    AssertLen,
    Splat,
    Pop,
    Add,
    Sub,
    // Or,
    // Not,
    // And,
    // ToString,
    // ToBool,
    // JumpZ,
    // JumpNZ,
    LiteralShimValue,
    LiteralString,
    LiteralNone,
    CreateFn,
    CreateList,
    VariableDeclaration,
    VariableLoad,
    Call,
    Return,
    Jmp,
    JmpNZ,
    JmpZ,
}

pub struct Program {
    bytecode: Vec<u8>,
    spans: Vec<Span>,
    script: Vec<u8>,
}

pub fn compile_ast(ast: &Ast) -> Result<Program, String> {
    let mut program: Vec<(u8, Span)> = Vec::new();
    for stmt in ast.stmts.iter() {
        program.extend(compile_statement(stmt)?);
    }
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
        Statement::Fn(ident, params, body) => {
            // This will be replaced with a relative jump to after the function
            // declaration
            let mut asm = vec![
                (ByteCode::Jmp as u8, Span {start: 0, end: 1}),
                (0, Span {start: 0, end: 1}),
                (0, Span {start: 0, end: 1}),
            ];
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

            for stmt in body {
                asm.extend(compile_statement(stmt)?);
            }
            // Note: we know that last statement is a return at the AST-creation stage,
            // we we know we'll jump back to the return address

            if asm.len() > u16::MAX as usize {
                return Err(format!("Function has more than {} instructions", u16::MAX));
            }

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
        Statement::Return(expr) => {
            let mut res = Vec::new();
            if let Some(expr) = expr {
                res.extend(compile_expression(expr)?);
            } else {
                res.push((ByteCode::LiteralNone as u8, Span { start: 0, end: 1}));
            }
            res.push((ByteCode::Return as u8, Span { start: 0, end: 1}));
            Ok(res)
        },
        Statement::If(conditional, if_stmts, else_stmts) => {
            let mut asm = compile_expression(conditional)?;
            let conditional_check_idx = asm.len();
            asm.push((ByteCode::JmpZ as u8, Span { start:0, end:0}));
            asm.push((0, Span { start:0, end:0}));
            asm.push((0, Span { start:0, end:0}));
            for stmt in if_stmts {
                asm.extend(compile_statement(stmt)?);
            }
            asm.push((ByteCode::Jmp as u8, Span { start:0, end:0}));
            asm.push((0, Span { start:0, end:0}));
            asm.push((0, Span { start:0, end:0}));
            // We jump to here when the condition is false
            let else_case_start_idx = asm.len();

            for stmt in else_stmts {
                asm.extend(compile_statement(stmt)?);
            }

            // Offset from conditional to the else branch
            let else_jump_offset = u16_to_u8s(
                else_case_start_idx as u16 - conditional_check_idx as u16
            );
            asm[conditional_check_idx + 1].0 = else_jump_offset[0];
            asm[conditional_check_idx + 2].0 = else_jump_offset[1];

            // Offset from the end of the if branch to after the else branch
            let if_jump_offset = u16_to_u8s(
                else_case_start_idx as u16 - asm.len() as u16 + 3
            );
            asm[else_case_start_idx - 2].0 = if_jump_offset[0];
            asm[else_case_start_idx - 1].0 = if_jump_offset[1];

            Ok(asm)
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
            res.push((ident.len().try_into().expect("Ident should into u8"), expr.span));
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
        Expression::BinaryOp(op) => {
            match op {
                BinaryOp::Add(a, b) => {
                    let mut res = compile_expression(&a)?;
                    res.extend(compile_expression(&b)?);
                    res.push((ByteCode::Add as u8, expr.span));
                    Ok(res)
                },
                BinaryOp::Subtract(a, b) => {
                    let mut res = compile_expression(&a)?;
                    res.extend(compile_expression(&b)?);
                    res.push((ByteCode::Sub as u8, expr.span));
                    Ok(res)
                },
            }
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
    }
}

impl Interpreter {
    pub fn create(config: &Config) -> Self {
        let mmu = MMU::with_capacity(Word(config.memory_space_bytes / 8));

        Self {
            mem: mmu,
            source: HashMap::new(),
            env: HashMap::new(),
        }
    }

    pub fn execute_bytecode(&mut self, program: &Program) -> Result<(), String> {
        let mut pc = 0;
        let mut stack: Vec<ShimValue> = Vec::new();
        let mut stack_frame: Vec<usize> = Vec::new();

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
                val if val == ByteCode::LiteralNone as u8 => {
                    stack.push(ShimValue::None);
                },
                val if val == ByteCode::AssertLen as u8 => {
                    let len = bytes[pc+1] as usize;
                    if stack.is_empty() {
                        return Err(format!("stack is empty!"));
                    }
                    match stack[stack.len()-1] {
                        ShimValue::List(pos) => {
                            unsafe {
                                let ptr: *mut Vec<ShimValue> = std::mem::transmute(&mut self.mem.mem[pos.0 as usize]);
                                if (*ptr).len() != len {
                                    return Err(format!("len mismatch {} {}", (*ptr).len(), len));
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
                                let ptr: *mut Vec<ShimValue> = std::mem::transmute(&mut self.mem.mem[pos.0 as usize]);
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
                    let word_count = Word(3);
                    let position = self.mem.alloc(word_count);
                    unsafe {
                        let ptr: *mut Vec<u8> = std::mem::transmute(&mut self.mem.mem[position.0 as usize]);
                        *ptr = contents.to_vec();
                    }

                    stack.push(ShimValue::String(position));
                    pc += 1 + str_len;
                },
                val if val == ByteCode::VariableDeclaration as u8 => {
                    let val = stack.pop().expect("Value for declaration");
                    let ident_len = bytes[pc+1] as usize;
                    let ident = &bytes[pc+2..pc+2+ident_len as usize];
                    self.env.insert(ident.to_vec(), val.to_u64());
                    pc += 1 + ident_len;
                },
                val if val == ByteCode::VariableLoad as u8 => {
                    let ident_len = bytes[pc+1] as usize;
                    let ident = &bytes[pc+2..pc+2+ident_len as usize];
                    if ident == b"print" {
                        stack.push(
                            ShimValue::Print
                        );
                    } else if ident == b"true" {
                        stack.push(
                            ShimValue::Bool(true)
                        );
                    } else if ident == b"false" {
                        stack.push(
                            ShimValue::Bool(false)
                        );
                    } else if let Some(value) = self.env.get(ident) {
                        stack.push(
                            unsafe { ShimValue::from_u64(*value) }
                        );
                    } else {
                        return Err(format!("Unknown identifier {:?}", ident));
                    }
                    pc += 1 + ident_len;
                },
                val if val == ByteCode::Call as u8 => {
                    // When Call appears the args should already be in a list at
                    // the top of the stack, followed by the callable

                    // Remove the callable from the stack
                    let callable = stack.swap_remove(stack.len() - 2);

                    match callable.call(self, &mut stack)? {
                        CallResult::BuiltinReturnValue(res) => stack.push(res),
                        CallResult::PC(new_pc) => {
                            stack_frame.push(pc+1);
                            pc = new_pc as usize;
                            continue;
                        }
                    }
                },
                val if val == ByteCode::Return as u8 => {
                    // The value at the top of the stack is the return value of
                    // the function, so we just need to pop the PC
                    pc = stack_frame.pop().expect("stack frame to return to");
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
                    let conditional = stack.pop().expect("val to check");
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
                    let conditional = stack.pop().expect("val to check");
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

                    let word_count = Word(3);
                    let position = self.mem.alloc(word_count);
                    unsafe {
                        let ptr: *mut Vec<ShimValue> = std::mem::transmute(
                            &mut self.mem.mem[position.0 as usize]
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
                b => {
                    eprintln!("create fn code {}", ByteCode::CreateFn as u8);
                    for (idx, b) in bytes.iter().enumerate() {
                        eprint!("{idx}:  {b}  ");
                        if *b == ByteCode::Jmp as u8 {
                            eprint!("JMP");
                        } else if *b == ByteCode::VariableDeclaration as u8 {
                            eprint!("let");
                        } else if *b == ByteCode::Call as u8 {
                            eprint!("call");
                        } else if *b == ByteCode::CreateFn as u8 {
                            eprint!("fn");
                        } else if *b == ByteCode::JmpZ as u8 {
                            eprint!("JMPZ");
                        } else if *b == ByteCode::JmpNZ as u8 {
                            eprint!("JMPNZ");
                        } else if *b == ByteCode::CreateFn as u8 {
                            eprint!("fn");
                        } else if *b == ByteCode::Return as u8 {
                            eprint!("return");
                        }
                        eprintln!();
                    }
                    return Err(format!("Unknown bytecode {b} at PC {pc}"));
                }
            }
            pc += 1;
        }
        Ok(())
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
    fn it_works() {
        assert_eq!(2+2, 4);
    }
}
