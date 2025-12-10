use std::collections::HashMap;

#[cfg(feature = "facet")]
use facet::Facet;

use std::ops::{Add, Sub};
use std::ops::{AddAssign, SubAssign};

#[derive(Debug)]
pub enum Primary {
    True,
    False,
    None,
    Integer(i32),
    Float(f32),
    Identifier(String),
    Expression(Box<Expression>),
}

#[derive(Debug)]
pub enum BinaryOp {
    Add(Box<Expression>, Box<Expression>),
    Subtract(Box<Expression>, Box<Expression>),
}

#[derive(Debug)]
pub enum Expression {
    Primary(Primary),
    BinaryOp(BinaryOp),
    Call(Box<Expression>, Vec<Expression>),
}

#[derive(Debug)]
pub enum Statement {
    Let(String, Expression),
    Expression(Expression),
}

#[derive(Debug)]
pub struct Program {
    stmts: Vec<Statement>
}

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Comma,
    LBracket,
    RBracket,
    Plus,
    Minus,
    Let,
    Equal,
    Semicolon,
    Integer(i32),
    Identifier(String),
}

pub struct TokenStream {
    idx: usize,
    tokens: Vec<Token>,
}

impl TokenStream {
    /**
     * Return the next token (if there are tokens remaining) without advancing the stream
     */
    fn peek(&self) -> Result<&Token, String> {
        if self.is_empty() {
            Err("End of token stream".to_string())
        } else {
            Ok(&self.tokens[self.idx])
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
}

pub fn parse_primary(tokens: &mut TokenStream) -> Result<Expression, String> {
    match tokens.peek()? {
        Token::Integer(i) => {
            let result = Ok(Expression::Primary(Primary::Integer(*i)));
            tokens.advance()?;
            result
        },
        Token::Identifier(s) => {
            let result = Ok(Expression::Primary(Primary::Identifier(s.clone())));
            tokens.advance()?;
            result
        },
        Token::LBracket => {
            tokens.advance()?;
            let expr = parse_expression(tokens)?;
            tokens.consume(Token::RBracket)?;
            Ok(expr)
        },
        token => Err(format!("Could not parse_primary {:?}", token)),
    }
}

pub fn parse_arguments(tokens: &mut TokenStream) -> Result<Vec<Expression>, String> {
    let mut args = Vec::new();
    while !tokens.is_empty() {
        let expr = parse_expression(tokens)?;
        args.push(expr);

        match tokens.peek()? {
            Token::RBracket => {
                // Don't consume the closing bracket since we parse
                // function arguments and list literals the same way
                // and the parent needs to match the bracket
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
            token => return Err(format!("Expected command or closing bracket, found {:?}", token)),
        }
    }
    Ok(args)
}

pub fn parse_call(tokens: &mut TokenStream) -> Result<Expression, String> {
    let mut expr = parse_term(tokens)?;
    while !tokens.is_empty() {
        match *tokens.peek()? {
            Token::LBracket => {
                tokens.advance()?;
                expr = Expression::Call(Box::new(expr), parse_arguments(tokens)?);
                tokens.consume(Token::RBracket)?;
            },
            _ => return Ok(expr),
        }
    }

    Ok(expr)
}

pub fn parse_term(tokens: &mut TokenStream) -> Result<Expression, String> {
    let mut expr = parse_primary(tokens)?;
    while !tokens.is_empty() {
        match tokens.peek()? {
            Token::Plus => {
                tokens.advance()?;
                expr = Expression::BinaryOp(BinaryOp::Add(Box::new(expr), Box::new(parse_expression(tokens)?)));
            },
            Token::Minus => {
                tokens.advance()?;
                expr = Expression::BinaryOp(BinaryOp::Subtract(Box::new(expr), Box::new(parse_expression(tokens)?)));
            },
            _ => return Ok(expr),
        }
    }
    Ok(expr)
}

pub fn parse_expression(tokens: &mut TokenStream) -> Result<Expression, String> {
    parse_call(tokens)
}

pub fn parse_program(tokens: &mut TokenStream) -> Result<Program, String> {
    let mut stmts = Vec::new();
    while !tokens.is_empty() {
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

            stmts.push(Statement::Let(ident, expr));
        }
        else {
            let expr = parse_expression(tokens)?;

            match tokens.pop()? {
                Token::Semicolon => (),
                token => return Err(format!("Expected semicolon after expression statement, found {:?}", token))
            }

            stmts.push(Statement::Expression(expr));
        }
    }
    Ok(Program { stmts: stmts })
}

pub fn printable_byte(b: u8) -> String {
    match char::from_u32(b as u32) {
        Some(c) if !c.is_control() => c.to_string(),
        _ => format!("\\x{:02X}", b),
    }
}

pub fn lex_identifier(text: &mut &[u8]) -> Result<String, String> {
    for (idx, c) in text.iter().enumerate() {
        match c {
            b'a' ..= b'z' | b'A' ..= b'Z' | b'0' ..= b'9' => continue,
            _ => {
                let ident = unsafe { String::from_utf8_unchecked(text[0..idx].to_vec())};
                *text = &text[(idx-1)..];
                return Ok(ident);
            }
        }
    }
    unsafe {
        Ok(String::from_utf8_unchecked(text.to_vec()))
    }
}

pub fn lex_number(text: &mut &[u8]) -> Result<Token, String> {
    let found_decimal = false;
    for (idx, c) in text.iter().enumerate() {
        match c {
            b'0' ..= b'9' => continue,
            b'.' => {
                if found_decimal {
                    return Err(format!("Found multiple decimals in number"));
                }
            }
            _ => {
                let token = if found_decimal {
                    todo!("Implement float parsing");
                } else {
                    Token::Integer(
                        unsafe {
                            std::str::from_utf8_unchecked(&text[..idx]).parse().map_err(|e| format!("{:?}", e))?
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

pub fn lex(text: &[u8]) -> Result<Vec<Token>, String> {
    let mut text = text;
    let mut tokens = Vec::new();

    while !text.is_empty() {
        let c = text[0];
        match c {
            b'a' ..= b'z' => {
                let ident = lex_identifier(&mut text)?;
                if ident == "let" {
                    tokens.push(Token::Let);
                } else {
                    tokens.push(Token::Identifier(ident))
                }
            },
            b'0' ..= b'9' => tokens.push(lex_number(&mut text)?),
            b'(' => tokens.push(Token::LBracket),
            b')' => tokens.push(Token::RBracket),
            b'+' => tokens.push(Token::Plus),
            b'-' => tokens.push(Token::Minus),
            b'=' => tokens.push(Token::Equal),
            b';' => tokens.push(Token::Semicolon),
            b'\n' => (),
            b' ' => (),
            _ => return Err(format!("Unknown character '{}'", printable_byte(c)))
        }
        text = &text[1..];
    }
    Ok(tokens)
}

pub fn ast_from_text(text: &[u8]) -> Result<Program, String> {
    let tokens = lex(text)?;
    let mut tokens = TokenStream {
        idx: 0,
        tokens: tokens,
    };
    parse_program(&mut tokens)
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

#[derive(Facet)]
pub struct Interpreter {
    pub mem: MMU,
    pub source: HashMap<String, String>,
    pub env: HashMap<String, u64>,
}

#[derive(Debug)]
pub enum ShimValue {
    None,
    Print,
    Integer(i32),
}

use std::mem::{size_of, transmute};
const _: () = {
    assert!(std::mem::size_of::<ShimValue>() <= 8);
};

impl ShimValue {
    fn call(&self, args: Vec<ShimValue>) -> Result<ShimValue, String> {
        match self {
            ShimValue::None => Err(format!("Can't call None as function with args {:?}", args)),
            ShimValue::Print => {
                for (idx, arg) in args.iter().enumerate() {
                    if idx != 0 {
                        print!(" ");
                    }
                    print!("{}", arg.to_string());
                }
                println!();
                Ok(ShimValue::None)
            },
            ShimValue::Integer(i) => Err(format!("Can't call int {:?} as function with args {:?}", i, args)),
        }
    }

    fn to_string(&self) -> String {
        match self {
            ShimValue::Integer(i) => i.to_string(),
            value => format!("{:?}", value),
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

impl Interpreter {
    pub fn create(config: &Config) -> Self {
        let mmu = MMU::with_capacity(Word(config.memory_space_bytes / 8));

        Self {
            mem: mmu,
            source: HashMap::new(),
            env: HashMap::new(),
        }
    }

    pub fn execute(&mut self, program: &Program) -> Result<(), String> {
        for stmt in program.stmts.iter() {
            self.execute_statement(stmt)?;
        }
        Ok(())
    }

    pub fn evaluate(&mut self, expr: &Expression) -> Result<ShimValue, String>  {
        match expr {
            Expression::Primary(p) => match p {
                Primary::Identifier(s) => if s == "print" {
                    Ok(ShimValue::Print)
                } else {
                    if let Some(value) = self.env.get(s) {
                        Ok(
                            unsafe { ShimValue::from_u64(*value) }
                        )
                    } else {
                        Err(format!("Unknown identifier {:?}", s))
                    }
                },
                Primary::Integer(i) => Ok(ShimValue::Integer(*i)),
                prim => Err(format!("Can't evaluate primary {:?}", prim)),
            },
            Expression::Call(expr, args) => {
                let obj = self.evaluate(expr)?;
                let args = args.iter().map(|a| self.evaluate(a)).collect::<Result<Vec<ShimValue>, String>>()?;
                obj.call(args)
            },
            Expression::BinaryOp(BinaryOp::Add(a, b)) => {
                let a = self.evaluate(a)?;
                let b = self.evaluate(b)?;
                a.add(&b)
            },

            expr => Err(format!("Can't evaluate {:?}", expr)),
        }
    }

    pub fn execute_statement(&mut self, stmts: &Statement) -> Result<ShimValue, String>  {
        match stmts {
            Statement::Let(ident, expr) => {
                let value = self.evaluate(expr)?;
                self.env.insert(ident.clone(), value.to_u64());
                Ok(ShimValue::None)
            },
            Statement::Expression(expr) => {
                self.evaluate(expr)
            }
        }
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
