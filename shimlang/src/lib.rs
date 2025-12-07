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
pub struct Program {
    expressions: Vec<Expression>
}

#[derive(Debug, PartialEq)]
pub enum Token {
    Comma,
    LBracket,
    RBracket,
    Plus,
    Minus,
    Integer(i32),
    Identifier(String),
}

pub fn parse_primary(tokens: &mut &[Token]) -> Result<Expression, String> {
    match &tokens[0] {
        Token::Integer(i) => {
            *tokens = &tokens[1..];
            Ok(Expression::Primary(Primary::Integer(*i)))
        },
        Token::Identifier(s) => {
            *tokens = &tokens[1..];
            Ok(Expression::Primary(Primary::Identifier(s.clone())))
        },
        Token::LBracket => {
            *tokens = &tokens[1..];
            let expr = parse_expression(tokens)?;
            if tokens[0] != Token::RBracket {
                Err(format!("Expected closing ')' but found {:?}", tokens[0]))
            } else {
                Ok(expr)
            }
        },
        token => Err(format!("Could not parse_primary {:?}", token)),
    }
}

pub fn parse_arguments(tokens: &mut &[Token]) -> Result<Vec<Expression>, String> {
    let mut args = Vec::new();
    while !tokens.is_empty() {
        let expr = parse_expression(tokens)?;
        args.push(expr);

        match &tokens[0] {
            Token::RBracket => {
                // Don't consume the closing bracket since we parse
                // function arguments and list literals the same way
                // and the parent needs to match the bracket
                break;
            }
            Token::Comma => {
                *tokens = &tokens[1..];
                if !tokens.is_empty() && tokens[0] == Token::RBracket {
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

pub fn parse_call(tokens: &mut &[Token]) -> Result<Expression, String> {
    let mut expr = parse_primary(tokens)?;
    while !tokens.is_empty() {
        match &tokens[0] {
            Token::LBracket => {
                *tokens = &tokens[1..];
                expr = Expression::Call(Box::new(expr), parse_arguments(tokens)?);
                if tokens.is_empty() {
                    return Err(format!("No more tokens, expected closing ')' for parse_call"));
                }
                if tokens[0] != Token::RBracket {
                    return Err(format!("Expected closing ')' for parse_call, got {:?}", tokens[0]));
                }
                *tokens = &tokens[1..];
            },
            _ => return Ok(expr),
        }
    }

    Ok(expr)
}

pub fn parse_term(tokens: &mut &[Token]) -> Result<Expression, String> {
    let mut expr = parse_primary(tokens)?;
    while !tokens.is_empty() {
        match &tokens[0] {
            Token::Plus => {
                *tokens = &tokens[1..];
                expr = Expression::BinaryOp(BinaryOp::Add(Box::new(expr), Box::new(parse_expression(tokens)?)));
            },
            Token::Minus => {
                *tokens = &tokens[1..];
                expr = Expression::BinaryOp(BinaryOp::Subtract(Box::new(expr), Box::new(parse_expression(tokens)?)));
            },
            _ => return Ok(expr),
        }
    }
    Ok(expr)
}

pub fn parse_expression(tokens: &mut &[Token]) -> Result<Expression, String> {
    parse_call(tokens)
}

pub fn parse_program(tokens: &mut &[Token]) -> Result<Program, String> {
    let mut exprs = Vec::new();
    let mut count = 0;
    while !tokens.is_empty() {
        let expr = parse_expression(tokens)?;
        exprs.push(expr);
        count += 1;
    }
    Ok(Program { expressions: exprs })
}

pub fn printable_byte(b: u8) -> String {
    match char::from_u32(b as u32) {
        Some(c) if !c.is_control() => c.to_string(),
        _ => format!("\\x{:02X}", b),
    }
}

pub fn lex_identifier(text: &mut &[u8]) -> Result<Token, String> {
    for (idx, c) in text.iter().enumerate() {
        match c {
            b'a' ..= b'z' | b'A' ..= b'Z' | b'0' ..= b'9' => continue,
            _ => {
                let token = Token::Identifier(unsafe { String::from_utf8_unchecked(text[0..idx].to_vec())});
                *text = &text[(idx-1)..];
                return Ok(token);
            }
        }
    }
    let token = Token::Identifier(
        unsafe {
            String::from_utf8_unchecked(text.to_vec())
        }
    );
    Ok(token)
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
            b'a' ..= b'z' => tokens.push(lex_identifier(&mut text)?),
            b'0' ..= b'9' => tokens.push(lex_number(&mut text)?),
            b'(' => tokens.push(Token::LBracket),
            b')' => tokens.push(Token::RBracket),
            _ => return Err(format!("Unknown character '{}'", printable_byte(c)))
        }
        text = &text[1..];
    }
    Ok(tokens)
}

pub fn ast_from_text(text: &[u8]) -> Result<Program, String> {
    let mut tokens = lex(text)?;
    let mut foo = tokens.as_slice();
    parse_program(&mut foo)
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

    fn free(&mut self, words: u32, ptr: *const u64) {
    }
}

#[derive(Facet)]
pub struct Interpreter {
    pub mem: MMU,
    pub source: HashMap<String, String>,
}

#[derive(Debug)]
enum ShimValue {
    None,
    Print,
    Integer(i32),
}

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
}

impl Interpreter {
    pub fn create(config: &Config) -> Self {
        let mmu = MMU::with_capacity(Word(config.memory_space_bytes / 8));

        Self {
            mem: mmu,
            source: HashMap::new(),
        }
    }

    pub fn execute(&self, program: &Program) -> Result<(), String> {
        for expr in program.expressions.iter() {
            self.evaluate(expr);
            match expr {
                expr => return Err(format!("Can't evaluate {:?}", expr)),
            }
        }
        Ok(())
    }

    pub fn evaluate(&self, expr: &Expression) -> Result<ShimValue, String>  {
        match expr {
            Expression::Primary(p) => match p {
                Primary::Identifier(s) => if s == "print" {
                    Ok(ShimValue::Print)
                } else {
                    Err(format!("Unknown identifier {:?}", s))
                },
                Primary::Integer(i) => Ok(ShimValue::Integer(*i)),
                prim => Err(format!("Can't evaluate primary {:?}", prim)),
            },
            Expression::Call(expr, args) => {
                let obj = self.evaluate(expr)?;
                let args = args.iter().map(|a| self.evaluate(a)).collect::<Result<Vec<ShimValue>, String>>()?;
                obj.call(args)
            },
            expr => Err(format!("Can't evaluate {:?}", expr)),
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
