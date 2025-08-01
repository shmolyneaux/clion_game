use alloc::string::{String, ToString};
use alloc::vec::Vec;

use core::ops::ControlFlow;
use core::str;

/// Error encountered during tokenization
#[derive(Debug, Clone, PartialEq)]
pub struct TokenError {
    /// The specific type of error that occurred during tokenization
    pub kind: TokenErrorKind,
    /// The location in the source where the error occurred
    pub span: Span,
}

/// Types of errors that can occur during tokenization
#[derive(Debug, Clone, PartialEq)]
pub enum TokenErrorKind {
    /// Unexpected character encountered
    UnexpectedCharacter(char),
    /// End of file reached unexpectedly
    UnexpectedEof(&'static str),
    /// Invalid UTF-8 sequence
    InvalidUtf8(String),
    /// Number is out of range
    NumberOutOfRange(f64),
}

use alloc::borrow::Cow;
use core::fmt::{self, Display, Formatter};

use facet_deserialize::{Pos, Span, Spanned};

impl Display for TokenErrorKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            TokenErrorKind::UnexpectedCharacter(c) => write!(f, "unexpected character: '{}'", c),
            TokenErrorKind::UnexpectedEof(context) => write!(f, "unexpected EOF {}", context),
            TokenErrorKind::InvalidUtf8(detail) => write!(f, "invalid UTF-8: {}", detail),
            TokenErrorKind::NumberOutOfRange(n) => write!(f, "number out of range: {}", n),
        }
    }
}

/// Tokenization result, yielding a spanned token
pub type TokenizeResult<'input> = Result<Spanned<Token<'input>>, TokenError>;

/// JSON tokens (without positions)
#[derive(Debug, Clone, PartialEq)]
pub enum Token<'input> {
    /// Left brace character: '{'
    LBrace,
    /// Right brace character: '}'
    RBrace,
    /// Left bracket character: '['
    LBracket,
    /// Right bracket character: ']'
    RBracket,
    /// Colon character: ':'
    Colon,
    /// Comma character: ','
    Comma,
    /// A JSON string value
    /// TODO: should be a &[u8], lazily de-escaped if/when needed
    String(Cow<'input, str>),
    /// A 64-bit floating point number value — used if the value contains a decimal point
    F64(f64),
    /// A signed 64-bit integer number value — used if the value does not contain a decimal point but contains a sign
    I64(i64),
    /// An unsigned 64-bit integer number value — used if the value does not contain a decimal point and does not contain a sign
    U64(u64),
    /// A 128-bit unsigned integer number value
    U128(u128),
    /// A 128-bit signed integer number value
    I128(i128),
    /// The JSON boolean value 'true'
    True,
    /// The JSON boolean value 'false'
    False,
    /// The JSON null value
    Null,
    /// End of file marker
    Eof,
}

impl Display for Token<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::Colon => write!(f, ":"),
            Token::Comma => write!(f, ","),
            Token::String(s) => write!(f, "\"{}\"", s),
            Token::F64(n) => write!(f, "{}", n),
            Token::I64(n) => write!(f, "{}", n),
            Token::U64(n) => write!(f, "{}", n),
            Token::U128(n) => write!(f, "{}", n),
            Token::I128(n) => write!(f, "{}", n),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::Null => write!(f, "null"),
            Token::Eof => write!(f, "EOF"),
        }
    }
}

/// Simple JSON tokenizer producing spanned tokens from byte input.
pub struct Tokenizer<'input> {
    input: &'input [u8],
    pos: Pos,
}

impl<'input> Tokenizer<'input> {
    /// Create a new tokenizer for the given input slice.
    pub fn new(input: &'input [u8]) -> Self {
        Tokenizer { input, pos: 0 }
    }

    /// Return the next spanned token or a TokenizeError
    pub fn next_token(&mut self) -> TokenizeResult<'input> {
        self.skip_whitespace();
        let start = self.pos;
        let c = match self.input.get(self.pos).copied() {
            Some(c) => c,
            None => {
                // EOF at this position
                let span = Span::new(self.pos, 0);
                return Ok(Spanned {
                    node: Token::Eof,
                    span,
                });
            }
        };
        let sp = match c {
            b'{' => {
                self.pos += 1;
                Spanned {
                    node: Token::LBrace,
                    span: Span::new(start, 1),
                }
            }
            b'}' => {
                self.pos += 1;
                Spanned {
                    node: Token::RBrace,
                    span: Span::new(start, 1),
                }
            }
            b'[' => {
                self.pos += 1;
                Spanned {
                    node: Token::LBracket,
                    span: Span::new(start, 1),
                }
            }
            b']' => {
                self.pos += 1;
                Spanned {
                    node: Token::RBracket,
                    span: Span::new(start, 1),
                }
            }
            b':' => {
                self.pos += 1;
                Spanned {
                    node: Token::Colon,
                    span: Span::new(start, 1),
                }
            }
            b',' => {
                self.pos += 1;
                Spanned {
                    node: Token::Comma,
                    span: Span::new(start, 1),
                }
            }
            b'"' => return self.parse_string(start),
            b'-' | b'0'..=b'9' => return self.parse_number(start),
            b't' => return self.parse_literal(start, b"true", || Token::True),
            b'f' => return self.parse_literal(start, b"false", || Token::False),
            b'n' => return self.parse_literal(start, b"null", || Token::Null),
            _ => {
                return Err(TokenError {
                    kind: TokenErrorKind::UnexpectedCharacter(c as char),
                    span: Span::new(start, 1),
                });
            }
        };
        Ok(sp)
    }

    /// Skip whitespace characters
    fn skip_whitespace(&mut self) {
        while let Some(&b) = self.input.get(self.pos) {
            match b {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    #[inline(never)]
    fn parse_string(&mut self, start: Pos) -> TokenizeResult<'input> {
        const STEP_SIZE: usize = Window::BITS as usize / 8;
        type Window = u128;
        type Chunk = [u8; STEP_SIZE];

        // Skip opening quote
        self.pos += 1;
        let content_start = self.pos;
        let mut buf = CowBuf::Borrowed {
            input: self.input,
            buf_start: content_start,
            buf_end: content_start,
        };

        let mut done = false;
        'outer: while let Some(Ok(chunk)) =
            self.input[self.pos..].get(..STEP_SIZE).map(Chunk::try_from)
        {
            let window = Window::from_ne_bytes(chunk);
            let quote_free = !super::contains_0x22(window);
            let backslash_free = !super::contains_0x5c(window);
            if quote_free && backslash_free {
                buf.push_borrowed(&chunk);
                self.pos += STEP_SIZE;
            } else {
                let chunk_start = self.pos;
                while let Some(&b) = chunk.get(self.pos - chunk_start) {
                    if self.parse_char(b, &mut buf)? == ControlFlow::Break(()) {
                        done = true;
                        break 'outer;
                    }
                }
            }
        }
        if !done {
            while let Some(&b) = self.input.get(self.pos) {
                if self.parse_char(b, &mut buf)? == ControlFlow::Break(()) {
                    break;
                }
            }
        }

        // Check if we reached the end without finding a closing quote
        if self.pos > self.input.len()
            || (self.pos == self.input.len() && self.input[self.pos - 1] != b'"')
        {
            return Err(TokenError {
                kind: TokenErrorKind::UnexpectedEof("in string literal"),
                span: Span::new(start, self.pos - start),
            });
        }

        match buf {
            CowBuf::Borrowed {
                input,
                buf_start,
                buf_end,
            } => match str::from_utf8(&input[buf_start..buf_end]) {
                Ok(st) => Ok(Spanned {
                    node: Token::String(Cow::Borrowed(st)),
                    span: Span::new(start, self.pos - start),
                }),
                Err(e) => Err(TokenError {
                    kind: TokenErrorKind::InvalidUtf8(e.to_string()),
                    span: Span::new(content_start, buf_end - buf_start),
                }),
            },
            CowBuf::Owned(buf) => {
                let len = buf.len();
                match String::from_utf8(buf) {
                    Ok(st) => Ok(Spanned {
                        node: Token::String(Cow::Owned(st)),
                        span: Span::new(start, self.pos - start),
                    }),
                    Err(e) => Err(TokenError {
                        kind: TokenErrorKind::InvalidUtf8(e.to_string()),
                        span: Span::new(content_start, len),
                    }),
                }
            }
        }
    }

    #[inline]
    fn parse_char(&mut self, byte: u8, buf: &mut CowBuf) -> Result<ControlFlow<()>, TokenError> {
        match byte {
            b'"' => {
                self.pos += 1;
                Ok(ControlFlow::Break(()))
            }
            b'\\' => {
                self.pos += 1;
                if let Some(&esc) = self.input.get(self.pos) {
                    match esc {
                        b'"' | b'\\' | b'/' => buf.push_owned(&[esc]),
                        b'b' => buf.push_owned(b"\x08"), // backspace
                        b'f' => buf.push_owned(b"\x0C"), // form feed
                        b'n' => buf.push_owned(b"\n"),   // line feed
                        b'r' => buf.push_owned(b"\r"),   // carriage return
                        b't' => buf.push_owned(b"\t"),   // tab
                        b'u' => {
                            // Handle \uXXXX Unicode escape sequence
                            // We need to read 4 hexadecimal digits
                            self.pos += 1; // Move past 'u'
                            let hex_start = self.pos;
                            if self.pos + 4 > self.input.len() {
                                return Err(TokenError {
                                    kind: TokenErrorKind::UnexpectedEof(
                                        "in Unicode escape sequence",
                                    ),
                                    span: Span::new(hex_start, self.input.len() - hex_start),
                                });
                            }

                            // Read 4 hexadecimal digits
                            let hex_digits = &self.input[self.pos..self.pos + 4];
                            let hex_str = match str::from_utf8(hex_digits) {
                                Ok(s) => s,
                                Err(_) => {
                                    return Err(TokenError {
                                        kind: TokenErrorKind::InvalidUtf8(
                                            "invalid UTF-8 in Unicode escape".to_string(),
                                        ),
                                        span: Span::new(hex_start, 4),
                                    });
                                }
                            };

                            // Parse hexadecimal value
                            let code_point = match u16::from_str_radix(hex_str, 16) {
                                Ok(cp) => cp,
                                Err(_) => {
                                    return Err(TokenError {
                                        kind: TokenErrorKind::UnexpectedCharacter('?'),
                                        span: Span::new(hex_start, 4),
                                    });
                                }
                            };

                            // Convert to UTF-8 and append to buffer
                            // Handle basic Unicode code points (BMP)
                            let c = match char::from_u32(code_point as u32) {
                                Some(c) => c,
                                None => {
                                    return Err(TokenError {
                                        kind: TokenErrorKind::InvalidUtf8(
                                            "invalid Unicode code point".to_string(),
                                        ),
                                        span: Span::new(hex_start, 4),
                                    });
                                }
                            };

                            // Extend buffer with UTF-8 bytes for the character
                            let mut utf8_buf = [0u8; 4];
                            let utf8_bytes = c.encode_utf8(&mut utf8_buf).as_bytes();
                            buf.push_owned(utf8_bytes);

                            self.pos += 3; // +3 because we'll increment once more below
                        }
                        _ => buf.push_owned(&[esc]), // other escapes
                    }
                    self.pos += 1;
                    Ok(ControlFlow::Continue(()))
                } else {
                    Err(TokenError {
                        kind: TokenErrorKind::UnexpectedEof("in string escape"),
                        span: Span::new(self.pos, 0),
                    })
                }
            }
            _ => {
                buf.push_borrowed(&[byte]);
                self.pos += 1;
                Ok(ControlFlow::Continue(()))
            }
        }
    }

    #[inline(never)]
    fn parse_number(&mut self, start: Pos) -> TokenizeResult<'input> {
        let mut end = self.pos;
        if self.input[end] == b'-' {
            end += 1;
        }
        while end < self.input.len() && self.input[end].is_ascii_digit() {
            end += 1;
        }
        if end < self.input.len() && self.input[end] == b'.' {
            end += 1;
            while end < self.input.len() && self.input[end].is_ascii_digit() {
                end += 1;
            }
        }
        if end < self.input.len() && (self.input[end] == b'e' || self.input[end] == b'E') {
            end += 1;
            if end < self.input.len() && (self.input[end] == b'+' || self.input[end] == b'-') {
                end += 1;
            }
            while end < self.input.len() && self.input[end].is_ascii_digit() {
                end += 1;
            }
        }
        let slice = &self.input[start..end];
        let span = Span::new(start, end - start);

        let text = match str::from_utf8(slice) {
            Ok(t) => t,
            Err(e) => {
                return Err(TokenError {
                    kind: TokenErrorKind::InvalidUtf8(e.to_string()),
                    span,
                });
            }
        };

        let token = if text.contains('.') || text.contains('e') || text.contains('E') {
            // If the number contains a decimal point or exponent, parse as f64
            match text.parse::<f64>() {
                Ok(n) => Token::F64(n),
                Err(_) => {
                    return Err(TokenError {
                        kind: TokenErrorKind::NumberOutOfRange(0.0),
                        span,
                    });
                }
            }
        } else if text.starts_with('-') {
            // If the number starts with a negative sign, parse as i64
            match text.parse::<i64>() {
                Ok(n) => Token::I64(n),
                Err(_) => {
                    // If i64 parsing fails, try to parse as i128
                    match text.parse::<i128>() {
                        Ok(n) => Token::I128(n),
                        Err(_) => {
                            // If i128 parsing fails, try to parse as f64 for error reporting
                            let num = text.parse::<f64>().unwrap_or(0.0);
                            return Err(TokenError {
                                kind: TokenErrorKind::NumberOutOfRange(num),
                                span,
                            });
                        }
                    }
                }
            }
        } else {
            // Otherwise, parse as u64
            match text.parse::<u64>() {
                Ok(n) => Token::U64(n),
                Err(_) => {
                    // If u64 parsing fails, try to parse as u128
                    match text.parse::<u128>() {
                        Ok(n) => Token::U128(n),
                        Err(_) => {
                            // If u128 parsing fails, try to parse as f64 for error reporting
                            let num = text.parse::<f64>().unwrap_or(0.0);
                            return Err(TokenError {
                                kind: TokenErrorKind::NumberOutOfRange(num),
                                span,
                            });
                        }
                    }
                }
            }
        };

        self.pos = end;
        Ok(Spanned { node: token, span })
    }

    #[inline(never)]
    fn parse_literal<F>(&mut self, start: Pos, pat: &[u8], ctor: F) -> TokenizeResult<'input>
    where
        F: FnOnce() -> Token<'static>,
    {
        let end = start + pat.len();
        if end <= self.input.len() && &self.input[start..end] == pat {
            self.pos = end;
            let span = Span::new(start, pat.len());
            Ok(Spanned { node: ctor(), span })
        } else {
            // Determine how much of the pattern matched before mismatch
            let actual_len = self.input.len().saturating_sub(start).min(pat.len());
            let span = Span::new(start, actual_len.max(1)); // Ensure span covers at least one character

            let got = self.input.get(start).copied().unwrap_or(b'?') as char;
            Err(TokenError {
                kind: TokenErrorKind::UnexpectedCharacter(got),
                span,
            })
        }
    }
}

enum CowBuf<'a> {
    Borrowed {
        input: &'a [u8],
        buf_start: usize,
        buf_end: usize,
    },
    Owned(Vec<u8>),
}

impl CowBuf<'_> {
    fn push_borrowed(&mut self, data: &[u8]) {
        match self {
            CowBuf::Borrowed { buf_end, .. } => *buf_end += data.len(),
            CowBuf::Owned(owned) => owned.extend(data),
        }
    }

    fn push_owned(&mut self, data: &[u8]) {
        match self {
            CowBuf::Borrowed {
                input,
                buf_start,
                buf_end,
            } => {
                // Pour one out for our speed gains...
                let mut owned = Vec::with_capacity(*buf_end - *buf_start + data.len());
                owned.extend(&input[*buf_start..*buf_end]);
                owned.extend(data);
                *self = CowBuf::Owned(owned);
            }
            CowBuf::Owned(owned) => owned.extend(data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() {
        let mut tokenizer = Tokenizer::new(r#"{"foo":"abc","bar":42,"baz":true}"#.as_bytes());
        assert_eq!(
            tokenizer.next_token(),
            Ok(Spanned {
                node: Token::LBrace,
                span: Span::new(0, 1)
            })
        );
        assert_eq!(
            tokenizer.next_token(),
            Ok(Spanned {
                node: Token::String(Cow::Borrowed("foo")),
                span: Span::new(1, 5)
            })
        );
        assert_eq!(
            tokenizer.next_token(),
            Ok(Spanned {
                node: Token::Colon,
                span: Span::new(6, 1)
            })
        );
        assert_eq!(
            tokenizer.next_token(),
            Ok(Spanned {
                node: Token::String(Cow::Borrowed("abc")),
                span: Span::new(7, 5)
            })
        );
        assert_eq!(
            tokenizer.next_token(),
            Ok(Spanned {
                node: Token::Comma,
                span: Span::new(12, 1)
            })
        );
        assert_eq!(
            tokenizer.next_token(),
            Ok(Spanned {
                node: Token::String(Cow::Borrowed("bar")),
                span: Span::new(13, 5)
            })
        );
        assert_eq!(
            tokenizer.next_token(),
            Ok(Spanned {
                node: Token::Colon,
                span: Span::new(18, 1)
            })
        );
        assert_eq!(
            tokenizer.next_token(),
            Ok(Spanned {
                node: Token::U64(42),
                span: Span::new(19, 2)
            })
        );
        assert_eq!(
            tokenizer.next_token(),
            Ok(Spanned {
                node: Token::Comma,
                span: Span::new(21, 1)
            })
        );
        assert_eq!(
            tokenizer.next_token(),
            Ok(Spanned {
                node: Token::String(Cow::Borrowed("baz")),
                span: Span::new(22, 5)
            })
        );
        assert_eq!(
            tokenizer.next_token(),
            Ok(Spanned {
                node: Token::Colon,
                span: Span::new(27, 1)
            })
        );
        assert_eq!(
            tokenizer.next_token(),
            Ok(Spanned {
                node: Token::True,
                span: Span::new(28, 4)
            })
        );
        assert_eq!(
            tokenizer.next_token(),
            Ok(Spanned {
                node: Token::RBrace,
                span: Span::new(32, 1)
            })
        );
    }
}
