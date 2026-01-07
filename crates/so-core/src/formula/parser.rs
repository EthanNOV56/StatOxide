//! Formula parser for R-style formulas
//!
//! This parser implements R-style formula syntax with support for:
//! - Response variables: y ~ x1 + x2
//! - Intercept control: y ~ 0 + x1, y ~ 1
//! - Interactions: x1:x2, x1:x2:x3
//! - Parentheses: (x1 + x2):x3
//! - Function calls: log(x), sqrt(x), poly(x, 2)
//! - Automatic expansion of parentheses: (a+b):c -> a:c + b:c

use crate::formula::error::{FormulaError, FormulaResult};
use crate::formula::{Formula, Term, TermKind};
use std::iter::Peekable;
use std::str::Chars;

/// Formula parser
pub struct FormulaParser<'a> {
    chars: Peekable<Chars<'a>>,
    original: String,
    position: usize,
}

impl<'a> FormulaParser<'a> {
    /// Create a new parser
    pub fn new(input: &'a str) -> Self {
        Self {
            chars: input.chars().peekable(),
            original: input.to_string(),
            position: 0,
        }
    }

    /// Parse a formula
    pub fn parse(formula: &str) -> FormulaResult<Formula> {
        let mut parser = FormulaParser::new(formula);
        parser.parse_formula()
    }

    /// Parse the entire formula
    fn parse_formula(&mut self) -> FormulaResult<Formula> {
        // Skip initial whitespace
        self.skip_whitespace();

        // Check for empty formula
        if self.chars.peek().is_none() {
            return Err(FormulaError::syntax(self.position, "Empty formula"));
        }

        // Parse response (left side of ~)
        let response = self.parse_response()?;

        // Parse tilde
        self.parse_tilde()?;

        // Parse right-hand side
        let (has_intercept, terms) = self.parse_rhs()?;

        // Check for trailing characters
        self.skip_whitespace();
        if self.chars.peek().is_some() {
            let remaining: String = self.chars.clone().collect();
            return Err(FormulaError::syntax_with_context(
                self.position,
                "Trailing characters after formula",
                format!("Unexpected: '{}'", remaining),
            ));
        }

        Ok(Formula {
            response,
            terms,
            has_intercept,
            original: self.original.clone(),
        })
    }

    /// Parse response variable (left side of ~)
    fn parse_response(&mut self) -> FormulaResult<Option<String>> {
        self.skip_whitespace();

        // Check if formula starts with ~ (no response)
        if self.peek_char() == Some('~') {
            return Ok(None);
        }

        // Parse identifier
        let ident = self.parse_identifier()?;

        // Check for ~
        self.skip_whitespace();
        if self.peek_char() == Some('~') {
            Ok(Some(ident))
        } else {
            // If no ~ after identifier, it's not a valid formula
            Err(FormulaError::syntax_with_context(
                self.position,
                "Expected '~' after response variable",
                format!("Found '{}' instead", self.peek_char().unwrap_or(' ')),
            ))
        }
    }

    /// Parse right-hand side of formula
    fn parse_rhs(&mut self) -> FormulaResult<(bool, Vec<Term>)> {
        self.skip_whitespace();

        // Handle empty RHS (just intercept)
        if self.chars.peek().is_none() {
            return Ok((true, Vec::new()));
        }

        // Check for intercept specification (0 or 1)
        let mut has_intercept = true;
        let mut terms = Vec::new();

        // Check if RHS starts with 0 or 1
        if let Some(&c) = self.chars.peek() {
            if c == '0' || c == '1' {
                // Consume the intercept specifier
                self.chars.next();
                self.position += 1;
                has_intercept = c == '1';

                self.skip_whitespace();

                // Check if there are terms after intercept specifier
                if self.chars.peek().is_none() {
                    return Ok((has_intercept, terms));
                }

                // Must have '+' after 0/1
                if self.peek_char() != Some('+') {
                    return Err(FormulaError::syntax_with_context(
                        self.position,
                        "Expected '+' after intercept specification",
                        format!("Found '{}' instead", self.peek_char().unwrap_or(' ')),
                    ));
                }

                // Consume the '+'
                self.chars.next();
                self.position += 1;
            }
        }

        // Parse terms separated by '+'
        loop {
            self.skip_whitespace();

            // Check for end of input
            if self.chars.peek().is_none() {
                break;
            }

            // Check if we're at a '+' without a preceding term
            if self.peek_char() == Some('+') {
                return Err(FormulaError::syntax(
                    self.position,
                    "Expected term before '+'",
                ));
            }

            // Parse a term
            let term = self.parse_term()?;
            terms.push(term);

            self.skip_whitespace();

            // Check for more terms
            if self.peek_char() == Some('+') {
                // Consume the '+'
                self.chars.next();
                self.position += 1;

                // After consuming '+', we must have another term
                self.skip_whitespace();
                if self.chars.peek().is_none() {
                    return Err(FormulaError::syntax(
                        self.position,
                        "Expected term after '+'",
                    ));
                }

                continue;
            } else {
                // No more terms
                break;
            }
        }

        Ok((has_intercept, terms))
    }

    /// Parse a term (can be a product of factors separated by ':')
    fn parse_term(&mut self) -> FormulaResult<Term> {
        // Parse first factor
        let first_factor = self.parse_factor()?;

        // Check for colon (interaction)
        self.skip_whitespace();

        if self.peek_char() == Some(':') {
            // It's an interaction, collect all factors
            let mut factors = vec![first_factor];

            loop {
                self.skip_whitespace();

                if self.peek_char() == Some(':') {
                    self.chars.next(); // Skip ':'
                    self.position += 1;

                    self.skip_whitespace();

                    // Parse next factor
                    let factor = self.parse_factor()?;
                    factors.push(factor);
                } else {
                    break;
                }
            }

            // Create interaction term
            if factors.len() < 2 {
                return Err(FormulaError::syntax(
                    self.position,
                    "Interaction requires at least two factors",
                ));
            }

            // Convert factors to variable names
            let mut variable_names = Vec::new();
            for factor in factors {
                if let TermKind::Variable(name) = factor.kind {
                    variable_names.push(name);
                } else {
                    return Err(FormulaError::syntax(
                        self.position,
                        "Interaction terms must be simple variables",
                    ));
                }
            }

            Ok(Term::interaction(variable_names))
        } else {
            // Single factor, not an interaction
            Ok(first_factor)
        }
    }

    /// Parse a factor (variable, function call, or parenthesized expression)
    fn parse_factor(&mut self) -> FormulaResult<Term> {
        self.skip_whitespace();

        match self.peek_char() {
            Some('(') => {
                // Parenthesized expression
                self.parse_parenthesized_expression()
            }
            Some(c) if c.is_alphabetic() => {
                // Could be a variable or function call
                self.parse_identifier_or_function()
            }
            Some(c) if c.is_digit(10) => {
                // Numeric literal (only allowed in function arguments)
                self.parse_numeric_literal()
            }
            Some(c) => Err(FormulaError::syntax(
                self.position,
                format!("Unexpected character '{}' in factor", c),
            )),
            None => Err(FormulaError::syntax(
                self.position,
                "Unexpected end of input, expected factor",
            )),
        }
    }

    /// Parse a parenthesized expression
    fn parse_parenthesized_expression(&mut self) -> FormulaResult<Term> {
        // We know the next char is '('
        self.chars.next(); // Skip '('
        self.position += 1;

        // Parse the expression inside parentheses
        let (_, terms) = self.parse_rhs()?;

        self.skip_whitespace();

        // Expect closing ')'
        match self.chars.next() {
            Some(')') => {
                self.position += 1;

                // If there's only one term inside parentheses, return it directly
                if terms.len() == 1 {
                    Ok(terms.into_iter().next().unwrap())
                } else {
                    // Multiple terms inside parentheses - we'll handle the expansion
                    // by creating a special "parenthesized" term
                    Ok(Term {
                        kind: TermKind::Parenthesized(terms),
                        term_type: crate::formula::term::TermType::Main,
                        expanded: false,
                    })
                }
            }
            Some(c) => Err(FormulaError::syntax(
                self.position - 1,
                format!("Expected ')', found '{}'", c),
            )),
            None => Err(FormulaError::syntax(
                self.position,
                "Unexpected end of input, expected ')'",
            )),
        }
    }

    /// Parse an identifier or function call
    fn parse_identifier_or_function(&mut self) -> FormulaResult<Term> {
        let ident = self.parse_identifier()?;

        self.skip_whitespace();

        // Check if it's a function call
        if self.peek_char() == Some('(') {
            self.parse_function_call(&ident)
        } else {
            // It's just a variable
            Ok(Term::variable(&ident))
        }
    }

    /// Parse a function call
    fn parse_function_call(&mut self, func_name: &str) -> FormulaResult<Term> {
        // We know the next char is '('
        self.chars.next(); // Skip '('
        self.position += 1;

        let mut args = Vec::new();

        loop {
            self.skip_whitespace();

            // Check for closing parenthesis
            if self.peek_char() == Some(')') {
                // Function call must have at least one argument
                if args.is_empty() {
                    return Err(FormulaError::syntax(
                        self.position,
                        format!("Function '{}' requires at least one argument", func_name),
                    ));
                }

                self.chars.next(); // Skip ')'
                self.position += 1;
                break;
            }

            // Parse argument
            let arg = self.parse_factor()?;
            args.push(arg);

            self.skip_whitespace();

            // Check for comma or closing parenthesis
            match self.peek_char() {
                Some(',') => {
                    self.chars.next(); // Skip comma
                    self.position += 1;
                    continue;
                }
                Some(')') => {
                    self.chars.next(); // Skip ')'
                    self.position += 1;
                    break;
                }
                Some(c) => {
                    return Err(FormulaError::syntax(
                        self.position,
                        format!("Expected ',' or ')', found '{}'", c),
                    ));
                }
                None => {
                    return Err(FormulaError::syntax(
                        self.position,
                        "Unexpected end of input, expected ')'",
                    ));
                }
            }
        }

        Ok(Term::function(func_name, args))
    }

    /// Parse a numeric literal
    fn parse_numeric_literal(&mut self) -> FormulaResult<Term> {
        let mut literal = String::new();
        let start_pos = self.position;

        // Parse integer part
        while let Some(&c) = self.chars.peek() {
            if c.is_digit(10) {
                literal.push(c);
                self.chars.next();
                self.position += 1;
            } else {
                break;
            }
        }

        if literal.is_empty() {
            return Err(FormulaError::syntax(start_pos, "Invalid numeric literal"));
        }

        // Numeric literal as a special "variable"
        Ok(Term::variable(&literal))
    }

    /// Parse an identifier
    fn parse_identifier(&mut self) -> FormulaResult<String> {
        let mut ident = String::new();
        let start_pos = self.position;

        // First character must be alphabetic
        match self.chars.next() {
            Some(c) if c.is_alphabetic() => {
                self.position += 1;
                ident.push(c);
            }
            Some(c) => {
                return Err(FormulaError::syntax(
                    start_pos,
                    format!("Identifier must start with a letter, found '{}'", c),
                ));
            }
            None => {
                return Err(FormulaError::syntax(
                    start_pos,
                    "Unexpected end of input, expected identifier",
                ));
            }
        }

        // Subsequent characters can be alphanumeric, underscore, or period
        while let Some(&c) = self.chars.peek() {
            if c.is_alphanumeric() || c == '_' || c == '.' {
                ident.push(c);
                self.chars.next();
                self.position += 1;
            } else {
                break;
            }
        }

        Ok(ident)
    }

    /// Parse tilde operator
    fn parse_tilde(&mut self) -> FormulaResult<()> {
        self.skip_whitespace();

        match self.chars.next() {
            Some('~') => {
                self.position += 1;
                Ok(())
            }
            Some(c) => Err(FormulaError::syntax(
                self.position - 1,
                format!("Expected '~', found '{}'", c),
            )),
            None => Err(FormulaError::syntax(
                self.position,
                "Unexpected end of formula, expected '~'",
            )),
        }
    }

    /// Skip whitespace
    fn skip_whitespace(&mut self) {
        while let Some(&c) = self.chars.peek() {
            if c.is_whitespace() {
                self.chars.next();
                self.position += 1;
            } else {
                break;
            }
        }
    }

    /// Peek at next character
    fn peek_char(&mut self) -> Option<char> {
        self.chars.peek().copied()
    }
}
