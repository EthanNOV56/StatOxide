// In crates/so-core/src/formula/expander.rs
use super::{Formula, Interaction, Term, TermKind};
use crate::formula::error::FormulaResult;

/// Expands parenthesized expressions in formulas
pub struct FormulaExpander;

impl FormulaExpander {
    /// Fully expand a formula
    pub fn expand(formula: &Formula) -> FormulaResult<Formula> {
        let mut expanded = formula.clone();
        Self::expand_formula(&mut expanded)?;
        Ok(expanded)
    }

    /// Expand a formula in place
    fn expand_formula(formula: &mut Formula) -> FormulaResult<()> {
        // First, expand all parenthesized terms
        let mut new_terms = Vec::new();

        for term in &formula.terms {
            Self::expand_term(term, &mut new_terms)?;
        }

        formula.terms = new_terms;

        // Then apply distributive law for interactions
        Self::apply_distributive_law(formula)?;

        // Remove duplicate terms
        formula.terms.dedup();

        Ok(())
    }

    /// Expand a single term
    fn expand_term(term: &Term, result: &mut Vec<Term>) -> FormulaResult<()> {
        match &term.kind {
            TermKind::Parenthesized(terms) => {
                // Expand each term in the parentheses
                for t in terms {
                    result.push(t.clone());
                }
            }
            TermKind::Interaction(interaction) => {
                // For interactions, we need to check if any factor is parenthesized
                // This is a simplified version
                result.push(term.clone());
            }
            _ => {
                result.push(term.clone());
            }
        }

        Ok(())
    }

    /// Apply distributive law: (a + b):c -> a:c + b:c
    fn apply_distributive_law(formula: &mut Formula) -> FormulaResult<()> {
        let mut changed = true;

        while changed {
            changed = false;
            let mut new_terms = Vec::new();

            for term in &formula.terms {
                if let TermKind::Interaction(interaction) = &term.kind {
                    // Check if we need to distribute
                    if let Some(distributed) = Self::distribute_interaction(interaction)? {
                        new_terms.extend(distributed);
                        changed = true;
                    } else {
                        new_terms.push(term.clone());
                    }
                } else {
                    new_terms.push(term.clone());
                }
            }

            formula.terms = new_terms;
        }

        Ok(())
    }

    /// Distribute an interaction if it contains parenthesized terms
    fn distribute_interaction(interaction: &Interaction) -> FormulaResult<Option<Vec<Term>>> {
        // In a complete implementation, we would:
        // 1. Check if any variable in the interaction corresponds to a parenthesized term
        // 2. If so, distribute the interaction over the terms in the parentheses

        // For now, return None to indicate no distribution was needed
        Ok(None)
    }
}
