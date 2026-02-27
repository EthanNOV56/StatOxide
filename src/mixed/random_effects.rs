//! Random effects specification for mixed models

use std::collections::HashMap;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Type of random effect
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RandomEffectType {
    /// Random intercept only: (1 | group)
    Intercept,
    /// Random slope only: (0 + x | group)  
    Slope,
    /// Random intercept and slope (correlated): (1 + x | group)
    InterceptSlope,
    /// Random intercept and slope (uncorrelated): (1 || group) or (x || group)
    Uncorrelated,
}

/// Specification for a single random effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomEffect {
    /// Type of random effect
    pub effect_type: RandomEffectType,
    /// Grouping variable name
    pub grouping: String,
    /// Predictor variable for slope (if applicable)
    pub predictor: Option<String>,
    /// Nested grouping variable (optional)
    pub nested_within: Option<String>,
    /// Crossed with another grouping variable (optional)
    pub crossed_with: Option<String>,
    /// Custom covariance structure (overrides default)
    pub custom_covariance: Option<String>,
}

impl RandomEffect {
    /// Create a random intercept effect
    pub fn intercept(grouping: &str) -> Self {
        Self {
            effect_type: RandomEffectType::Intercept,
            grouping: grouping.to_string(),
            predictor: None,
            nested_within: None,
            crossed_with: None,
            custom_covariance: None,
        }
    }
    
    /// Create a random slope effect
    pub fn slope(grouping: &str, predictor: &str) -> Self {
        Self {
            effect_type: RandomEffectType::Slope,
            grouping: grouping.to_string(),
            predictor: Some(predictor.to_string()),
            nested_within: None,
            crossed_with: None,
            custom_covariance: None,
        }
    }
    
    /// Create correlated random intercept and slope
    pub fn intercept_slope(grouping: &str, predictor: &str) -> Self {
        Self {
            effect_type: RandomEffectType::InterceptSlope,
            grouping: grouping.to_string(),
            predictor: Some(predictor.to_string()),
            nested_within: None,
            crossed_with: None,
            custom_covariance: None,
        }
    }
    
    /// Create uncorrelated random intercept and slope
    pub fn uncorrelated_intercept_slope(grouping: &str, predictor: &str) -> Self {
        Self {
            effect_type: RandomEffectType::Uncorrelated,
            grouping: grouping.to_string(),
            predictor: Some(predictor.to_string()),
            nested_within: None,
            crossed_with: None,
            custom_covariance: None,
        }
    }
    
    /// Nest this random effect within another grouping
    pub fn nested_within(mut self, parent_grouping: &str) -> Self {
        self.nested_within = Some(parent_grouping.to_string());
        self
    }
    
    /// Cross this random effect with another grouping
    pub fn crossed_with(mut self, other_grouping: &str) -> Self {
        self.crossed_with = Some(other_grouping.to_string());
        self
    }
    
    /// Set custom covariance structure
    pub fn with_covariance(mut self, covariance: &str) -> Self {
        self.custom_covariance = Some(covariance.to_string());
        self
    }
    
    /// Get design matrix for this random effect
    pub fn design_matrix(&self, _data: &HashMap<String, Array1<f64>>, _group_ids: &Array1<usize>) 
        -> Result<Array2<f64>, String> {
        // TODO: Implement design matrix construction based on effect type
        // This would create Z matrix for this random effect
        
        Err("Design matrix construction not yet implemented".to_string())
    }
    
    /// Get number of random effect parameters
    pub fn n_parameters(&self, n_groups: usize) -> usize {
        match self.effect_type {
            RandomEffectType::Intercept => n_groups,
            RandomEffectType::Slope => n_groups,
            RandomEffectType::InterceptSlope => n_groups * 2,
            RandomEffectType::Uncorrelated => n_groups * 2,
        }
    }
    
    /// Parse random effect from formula syntax (e.g., "(1 | group)", "(x | group)")
    pub fn from_formula(formula: &str) -> Result<Self, String> {
        // Simple parsing for common patterns
        let formula = formula.trim();
        
        if !formula.starts_with('(') || !formula.ends_with(')') {
            return Err("Random effect formula must be in parentheses".to_string());
        }
        
        let inner = &formula[1..formula.len()-1];
        let parts: Vec<&str> = inner.split('|').map(|s| s.trim()).collect();
        
        if parts.len() != 2 {
            return Err("Random effect formula must contain '|'".to_string());
        }
        
        let effect_part = parts[0];
        let grouping = parts[1];
        
        // Parse effect part
        if effect_part == "1" {
            Ok(Self::intercept(grouping))
        } else if effect_part.starts_with("0+") {
            let predictor = &effect_part[2..];
            Ok(Self::slope(grouping, predictor))
        } else if effect_part.contains('+') {
            // Handle things like "1 + x"
            let predictors: Vec<&str> = effect_part.split('+').map(|s| s.trim()).collect();
            if predictors.len() == 2 && (predictors[0] == "1" || predictors[1] == "1") {
                let predictor = if predictors[0] == "1" { predictors[1] } else { predictors[0] };
                Ok(Self::intercept_slope(grouping, predictor))
            } else {
                Err("Complex random effect formulas not yet supported".to_string())
            }
        } else if effect_part.contains("||") {
            // Uncorrelated random effects
            let effect_part = effect_part.replace("||", "|");
            let subparts: Vec<&str> = effect_part.split('|').map(|s| s.trim()).collect();
            if subparts.len() == 2 && subparts[0] == "1" {
                Ok(Self::uncorrelated_intercept_slope(grouping, subparts[1]))
            } else {
                Err("Uncorrelated random effect parsing not yet implemented".to_string())
            }
        } else {
            // Assume it's a random slope
            Ok(Self::slope(grouping, effect_part))
        }
    }
    
    /// Convert to formula string representation
    pub fn to_formula(&self) -> String {
        match self.effect_type {
            RandomEffectType::Intercept => {
                format!("(1 | {})", self.grouping)
            }
            RandomEffectType::Slope => {
                if let Some(ref predictor) = self.predictor {
                    format!("(0 + {} | {})", predictor, self.grouping)
                } else {
                    format!("(0 | {})", self.grouping) // Should not happen
                }
            }
            RandomEffectType::InterceptSlope => {
                if let Some(ref predictor) = self.predictor {
                    format!("(1 + {} | {})", predictor, self.grouping)
                } else {
                    format!("(1 | {})", self.grouping) // Should not happen
                }
            }
            RandomEffectType::Uncorrelated => {
                if let Some(ref predictor) = self.predictor {
                    format!("(1 || {}) + ({} || {})", self.grouping, predictor, self.grouping)
                } else {
                    format!("(1 || {})", self.grouping) // Should not happen
                }
            }
        }
    }
}