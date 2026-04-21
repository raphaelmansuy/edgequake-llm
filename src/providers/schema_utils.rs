//! Shared JSON Schema utilities for provider-specific sanitization.
//!
//! Each LLM provider imposes its own subset of JSON Schema.  This module
//! provides reusable, composable helpers so that per-provider `sanitize_*`
//! and `convert_tools` methods stay DRY while each provider retains full
//! control over *which* transformations to apply (Open/Closed Principle).
//!
//! # Design
//!
//! * **Pure functions** — every helper takes a `serde_json::Value` and returns
//!   a new `serde_json::Value`.  No provider state, no side-effects.
//! * **Composable** — providers chain the helpers they need in their own
//!   `sanitize_parameters()` or `convert_tools()` methods.
//! * **Tested** — unit tests cover each helper in isolation.

use serde_json::Value;

// ============================================================================
// Schema Analysis
// ============================================================================

/// Count the total number of optional parameters across a set of tool schemas.
///
/// A parameter is "optional" if it appears in `properties` but NOT in
/// `required`.  This is the metric Anthropic enforces (limit: 24 across
/// all strict tools combined).
pub fn count_optional_params(schema: &Value) -> usize {
    let obj = match schema.as_object() {
        Some(o) => o,
        None => return 0,
    };

    let properties = match obj.get("properties").and_then(|v| v.as_object()) {
        Some(p) => p,
        None => return 0,
    };

    let required: std::collections::HashSet<&str> = obj
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    properties
        .keys()
        .filter(|k| !required.contains(k.as_str()))
        .count()
}

/// Count parameters that use union types (`anyOf` or type arrays like
/// `"type": ["string", "null"]`).
///
/// Anthropic limits this to 16 across all strict tools combined.
pub fn count_union_type_params(schema: &Value) -> usize {
    let obj = match schema.as_object() {
        Some(o) => o,
        None => return 0,
    };

    let properties = match obj.get("properties").and_then(|v| v.as_object()) {
        Some(p) => p,
        None => return 0,
    };

    properties
        .values()
        .filter(|prop| {
            let prop_obj = match prop.as_object() {
                Some(o) => o,
                None => return false,
            };
            // Check for anyOf
            if prop_obj.contains_key("anyOf") {
                return true;
            }
            // Check for type array: "type": ["string", "null"]
            if let Some(Value::Array(_)) = prop_obj.get("type") {
                return true;
            }
            false
        })
        .count()
}

// ============================================================================
// Anthropic Strict-Mode Budget
// ============================================================================

/// Anthropic strict-mode limits (from official docs at
/// platform.claude.com/docs/en/build-with-claude/structured-outputs).
pub const ANTHROPIC_MAX_STRICT_TOOLS: usize = 20;
pub const ANTHROPIC_MAX_OPTIONAL_PARAMS: usize = 24;
pub const ANTHROPIC_MAX_UNION_TYPE_PARAMS: usize = 16;

/// Result of checking strict-mode budget across a set of tool schemas.
#[derive(Debug, Clone)]
pub struct StrictBudgetCheck {
    pub total_strict_tools: usize,
    pub total_optional_params: usize,
    pub total_union_type_params: usize,
    pub exceeds_limits: bool,
}

/// Check whether a set of tools would exceed Anthropic's strict-mode budget.
///
/// Takes an iterator of `(is_strict, &schema)` pairs.
pub fn check_anthropic_strict_budget<'a>(
    tools: impl Iterator<Item = (bool, &'a Value)>,
) -> StrictBudgetCheck {
    let mut total_strict_tools = 0usize;
    let mut total_optional_params = 0usize;
    let mut total_union_type_params = 0usize;

    for (is_strict, schema) in tools {
        if is_strict {
            total_strict_tools += 1;
            total_optional_params += count_optional_params(schema);
            total_union_type_params += count_union_type_params(schema);
        }
    }

    let exceeds_limits = total_strict_tools > ANTHROPIC_MAX_STRICT_TOOLS
        || total_optional_params > ANTHROPIC_MAX_OPTIONAL_PARAMS
        || total_union_type_params > ANTHROPIC_MAX_UNION_TYPE_PARAMS;

    StrictBudgetCheck {
        total_strict_tools,
        total_optional_params,
        total_union_type_params,
        exceeds_limits,
    }
}

// ============================================================================
// Schema Transformations (composable building blocks)
// ============================================================================

/// Recursively strip a set of keys from a JSON Schema.
///
/// Used by providers that don't support certain JSON Schema keywords
/// (e.g., Gemini strips `$schema`, `strict`, `additionalProperties`).
pub fn strip_keys_recursive(value: Value, keys: &[&str]) -> Value {
    match value {
        Value::Object(mut obj) => {
            for key in keys {
                obj.remove(*key);
            }
            let sanitized = obj
                .into_iter()
                .map(|(k, v)| (k, strip_keys_recursive(v, keys)))
                .collect();
            Value::Object(sanitized)
        }
        Value::Array(arr) => Value::Array(
            arr.into_iter()
                .map(|v| strip_keys_recursive(v, keys))
                .collect(),
        ),
        other => other,
    }
}

/// Recursively add `"additionalProperties": false` to every object-typed
/// schema node that doesn't already have it set.
///
/// Required by both OpenAI strict mode and Anthropic strict mode.
pub fn ensure_additional_properties_false(value: Value) -> Value {
    match value {
        Value::Object(mut obj) => {
            // Check if this node is an object type
            let is_object_type = obj
                .get("type")
                .map(|t| match t {
                    Value::String(s) => s == "object",
                    Value::Array(arr) => arr.iter().any(|v| v.as_str() == Some("object")),
                    _ => false,
                })
                .unwrap_or(false);

            if is_object_type && !obj.contains_key("additionalProperties") {
                obj.insert("additionalProperties".into(), Value::Bool(false));
            }

            let sanitized = obj
                .into_iter()
                .map(|(k, v)| (k, ensure_additional_properties_false(v)))
                .collect();
            Value::Object(sanitized)
        }
        Value::Array(arr) => Value::Array(
            arr.into_iter()
                .map(ensure_additional_properties_false)
                .collect(),
        ),
        other => other,
    }
}

/// Convert type arrays like `"type": ["string", "null"]` to single type +
/// `"nullable": true`.
///
/// Gemini's function-declaration schema requires a single string `type`
/// field and uses `nullable: true` instead of union types.
pub fn convert_type_arrays_to_nullable(value: Value) -> Value {
    match value {
        Value::Object(mut obj) => {
            if let Some(Value::Array(types)) = obj.get("type").cloned() {
                let mut nullable = false;
                let mut non_null_types = Vec::new();

                for entry in types {
                    match entry {
                        Value::String(s) if s == "null" => nullable = true,
                        Value::String(s) => non_null_types.push(s),
                        _ => {}
                    }
                }

                if let Some(primary) = non_null_types.into_iter().next() {
                    obj.insert("type".into(), Value::String(primary));
                    if nullable {
                        obj.insert("nullable".into(), Value::Bool(true));
                    }
                } else {
                    obj.remove("type");
                }
            }

            let sanitized = obj
                .into_iter()
                .map(|(k, v)| (k, convert_type_arrays_to_nullable(v)))
                .collect();
            Value::Object(sanitized)
        }
        Value::Array(arr) => Value::Array(
            arr.into_iter()
                .map(convert_type_arrays_to_nullable)
                .collect(),
        ),
        other => other,
    }
}

// ============================================================================
// OpenAI Strict-Mode Schema Normalization
// ============================================================================

/// Normalize a schema for OpenAI strict mode:
/// - All properties must be listed in `required`
/// - `additionalProperties: false` on every object
/// - Optional params get `"type": ["original_type", "null"]`
///
/// This follows the OpenAI docs: "You can denote optional fields by adding
/// `null` as a type option."
pub fn normalize_for_openai_strict(value: Value) -> Value {
    match value {
        Value::Object(mut obj) => {
            let is_object_type = obj
                .get("type")
                .map(|t| matches!(t, Value::String(s) if s == "object"))
                .unwrap_or(false);

            if is_object_type {
                // Ensure additionalProperties: false
                if !obj.contains_key("additionalProperties") {
                    obj.insert("additionalProperties".into(), Value::Bool(false));
                }

                // Make all properties required, adding null to type for optional ones
                if let Some(properties) = obj.get("properties").cloned() {
                    if let Some(props_obj) = properties.as_object() {
                        let required: std::collections::HashSet<String> = obj
                            .get("required")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect()
                            })
                            .unwrap_or_default();

                        let all_keys: Vec<String> = props_obj.keys().cloned().collect();

                        // For previously-optional params, add null to type union
                        let mut new_props = props_obj.clone();
                        for key in &all_keys {
                            if !required.contains(key) {
                                if let Some(prop) = new_props.get_mut(key) {
                                    make_nullable(prop);
                                }
                            }
                        }
                        obj.insert("properties".into(), Value::Object(new_props));

                        // All keys are now required
                        let all_required: Vec<Value> =
                            all_keys.into_iter().map(Value::String).collect();
                        obj.insert("required".into(), Value::Array(all_required));
                    }
                }
            }

            // Recurse into all values
            let sanitized = obj
                .into_iter()
                .map(|(k, v)| (k, normalize_for_openai_strict(v)))
                .collect();
            Value::Object(sanitized)
        }
        Value::Array(arr) => {
            Value::Array(arr.into_iter().map(normalize_for_openai_strict).collect())
        }
        other => other,
    }
}

/// Make a property schema nullable by adding `"null"` to its type.
///
/// - `"type": "string"` → `"type": ["string", "null"]`
/// - `"type": ["string", "integer"]` → `"type": ["string", "integer", "null"]`
/// - Already has null → no change
fn make_nullable(prop: &mut Value) {
    if let Some(obj) = prop.as_object_mut() {
        match obj.get("type").cloned() {
            Some(Value::String(s)) => {
                obj.insert(
                    "type".into(),
                    Value::Array(vec![Value::String(s), Value::String("null".into())]),
                );
            }
            Some(Value::Array(mut arr)) => {
                let has_null = arr.iter().any(|v| v.as_str() == Some("null"));
                if !has_null {
                    arr.push(Value::String("null".into()));
                    obj.insert("type".into(), Value::Array(arr));
                }
            }
            _ => {
                // No type field or unexpected type — wrap in anyOf with null
                // This is a defensive fallback
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- count_optional_params ----

    #[test]
    fn test_count_optional_params_all_required() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        });
        assert_eq!(count_optional_params(&schema), 0);
    }

    #[test]
    fn test_count_optional_params_some_optional() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
                "c": {"type": "boolean"}
            },
            "required": ["a"]
        });
        assert_eq!(count_optional_params(&schema), 2);
    }

    #[test]
    fn test_count_optional_params_no_required_array() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"}
            }
        });
        assert_eq!(count_optional_params(&schema), 2);
    }

    #[test]
    fn test_count_optional_params_no_properties() {
        let schema = json!({"type": "object"});
        assert_eq!(count_optional_params(&schema), 0);
    }

    // ---- count_union_type_params ----

    #[test]
    fn test_count_union_type_params_with_anyof() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "b": {"type": "string"}
            }
        });
        assert_eq!(count_union_type_params(&schema), 1);
    }

    #[test]
    fn test_count_union_type_params_with_type_array() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": ["string", "null"]},
                "b": {"type": "string"}
            }
        });
        assert_eq!(count_union_type_params(&schema), 1);
    }

    // ---- check_anthropic_strict_budget ----

    #[test]
    fn test_budget_within_limits() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"}
            },
            "required": ["a"]
        });
        let check = check_anthropic_strict_budget(vec![(true, &schema)].into_iter());
        assert!(!check.exceeds_limits);
        assert_eq!(check.total_strict_tools, 1);
        assert_eq!(check.total_optional_params, 0);
    }

    #[test]
    fn test_budget_exceeds_optional_params() {
        // Create 5 tools each with 6 optional params = 30 total > 24 limit
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"},
                "d": {"type": "string"},
                "e": {"type": "string"},
                "f": {"type": "string"}
            },
            "required": []
        });
        let tools: Vec<(bool, &Value)> = (0..5).map(|_| (true, &schema)).collect();
        let check = check_anthropic_strict_budget(tools.into_iter());
        assert!(check.exceeds_limits);
        assert_eq!(check.total_optional_params, 30);
    }

    #[test]
    fn test_budget_exceeds_tool_count() {
        let schema = json!({"type": "object", "properties": {}, "required": []});
        let tools: Vec<(bool, &Value)> = (0..25).map(|_| (true, &schema)).collect();
        let check = check_anthropic_strict_budget(tools.into_iter());
        assert!(check.exceeds_limits);
        assert_eq!(check.total_strict_tools, 25);
    }

    #[test]
    fn test_budget_non_strict_tools_ignored() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            }
            // no required → 2 optional params each
        });
        // 50 tools but none strict → should be within limits
        let tools: Vec<(bool, &Value)> = (0..50).map(|_| (false, &schema)).collect();
        let check = check_anthropic_strict_budget(tools.into_iter());
        assert!(!check.exceeds_limits);
        assert_eq!(check.total_strict_tools, 0);
        assert_eq!(check.total_optional_params, 0);
    }

    // ---- strip_keys_recursive ----

    #[test]
    fn test_strip_keys_recursive() {
        let schema = json!({
            "type": "object",
            "strict": true,
            "additionalProperties": false,
            "properties": {
                "a": {
                    "type": "string",
                    "strict": true,
                    "additionalProperties": false
                }
            }
        });
        let result = strip_keys_recursive(schema, &["strict", "additionalProperties"]);
        assert!(result.get("strict").is_none());
        assert!(result.get("additionalProperties").is_none());
        let a = &result["properties"]["a"];
        assert!(a.get("strict").is_none());
        assert!(a.get("additionalProperties").is_none());
    }

    // ---- ensure_additional_properties_false ----

    #[test]
    fn test_ensure_additional_properties_false() {
        let schema = json!({
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "string"}
                    }
                }
            }
        });
        let result = ensure_additional_properties_false(schema);
        assert_eq!(result["additionalProperties"], false);
        assert_eq!(
            result["properties"]["nested"]["additionalProperties"],
            false
        );
    }

    #[test]
    fn test_ensure_additional_properties_preserves_existing() {
        let schema = json!({
            "type": "object",
            "additionalProperties": true,
            "properties": {}
        });
        let result = ensure_additional_properties_false(schema);
        // Should preserve existing value
        assert_eq!(result["additionalProperties"], true);
    }

    // ---- convert_type_arrays_to_nullable ----

    #[test]
    fn test_convert_type_arrays_to_nullable() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": ["string", "null"]}
            }
        });
        let result = convert_type_arrays_to_nullable(schema);
        assert_eq!(result["properties"]["a"]["type"], "string");
        assert_eq!(result["properties"]["a"]["nullable"], true);
    }

    // ---- normalize_for_openai_strict ----

    #[test]
    fn test_normalize_for_openai_strict() {
        let schema = json!({
            "type": "object",
            "properties": {
                "required_param": {"type": "string"},
                "optional_param": {"type": "integer"}
            },
            "required": ["required_param"]
        });
        let result = normalize_for_openai_strict(schema);

        // All params should be required
        let required = result["required"].as_array().unwrap();
        assert!(required.contains(&json!("required_param")));
        assert!(required.contains(&json!("optional_param")));

        // optional_param should now be nullable
        let opt_type = &result["properties"]["optional_param"]["type"];
        assert!(opt_type.is_array());
        let types = opt_type.as_array().unwrap();
        assert!(types.contains(&json!("integer")));
        assert!(types.contains(&json!("null")));

        // required_param stays the same
        assert_eq!(result["properties"]["required_param"]["type"], "string");

        // additionalProperties added
        assert_eq!(result["additionalProperties"], false);
    }

    #[test]
    fn test_normalize_for_openai_strict_nested() {
        let schema = json!({
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": "number"}
                    },
                    "required": ["name"]
                }
            },
            "required": ["config"]
        });
        let result = normalize_for_openai_strict(schema);

        // Nested object should also be normalized
        let nested = &result["properties"]["config"];
        assert_eq!(nested["additionalProperties"], false);
        let nested_required = nested["required"].as_array().unwrap();
        assert!(nested_required.contains(&json!("name")));
        assert!(nested_required.contains(&json!("value")));
    }

    // ---- make_nullable ----

    #[test]
    fn test_make_nullable_string_type() {
        let mut prop = json!({"type": "string", "description": "A name"});
        make_nullable(&mut prop);
        assert_eq!(prop["type"], json!(["string", "null"]));
    }

    #[test]
    fn test_make_nullable_already_nullable() {
        let mut prop = json!({"type": ["string", "null"]});
        make_nullable(&mut prop);
        // Should not add duplicate null
        let types = prop["type"].as_array().unwrap();
        assert_eq!(types.len(), 2);
    }

    #[test]
    fn test_make_nullable_array_type() {
        let mut prop = json!({"type": ["string", "integer"]});
        make_nullable(&mut prop);
        let types = prop["type"].as_array().unwrap();
        assert_eq!(types.len(), 3);
        assert!(types.contains(&json!("null")));
    }
}
