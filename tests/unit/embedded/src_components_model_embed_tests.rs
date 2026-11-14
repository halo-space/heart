use super::*;

#[test]
fn validates_single_input_and_dimension() {
    let request = Request {
        input: " \n".into(),
        dimensions: None,
        options: Map::new(),
    };
    assert_eq!(request.validate().unwrap_err().code, "INVALID_ARGUMENTS");
    let request = Request {
        input: "北京".into(),
        dimensions: Some(0),
        options: Map::new(),
    };
    assert_eq!(request.validate().unwrap_err().code, "INVALID_ARGUMENTS");
}

#[test]
fn vector_uses_values_and_source_uses_metadata() {
    let output = message(vec![0.1, -0.3], "example", "embed-v1", Some(2)).unwrap();
    assert_eq!(output.role, Role::Assistant);
    assert_eq!(output.values["vector"].as_array().unwrap().len(), 2);
    assert_eq!(output.metadata["dimension"], 2);
    assert!(output.values.get("dimension").is_none());
    assert_eq!(
        message(vec![0.1], "example", "embed-v1", Some(2))
            .unwrap_err()
            .code,
        "INVALID_RESPONSE"
    );
}
