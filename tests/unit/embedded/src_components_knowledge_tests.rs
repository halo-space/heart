use super::{hybrid, keyword, vector};

#[test]
fn search_modes_validate_their_own_query_shape() {
    assert!(
        keyword::Request {
            query: "北京".into(),
            ..Default::default()
        }
        .validate()
        .is_ok()
    );
    assert!(keyword::Request::default().validate().is_err());
    assert!(
        vector::Request {
            vector: vec![0.1, 0.2],
            ..Default::default()
        }
        .validate()
        .is_ok()
    );
    assert!(vector::Request::default().validate().is_err());
    assert!(
        hybrid::Request {
            query: "北京".into(),
            vector: vec![0.1],
            ..Default::default()
        }
        .validate()
        .is_ok()
    );
    assert!(hybrid::Request::default().validate().is_err());
}
