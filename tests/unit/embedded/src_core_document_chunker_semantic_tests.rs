use super::{candidate_cache_key, estimate, similarity};

#[test]
fn similarity_computes_cosine_without_scale_overflow() {
    assert_eq!(similarity(Some(&[1.0, 0.0]), Some(&[1.0, 0.0])), Some(1.0));
    assert_eq!(similarity(Some(&[1.0, 0.0]), Some(&[0.0, 1.0])), Some(0.0));
    assert_eq!(
        similarity(Some(&[1.0, 0.0]), Some(&[-1.0, 0.0])),
        Some(-1.0)
    );
    assert_eq!(similarity(None, Some(&[1.0])), None);
    assert_eq!(similarity(Some(&[f64::MAX]), Some(&[f64::MAX])), Some(1.0));
}

#[test]
fn candidate_cache_key_tracks_source_and_candidate_boundaries() {
    let whole = candidate_cache_key("first. second.", &[0..7, 7..14]);
    assert_eq!(whole, candidate_cache_key("first. second.", &[0..7, 7..14]));
    assert_ne!(whole, candidate_cache_key("other. second.", &[0..7, 7..14]));
    assert_ne!(whole, candidate_cache_key("first. second.", &[0..6, 6..14]));
}

#[test]
fn default_length_uses_the_documented_utf8_byte_estimate() {
    assert_eq!(estimate("abcd").unwrap(), 1);
    assert_eq!(estimate("北京").unwrap(), 2);
    assert_eq!(estimate("").unwrap(), 0);
}
