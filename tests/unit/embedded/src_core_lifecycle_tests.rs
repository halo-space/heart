#![allow(unused_imports)]
use crate::core::lifecycle::{Methods, Phase, advance};
use halo::agent::{Cancellation, Error};

mod unit {
    use super::*;
    include!("../lifecycle.rs");
}
