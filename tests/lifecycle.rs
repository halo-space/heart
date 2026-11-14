#![allow(unused_imports)]
use halo::agent::core::lifecycle::{Methods, Phase, advance};
use halo::agent::{Cancellation, Error};
mod unit {
    use super::*;
    include!("unit/lifecycle.rs");
}
