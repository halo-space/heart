#![allow(unused_imports)]
use crate::runtime::agent::model_steps;
use halo::agent::message::Role;
use halo::agent::model;
use halo::agent::model::chat::Request;
use halo::agent::model::content::part::Part;
use halo::agent::model::token::Usage;
use halo::agent::{Error, Message, Messages};

mod unit {
    use super::*;
    include!("../agent_model_partial.rs");
}
