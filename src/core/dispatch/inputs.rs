//! Private input adaptation for the public `.tools(...)` builder methods.
//!
//! Dispatch remains responsible for the assembled capability set; this file
//! only handles accepting one Tool or a homogeneous collection without
//! leaking another public registry or adapter type.

use std::sync::Arc;

use crate::components::tool::Tool;

use super::{Dispatch, tool};

pub trait ToolInputs {
    fn append_to(self, dispatch: &mut Dispatch);
}

impl<T> ToolInputs for T
where
    T: Tool + 'static,
{
    fn append_to(self, dispatch: &mut Dispatch) {
        dispatch.tools.push(Arc::new(tool::adapter::Tool {
            tool: Arc::new(self),
        }));
    }
}

impl<T, const N: usize> ToolInputs for [T; N]
where
    T: Tool + 'static,
{
    fn append_to(self, dispatch: &mut Dispatch) {
        for tool in self {
            dispatch.tools.push(Arc::new(tool::adapter::Tool {
                tool: Arc::new(tool),
            }));
        }
    }
}

impl<T> ToolInputs for Vec<T>
where
    T: Tool + 'static,
{
    fn append_to(self, dispatch: &mut Dispatch) {
        for tool in self {
            dispatch.tools.push(Arc::new(tool::adapter::Tool {
                tool: Arc::new(tool),
            }));
        }
    }
}
