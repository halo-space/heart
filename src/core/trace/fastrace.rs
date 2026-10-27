//! `fastrace` implementation for Heart's trace projection.

use std::borrow::Cow;
use std::future::Future;
use std::sync::Arc;

use ::fastrace::collector::{Reporter, SpanContext};
use ::fastrace::{Event, Span};

pub use ::fastrace::collector::{Config, ConsoleReporter};

/// A trace root for one finite Agent or Workflow invocation.
///
/// A waiting invocation is closed when control returns to the caller. A later
/// `reply` creates a new invocation span and carries logical correlation
/// fields as properties.
#[derive(Clone)]
pub struct Root {
    span: Arc<Span>,
}

impl Root {
    /// Create a disabled trace handle for a stream that has no active run.
    pub(crate) fn noop() -> Self {
        Self {
            span: Arc::new(Span::noop()),
        }
    }

    /// Start a new trace root.
    pub fn start(name: impl Into<Cow<'static, str>>) -> Self {
        Self {
            span: Arc::new(Span::root(name, SpanContext::random())),
        }
    }

    /// Start a child span with this root as its explicit parent.
    pub fn child(&self, name: impl Into<Cow<'static, str>>) -> SpanScope {
        SpanScope {
            span: Span::enter_with_parent(name, &self.span),
        }
    }

    /// Create a handle that treats the new child span as the parent of a
    /// nested execution. This is used for planned Agent nodes so their model
    /// and tool spans appear below the node in the same trace tree.
    pub(crate) fn child_root(&self, name: impl Into<Cow<'static, str>>) -> Self {
        Self {
            span: Arc::new(Span::enter_with_parent(name, &self.span)),
        }
    }

    /// Add an event to the run span.
    pub fn event(&self, name: impl Into<Cow<'static, str>>) {
        self.span.add_event(Event::new(name));
    }

    /// Add a lifecycle event with string properties.
    pub fn event_properties(
        &self,
        name: impl Into<Cow<'static, str>>,
        properties: Vec<(String, String)>,
    ) {
        self.span
            .add_event(Event::new(name).with_properties(|| properties));
    }

    /// Add a property to the run span.
    pub fn property<K, V>(&self, key: K, value: V)
    where
        K: Into<Cow<'static, str>>,
        V: Into<Cow<'static, str>>,
    {
        self.span.add_property(|| (key, value));
    }

    /// Run a future inside a child span. This keeps the span alive across
    /// `.await` and is safe for concurrent task execution.
    pub fn in_span<F>(
        &self,
        name: impl Into<Cow<'static, str>>,
        future: F,
    ) -> impl Future<Output = F::Output> + 'static
    where
        F: Future + 'static,
    {
        use ::fastrace::future::FutureExt as _;

        let span = Span::enter_with_parent(name, &self.span);
        future.in_span(span)
    }
}

/// A finite unit of work below a [`Root`] or another span.
pub struct SpanScope {
    span: Span,
}

impl SpanScope {
    /// Start a child span with this scope as its explicit parent.
    pub fn child(&self, name: impl Into<Cow<'static, str>>) -> SpanScope {
        SpanScope {
            span: Span::enter_with_parent(name, &self.span),
        }
    }

    /// Add an event to this span.
    pub fn event(&self, name: impl Into<Cow<'static, str>>) {
        self.span.add_event(Event::new(name));
    }

    /// Add a property to this span.
    pub fn property<K, V>(&self, key: K, value: V)
    where
        K: Into<Cow<'static, str>>,
        V: Into<Cow<'static, str>>,
    {
        self.span.add_property(|| (key, value));
    }

    /// Run a future inside a child span below this scope.
    pub fn in_span<F>(
        self,
        name: impl Into<Cow<'static, str>>,
        future: F,
    ) -> impl Future<Output = F::Output> + 'static
    where
        F: Future + 'static,
    {
        use ::fastrace::future::FutureExt as _;

        let span = Span::enter_with_parent(name, &self.span);
        future.in_span(span)
    }

    /// Run a future with a borrowed span parent. This is used when the future
    /// itself has a non-static lifetime, such as a workflow node callback.
    pub fn in_span_ref<'a, F>(
        &'a self,
        name: impl Into<Cow<'static, str>>,
        future: F,
    ) -> impl Future<Output = F::Output> + 'a
    where
        F: Future + 'a,
    {
        use ::fastrace::future::FutureExt as _;

        let span = Span::enter_with_parent(name, &self.span);
        future.in_span(span)
    }
}

/// Flush pending fastrace records to the configured application reporter.
pub fn flush() {
    ::fastrace::flush();
}

/// Install the application's fastrace Reporter.
///
/// Heart does not choose a backend. Applications can pass the built-in
/// ConsoleReporter or a Reporter supplied by an integration crate.
pub fn set_reporter<R>(reporter: R, config: Config)
where
    R: Reporter,
{
    ::fastrace::set_reporter(reporter, config);
}

#[cfg(test)]
mod tests {
    use super::Root;

    #[test]
    fn builds_explicit_parent_child_spans() {
        let root = Root::start("heart.test.run");
        root.property("chat_id", "1");
        root.event("run.started");

        let node = root.child("heart.test.node");
        node.property("node_id", "answer");
        node.event("node.completed");

        let model = node.child("heart.test.model");
        model.property("provider", "test");
        model.event("model.completed");
    }
}
