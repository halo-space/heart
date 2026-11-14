//! Admission slots shared by nested live reply consumers.

use super::*;

#[derive(Clone)]
pub(crate) struct Slot(Arc<Admission>);
struct Admission {
    slots: Arc<tokio::sync::Semaphore>,
    permit: Mutex<Option<tokio::sync::OwnedSemaphorePermit>>,
    entering: tokio::sync::Mutex<()>,
    changed: Arc<AtomicWaker>,
}
impl Slot {
    pub(crate) fn waiting(slots: Arc<tokio::sync::Semaphore>, changed: Arc<AtomicWaker>) -> Self {
        Self(Arc::new(Admission {
            slots,
            permit: Mutex::new(None),
            entering: tokio::sync::Mutex::new(()),
            changed,
        }))
    }
    pub(crate) fn new(
        slots: Arc<tokio::sync::Semaphore>,
        changed: Arc<AtomicWaker>,
    ) -> Result<Self, Error> {
        let permit = slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::new("CONCURRENCY_LIMIT", "no planned task slot available"))?;
        Ok(Self::with_permit(slots, changed, permit))
    }
    pub(crate) fn with_permit(
        slots: Arc<tokio::sync::Semaphore>,
        changed: Arc<AtomicWaker>,
        permit: tokio::sync::OwnedSemaphorePermit,
    ) -> Self {
        let slot = Self::waiting(slots, changed);
        *slot.0.permit.lock().expect("task slot lock poisoned") = Some(permit);
        slot
    }
    pub(crate) fn held(&self) -> bool {
        self.0
            .permit
            .lock()
            .expect("task slot lock poisoned")
            .is_some()
    }
    pub(crate) fn release(&self) {
        let permit = self
            .0
            .permit
            .lock()
            .expect("task slot lock poisoned")
            .take();
        if permit.is_some() {
            drop(permit);
            self.0.changed.wake();
        }
    }
    pub(crate) async fn acquire(&self, cancellation: &Cancellation) -> Result<(), Error> {
        crate::cancellation::cancellable(
            async {
                let _entering = self.0.entering.lock().await;
                if !self.held() {
                    let permit =
                        self.0.slots.clone().acquire_owned().await.map_err(|_| {
                            Error::new("CANCELLED", "planned task admission closed")
                        })?;
                    *self.0.permit.lock().expect("task slot lock poisoned") = Some(permit);
                    self.0.changed.wake();
                }
                Ok(())
            },
            cancellation,
        )
        .await
    }
}
