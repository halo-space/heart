use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::Cancellation;
use crate::components::agent::Error;
use crate::components::skill::{Definition as SkillDefinition, Loader, Skill};
use crate::components::tool::Call;
use crate::workspace::Workspace as WorkspaceContract;

pub(crate) const READER_NAME: &str = "skill";

type SkillFuture = Pin<Box<dyn Future<Output = Result<String, Error>> + Send + 'static>>;
type ResourceFuture = Pin<Box<dyn Future<Output = Result<Vec<u8>, Error>> + Send + 'static>>;
type DiscoveryFuture =
    Pin<Box<dyn Future<Output = Result<Vec<Arc<dyn SkillView>>, Error>> + Send + 'static>>;

#[derive(Clone)]
pub(crate) struct SkillSnapshot {
    version: String,
    entries: Vec<Arc<dyn SkillView>>,
}

/// One Skill that can expose its instructions and optional resources.
/// The entry hides whether the content is direct, workspace-backed or loaded
/// from another source.
pub(crate) trait SkillView: Send + Sync {
    fn definition(&self) -> &SkillDefinition;
    fn read(&self, cancellation: Cancellation) -> SkillFuture;
    fn read_resource(&self, _path: String, _cancellation: Cancellation) -> ResourceFuture {
        Box::pin(async {
            Err(Error::new(
                "NOT_FOUND",
                "Skill has no Workspace resource source",
            ))
        })
    }
}

/// Object-safe discovery boundary for the public `skill::Loader` contract.
pub(crate) trait LoaderSource: Send + Sync {
    fn discover(&self, cancellation: Cancellation) -> DiscoveryFuture;
}

/// Adapts one concrete `skill::Loader` and keeps its versioned catalog cache.
pub(crate) struct LoaderAdapter<S> {
    pub(crate) loader: Arc<S>,
    pub(crate) cache: Arc<RwLock<Option<SkillSnapshot>>>,
}

impl<S: Loader + 'static> LoaderSource for LoaderAdapter<S> {
    fn discover(&self, cancellation: Cancellation) -> DiscoveryFuture {
        let loader = Arc::clone(&self.loader);
        let cache = Arc::clone(&self.cache);
        Box::pin(async move {
            let version = loader.version(&cancellation).await?;
            if let Some(snapshot) = cache.read().await.as_ref()
                && snapshot.version == version
            {
                return Ok(snapshot.entries.clone());
            }

            let definitions = loader.list(&cancellation).await?;
            let entries: Vec<Arc<dyn SkillView>> = definitions
                .into_iter()
                .map(|definition| {
                    Arc::new(LoadedSkill {
                        loader: Arc::clone(&loader),
                        definition,
                    }) as Arc<dyn SkillView>
                })
                .collect();
            *cache.write().await = Some(SkillSnapshot {
                version,
                entries: entries.clone(),
            });
            Ok(entries)
        })
    }
}

/// A catalog definition whose body is read from its Loader by name.
struct LoadedSkill<S> {
    loader: Arc<S>,
    definition: SkillDefinition,
}

impl<S: Loader + 'static> SkillView for LoadedSkill<S> {
    fn definition(&self) -> &SkillDefinition {
        &self.definition
    }

    fn read(&self, cancellation: Cancellation) -> SkillFuture {
        let loader = Arc::clone(&self.loader);
        let name = self.definition.name.clone();
        Box::pin(async move { loader.read(&name, &cancellation).await })
    }

    fn read_resource(&self, path: String, cancellation: Cancellation) -> ResourceFuture {
        let loader = Arc::clone(&self.loader);
        let name = self.definition.name.clone();
        Box::pin(async move { loader.read_resource(&name, &path, &cancellation).await })
    }
}

/// A Skill supplied directly by the application instead of a Loader.
pub(crate) struct Direct<S> {
    pub(crate) skill: Arc<S>,
    pub(crate) definition: SkillDefinition,
}

impl<S: Skill + 'static> SkillView for Direct<S> {
    fn definition(&self) -> &SkillDefinition {
        &self.definition
    }

    fn read(&self, cancellation: Cancellation) -> SkillFuture {
        let skill = Arc::clone(&self.skill);
        Box::pin(async move { skill.read(&cancellation).await })
    }
}

/// A Skill exposed by a Workspace's own Skill read capability.
pub(crate) struct Workspace {
    pub(crate) workspace: Arc<dyn WorkspaceContract>,
    pub(crate) definition: SkillDefinition,
}

impl SkillView for Workspace {
    fn definition(&self) -> &SkillDefinition {
        &self.definition
    }

    fn read(&self, cancellation: Cancellation) -> SkillFuture {
        let workspace = Arc::clone(&self.workspace);
        let name = self.definition.name.clone();
        Box::pin(async move {
            let call = Call {
                call_id: format!("skill_{name}"),
                name: READER_NAME.into(),
                arguments: serde_json::Map::from_iter([("skill".into(), name.into())]),
            };
            let (message, evaluation) = workspace.execute(call, &cancellation).await?;
            if !evaluation.passed {
                return Err(Error::new(
                    "EXECUTION_FAILED",
                    "Workspace Skill evaluation failed",
                ));
            }
            message
                .values
                .get("instructions")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| {
                    Error::new(
                        "INVALID_ARGUMENTS",
                        "Workspace Skill returned no instructions",
                    )
                })
        })
    }

    fn read_resource(&self, path: String, cancellation: Cancellation) -> ResourceFuture {
        let workspace = Arc::clone(&self.workspace);
        let name = self.definition.name.clone();
        Box::pin(async move {
            let call = Call {
                call_id: format!("skill_{name}_resource"),
                name: READER_NAME.into(),
                arguments: serde_json::Map::from_iter([
                    ("skill".into(), name.into()),
                    ("resource".into(), path.into()),
                ]),
            };
            let (message, evaluation) = workspace.execute(call, &cancellation).await?;
            if !evaluation.passed {
                return Err(Error::new(
                    "EXECUTION_FAILED",
                    "Workspace Skill evaluation failed",
                ));
            }
            message
                .values
                .get("content")
                .and_then(serde_json::Value::as_str)
                .map(|content| content.as_bytes().to_vec())
                .ok_or_else(|| {
                    Error::new("INVALID_ARGUMENTS", "Workspace Skill returned no resource")
                })
        })
    }
}
