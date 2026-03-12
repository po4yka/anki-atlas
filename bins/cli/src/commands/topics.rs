use surface_runtime::SurfaceServices;

use crate::args::{TopicsArgs, TopicsCommand};
use crate::output;
use crate::usecases::{
    self, RuntimeHandles, TopicsLabelRequest, TopicsLoadRequest, TopicsTreeRequest,
};

pub async fn run(args: &TopicsArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    let handles = RuntimeHandles::from(services);
    match &args.command {
        TopicsCommand::Tree(tree) => {
            let topics = usecases::topics_tree(
                handles.clone(),
                TopicsTreeRequest {
                    root_path: tree.root_path.clone(),
                },
            )
            .await?;
            output::print_topics_tree(&topics)?;
        }
        TopicsCommand::Load(load) => {
            let taxonomy = usecases::topics_load(
                handles.clone(),
                TopicsLoadRequest {
                    file: load.file.clone(),
                },
            )
            .await?;
            output::print_taxonomy_load(taxonomy.topics.len(), taxonomy.roots.len());
        }
        TopicsCommand::Label(label) => {
            let stats = usecases::topics_label(
                handles,
                TopicsLabelRequest {
                    file: label.file.clone(),
                    min_confidence: label.min_confidence,
                },
            )
            .await?;
            output::print_labeling_summary(&stats);
        }
    }
    Ok(())
}
