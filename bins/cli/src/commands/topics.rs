use surface_runtime::SurfaceServices;

use crate::args::{TopicsArgs, TopicsCommand};
use crate::output;

pub async fn run(args: &TopicsArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    match &args.command {
        TopicsCommand::Tree(tree) => {
            let topics = services
                .analytics
                .get_taxonomy_tree(tree.root_path.clone())
                .await?;
            output::print_topics_tree(&topics)?;
        }
        TopicsCommand::Load(load) => {
            let taxonomy = services
                .analytics
                .load_taxonomy(Some(load.file.clone()))
                .await?;
            output::print_taxonomy_load(taxonomy.topics.len(), taxonomy.roots.len());
        }
        TopicsCommand::Label(label) => {
            let stats = services
                .analytics
                .label_notes(label.file.clone(), label.min_confidence)
                .await?;
            output::print_labeling_summary(&stats);
        }
    }
    Ok(())
}
