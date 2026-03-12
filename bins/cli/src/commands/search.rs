use surface_runtime::SurfaceServices;

use crate::args::SearchArgs;
use crate::output;
use crate::usecases::{self, RuntimeHandles, SearchRequest};

pub async fn run(args: &SearchArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    let handles = RuntimeHandles::from(services);
    let result = usecases::search(
        handles,
        SearchRequest {
            query: args.query.clone(),
            deck_names: args.deck_names.clone(),
            tags: args.tags.clone(),
            limit: args.limit,
            semantic_only: args.semantic,
            fts_only: args.fts,
        },
    )
    .await?;
    output::print_search_result(&result, args.verbose);
    Ok(())
}
