use surface_contracts::search::SearchMode;
use surface_runtime::SurfaceServices;

use crate::args::SearchArgs;
use crate::output;
use crate::usecases::{self, RuntimeHandles, SearchRequest};

pub async fn run(args: &SearchArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    let handles = RuntimeHandles::from(services);
    if args.chunks {
        anyhow::ensure!(
            !args.fts,
            "--fts is not supported with --chunks; chunk search is semantic-only"
        );
        let result = usecases::search_chunks(
            handles,
            usecases::ChunkSearchRequest {
                query: args.query.clone(),
                deck_names: args.deck_names.clone(),
                tags: args.tags.clone(),
                limit: args.limit,
            },
        )
        .await?;
        output::print_chunk_search_result(&result, args.verbose);
    } else {
        let search_mode = if args.semantic {
            SearchMode::SemanticOnly
        } else if args.fts {
            SearchMode::FtsOnly
        } else {
            SearchMode::Hybrid
        };
        let result = usecases::search(
            handles,
            SearchRequest {
                query: args.query.clone(),
                deck_names: args.deck_names.clone(),
                tags: args.tags.clone(),
                limit: args.limit,
                search_mode,
            },
        )
        .await?;
        output::print_search_result(&result, args.verbose);
    }
    Ok(())
}
