use search::fts::SearchFilters;
use search::service::SearchParams;
use surface_runtime::SurfaceServices;

use crate::args::SearchArgs;
use crate::output;

pub async fn run(args: &SearchArgs, services: &SurfaceServices) -> anyhow::Result<()> {
    anyhow::ensure!(
        !(args.semantic && args.fts),
        "--semantic and --fts are mutually exclusive"
    );
    let filters = (!args.deck_names.is_empty() || !args.tags.is_empty()).then(|| SearchFilters {
        deck_names: (!args.deck_names.is_empty()).then(|| args.deck_names.clone()),
        tags: (!args.tags.is_empty()).then(|| args.tags.clone()),
        ..Default::default()
    });
    let params = SearchParams {
        query: args.query.clone(),
        filters,
        limit: args.limit,
        semantic_weight: 1.0,
        fts_weight: 1.0,
        semantic_only: args.semantic,
        fts_only: args.fts,
        rerank_override: None,
        rerank_top_n_override: None,
    };
    let result = services.search.search(&params).await?;
    output::print_search_result(&result, args.verbose);
    Ok(())
}
