use common::config::ApiSettings;

use crate::state::AppState;

pub use surface_runtime::services::{
    AnalyticsFacade, SearchFacade, SurfaceServices as ApiServices,
};

pub async fn build_api_services(
    settings: &common::config::Settings,
) -> anyhow::Result<ApiServices> {
    Ok(surface_runtime::build_surface_services(
        settings,
        surface_runtime::BuildSurfaceServicesOptions::default(),
    )
    .await?)
}

pub fn build_app_state(api_settings: ApiSettings, services: ApiServices) -> AppState {
    AppState {
        api: std::sync::Arc::new(api_settings),
        services: std::sync::Arc::new(services),
    }
}
