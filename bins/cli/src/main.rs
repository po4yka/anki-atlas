mod args;
mod commands;
mod output;

use clap::Parser;

use crate::args::{Cli, Commands};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let result = match &cli.command {
        Commands::Version => commands::version::run(),
        Commands::Migrate => {
            let settings = common::config::Settings::load()?;
            commands::migrate::run(&settings).await
        }
        Commands::Generate(args) => commands::generate::run(args).await,
        Commands::Validate(args) => commands::validate::run(args).await,
        Commands::ObsidianSync(args) => commands::obsidian_sync::run(args).await,
        Commands::TagAudit(args) => commands::tag_audit::run(args).await,
        Commands::Sync(args) => {
            let settings = common::config::Settings::load()?;
            let services = surface_runtime::build_surface_services(
                &settings,
                surface_runtime::BuildSurfaceServicesOptions {
                    enable_direct_execution: true,
                },
            )
            .await?;
            commands::sync::run(args, &services).await
        }
        Commands::Index(args) => {
            let settings = common::config::Settings::load()?;
            let services = surface_runtime::build_surface_services(
                &settings,
                surface_runtime::BuildSurfaceServicesOptions {
                    enable_direct_execution: true,
                },
            )
            .await?;
            commands::index::run(args, &services).await
        }
        Commands::Search(args) => {
            let settings = common::config::Settings::load()?;
            let services = surface_runtime::build_surface_services(
                &settings,
                surface_runtime::BuildSurfaceServicesOptions::default(),
            )
            .await?;
            commands::search::run(args, &services).await
        }
        Commands::Topics(args) => {
            let settings = common::config::Settings::load()?;
            let services = surface_runtime::build_surface_services(
                &settings,
                surface_runtime::BuildSurfaceServicesOptions::default(),
            )
            .await?;
            commands::topics::run(args, &services).await
        }
        Commands::Coverage(args) => {
            let settings = common::config::Settings::load()?;
            let services = surface_runtime::build_surface_services(
                &settings,
                surface_runtime::BuildSurfaceServicesOptions::default(),
            )
            .await?;
            commands::coverage::run(args, &services).await
        }
        Commands::Gaps(args) => {
            let settings = common::config::Settings::load()?;
            let services = surface_runtime::build_surface_services(
                &settings,
                surface_runtime::BuildSurfaceServicesOptions::default(),
            )
            .await?;
            commands::gaps::run(args, &services).await
        }
        Commands::WeakNotes(args) => {
            let settings = common::config::Settings::load()?;
            let services = surface_runtime::build_surface_services(
                &settings,
                surface_runtime::BuildSurfaceServicesOptions::default(),
            )
            .await?;
            commands::weak_notes::run(args, &services).await
        }
        Commands::Duplicates(args) => {
            let settings = common::config::Settings::load()?;
            let services = surface_runtime::build_surface_services(
                &settings,
                surface_runtime::BuildSurfaceServicesOptions::default(),
            )
            .await?;
            commands::duplicates::run(args, &services).await
        }
    };

    if let Err(error) = result {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use crate::args::{Cli, Commands, TopicsCommand};

    #[test]
    fn parse_sync_command() {
        let cli = Cli::parse_from(["anki-atlas", "sync", "collection.anki2"]);
        match cli.command {
            Commands::Sync(args) => {
                assert_eq!(args.source.to_string_lossy(), "collection.anki2");
                assert!(!args.no_migrate);
                assert!(!args.no_index);
            }
            _ => panic!("expected sync command"),
        }
    }

    #[test]
    fn parse_search_command() {
        let cli = Cli::parse_from([
            "anki-atlas",
            "search",
            "ownership",
            "--deck",
            "Rust",
            "--tag",
            "topic::ownership",
            "-n",
            "5",
            "--verbose",
        ]);
        match cli.command {
            Commands::Search(args) => {
                assert_eq!(args.query, "ownership");
                assert_eq!(args.deck_names, vec!["Rust"]);
                assert_eq!(args.tags, vec!["topic::ownership"]);
                assert_eq!(args.limit, 5);
                assert!(args.verbose);
            }
            _ => panic!("expected search command"),
        }
    }

    #[test]
    fn parse_nested_topics_commands() {
        let cli = Cli::parse_from(["anki-atlas", "topics", "tree", "--root-path", "rust"]);
        match cli.command {
            Commands::Topics(args) => match args.command {
                TopicsCommand::Tree(tree) => {
                    assert_eq!(tree.root_path.as_deref(), Some("rust"));
                }
                _ => panic!("expected topics tree"),
            },
            _ => panic!("expected topics command"),
        }
    }

    #[test]
    fn parse_weak_notes_command() {
        let cli = Cli::parse_from(["anki-atlas", "weak-notes", "rust/ownership", "-n", "5"]);
        match cli.command {
            Commands::WeakNotes(args) => {
                assert_eq!(args.topic, "rust/ownership");
                assert_eq!(args.limit, 5);
            }
            _ => panic!("expected weak-notes command"),
        }
    }
}
