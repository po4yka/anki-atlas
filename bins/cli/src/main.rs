mod args;
mod commands;
mod output;
mod runtime;
mod tui;
mod usecases;

use clap::Parser;

use crate::args::{Cli, Commands};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
    let cli = Cli::parse();

    let result = match &cli.command {
        Commands::Version => commands::version::run(),
        Commands::Migrate => {
            let settings = common::config::Settings::load()?;
            commands::migrate::run(&settings).await
        }
        Commands::Tui => tui::run().await,
        Commands::Generate(args) => commands::generate::run(args).await,
        Commands::Validate(args) => commands::validate::run(args).await,
        Commands::ObsidianSync(args) => commands::obsidian_sync::run(args).await,
        Commands::TagAudit(args) => commands::tag_audit::run(args).await,
        Commands::Cardloop(args) => commands::cardloop::run(args).await,
        Commands::Sync(_)
        | Commands::Index(_)
        | Commands::Search(_)
        | Commands::Topics(_)
        | Commands::Coverage(_)
        | Commands::Gaps(_)
        | Commands::WeakNotes(_)
        | Commands::Duplicates(_) => runtime::dispatch_service_command(&cli.command).await,
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
    fn parse_tui_command() {
        let cli = Cli::parse_from(["anki-atlas", "tui"]);
        match cli.command {
            Commands::Tui => {}
            _ => panic!("expected tui command"),
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
