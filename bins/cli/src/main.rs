mod args;
mod commands;
mod output;

use clap::Parser;

use crate::args::{Cli, Commands};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Version => commands::version::run(),
        Commands::Migrate => commands::migrate::run().await,
        Commands::Generate(ref args) => commands::generate::run(args).await,
        Commands::Validate(ref args) => commands::validate::run(args).await,
        Commands::ObsidianSync(ref args) => commands::obsidian_sync::run(args).await,
        Commands::TagAudit(ref args) => commands::tag_audit::run(args).await,
    };

    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use crate::args::{Cli, Commands};

    fn parse(args: &[&str]) -> Cli {
        Cli::parse_from(args)
    }

    fn try_parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(args)
    }

    // ---- Version ----

    #[test]
    fn parse_version_command() {
        let cli = parse(&["anki-atlas", "version"]);
        assert!(matches!(cli.command, Commands::Version));
    }

    // ---- Migrate ----

    #[test]
    fn parse_migrate_command() {
        let cli = parse(&["anki-atlas", "migrate"]);
        assert!(matches!(cli.command, Commands::Migrate));
    }

    // ---- Generate ----

    #[test]
    fn parse_generate_with_file() {
        let cli = parse(&["anki-atlas", "generate", "note.md"]);
        if let Commands::Generate(args) = cli.command {
            assert_eq!(args.file.to_str().unwrap(), "note.md");
            assert!(!args.dry_run);
        } else {
            panic!("expected Generate command");
        }
    }

    #[test]
    fn parse_generate_dry_run() {
        let cli = parse(&["anki-atlas", "generate", "note.md", "--dry-run"]);
        if let Commands::Generate(args) = cli.command {
            assert!(args.dry_run);
        } else {
            panic!("expected Generate command");
        }
    }

    #[test]
    fn parse_generate_requires_file() {
        let result = try_parse(&["anki-atlas", "generate"]);
        assert!(result.is_err());
    }

    // ---- Validate ----

    #[test]
    fn parse_validate_with_file() {
        let cli = parse(&["anki-atlas", "validate", "cards.txt"]);
        if let Commands::Validate(args) = cli.command {
            assert_eq!(args.file.to_str().unwrap(), "cards.txt");
            assert!(!args.quality);
        } else {
            panic!("expected Validate command");
        }
    }

    #[test]
    fn parse_validate_with_quality() {
        let cli = parse(&["anki-atlas", "validate", "cards.txt", "--quality"]);
        if let Commands::Validate(args) = cli.command {
            assert!(args.quality);
        } else {
            panic!("expected Validate command");
        }
    }

    #[test]
    fn parse_validate_requires_file() {
        let result = try_parse(&["anki-atlas", "validate"]);
        assert!(result.is_err());
    }

    // ---- ObsidianSync ----

    #[test]
    fn parse_obsidian_sync_with_vault() {
        let cli = parse(&["anki-atlas", "obsidian-sync", "/vault"]);
        if let Commands::ObsidianSync(args) = cli.command {
            assert_eq!(args.vault.to_str().unwrap(), "/vault");
            assert!(args.source_dirs.is_none());
            assert!(!args.dry_run);
        } else {
            panic!("expected ObsidianSync command");
        }
    }

    #[test]
    fn parse_obsidian_sync_with_options() {
        let cli = parse(&[
            "anki-atlas",
            "obsidian-sync",
            "/vault",
            "--source-dirs",
            "notes,projects",
            "--dry-run",
        ]);
        if let Commands::ObsidianSync(args) = cli.command {
            assert_eq!(args.source_dirs.as_deref(), Some("notes,projects"));
            assert!(args.dry_run);
        } else {
            panic!("expected ObsidianSync command");
        }
    }

    #[test]
    fn parse_obsidian_sync_requires_vault() {
        let result = try_parse(&["anki-atlas", "obsidian-sync"]);
        assert!(result.is_err());
    }

    // ---- TagAudit ----

    #[test]
    fn parse_tag_audit_with_file() {
        let cli = parse(&["anki-atlas", "tag-audit", "tags.txt"]);
        if let Commands::TagAudit(args) = cli.command {
            assert_eq!(args.file.to_str().unwrap(), "tags.txt");
            assert!(!args.fix);
        } else {
            panic!("expected TagAudit command");
        }
    }

    #[test]
    fn parse_tag_audit_with_fix() {
        let cli = parse(&["anki-atlas", "tag-audit", "tags.txt", "--fix"]);
        if let Commands::TagAudit(args) = cli.command {
            assert!(args.fix);
        } else {
            panic!("expected TagAudit command");
        }
    }

    #[test]
    fn parse_tag_audit_requires_file() {
        let result = try_parse(&["anki-atlas", "tag-audit"]);
        assert!(result.is_err());
    }

    // ---- Error cases ----

    #[test]
    fn parse_unknown_command_fails() {
        let result = try_parse(&["anki-atlas", "unknown"]);
        assert!(result.is_err());
    }

    #[test]
    fn parse_no_command_fails() {
        let result = try_parse(&["anki-atlas"]);
        assert!(result.is_err());
    }

    // ---- Help ----

    #[test]
    fn help_flag_is_recognized() {
        let result = try_parse(&["anki-atlas", "--help"]);
        // --help causes clap to return an error (DisplayHelp variant)
        assert!(result.is_err());
    }

    #[test]
    fn subcommand_help_is_recognized() {
        let result = try_parse(&["anki-atlas", "generate", "--help"]);
        assert!(result.is_err());
    }
}
