mod args;
mod commands;
mod output;

use clap::Parser;

use crate::args::{Cli, Commands};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Version => {
            todo!("version command not implemented")
        }
        Commands::Sync(_args) => {
            todo!("sync command not implemented")
        }
        Commands::Migrate => {
            todo!("migrate command not implemented")
        }
        Commands::Index(_args) => {
            todo!("index command not implemented")
        }
        Commands::Search(_args) => {
            todo!("search command not implemented")
        }
        Commands::Topics(_args) => {
            todo!("topics command not implemented")
        }
        Commands::Coverage(_args) => {
            todo!("coverage command not implemented")
        }
        Commands::Gaps(_args) => {
            todo!("gaps command not implemented")
        }
        Commands::Duplicates(_args) => {
            todo!("duplicates command not implemented")
        }
        Commands::Generate(_args) => {
            todo!("generate command not implemented")
        }
        Commands::Validate(_args) => {
            todo!("validate command not implemented")
        }
        Commands::ObsidianSync(_args) => {
            todo!("obsidian-sync command not implemented")
        }
        Commands::TagAudit(_args) => {
            todo!("tag-audit command not implemented")
        }
    }
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

    // ---- Sync ----

    #[test]
    fn parse_sync_with_source() {
        let cli = parse(&["anki-atlas", "sync", "--source", "/path/to/collection.anki2"]);
        if let Commands::Sync(args) = cli.command {
            assert_eq!(args.source, "/path/to/collection.anki2");
            assert!(!args.no_migrate);
            assert!(!args.no_index);
            assert!(!args.force_reindex);
        } else {
            panic!("expected Sync command");
        }
    }

    #[test]
    fn parse_sync_short_source() {
        let cli = parse(&["anki-atlas", "sync", "-s", "/path/to/col.anki2"]);
        if let Commands::Sync(args) = cli.command {
            assert_eq!(args.source, "/path/to/col.anki2");
        } else {
            panic!("expected Sync command");
        }
    }

    #[test]
    fn parse_sync_with_all_flags() {
        let cli = parse(&[
            "anki-atlas",
            "sync",
            "--source",
            "/col.anki2",
            "--no-migrate",
            "--no-index",
            "--force-reindex",
        ]);
        if let Commands::Sync(args) = cli.command {
            assert!(args.no_migrate);
            assert!(args.no_index);
            assert!(args.force_reindex);
        } else {
            panic!("expected Sync command");
        }
    }

    #[test]
    fn parse_sync_requires_source() {
        let result = try_parse(&["anki-atlas", "sync"]);
        assert!(result.is_err());
    }

    // ---- Migrate ----

    #[test]
    fn parse_migrate_command() {
        let cli = parse(&["anki-atlas", "migrate"]);
        assert!(matches!(cli.command, Commands::Migrate));
    }

    // ---- Index ----

    #[test]
    fn parse_index_defaults() {
        let cli = parse(&["anki-atlas", "index"]);
        if let Commands::Index(args) = cli.command {
            assert!(!args.force);
        } else {
            panic!("expected Index command");
        }
    }

    #[test]
    fn parse_index_with_force() {
        let cli = parse(&["anki-atlas", "index", "--force"]);
        if let Commands::Index(args) = cli.command {
            assert!(args.force);
        } else {
            panic!("expected Index command");
        }
    }

    #[test]
    fn parse_index_short_force() {
        let cli = parse(&["anki-atlas", "index", "-f"]);
        if let Commands::Index(args) = cli.command {
            assert!(args.force);
        } else {
            panic!("expected Index command");
        }
    }

    // ---- Search ----

    #[test]
    fn parse_search_with_query() {
        let cli = parse(&["anki-atlas", "search", "rust ownership"]);
        if let Commands::Search(args) = cli.command {
            assert_eq!(args.query, "rust ownership");
            assert_eq!(args.top, 10);
            assert!(args.deck.is_none());
            assert!(args.tag.is_none());
            assert!(!args.semantic);
            assert!(!args.fts);
            assert!(!args.verbose);
        } else {
            panic!("expected Search command");
        }
    }

    #[test]
    fn parse_search_with_all_options() {
        let cli = parse(&[
            "anki-atlas",
            "search",
            "query",
            "--deck",
            "CS",
            "--tag",
            "rust",
            "-n",
            "5",
            "--semantic",
            "--verbose",
        ]);
        if let Commands::Search(args) = cli.command {
            assert_eq!(args.query, "query");
            assert_eq!(args.deck.as_deref(), Some("CS"));
            assert_eq!(args.tag.as_deref(), Some("rust"));
            assert_eq!(args.top, 5);
            assert!(args.semantic);
            assert!(!args.fts);
            assert!(args.verbose);
        } else {
            panic!("expected Search command");
        }
    }

    #[test]
    fn parse_search_requires_query() {
        let result = try_parse(&["anki-atlas", "search"]);
        assert!(result.is_err());
    }

    // ---- Topics ----

    #[test]
    fn parse_topics_defaults() {
        let cli = parse(&["anki-atlas", "topics"]);
        if let Commands::Topics(args) = cli.command {
            assert!(args.file.is_none());
            assert!(!args.label);
            assert!((args.min_confidence - 0.3).abs() < f64::EPSILON);
        } else {
            panic!("expected Topics command");
        }
    }

    #[test]
    fn parse_topics_with_file_and_label() {
        let cli = parse(&[
            "anki-atlas",
            "topics",
            "--file",
            "topics.yml",
            "--label",
            "--min-confidence",
            "0.5",
        ]);
        if let Commands::Topics(args) = cli.command {
            assert_eq!(args.file.unwrap().to_str().unwrap(), "topics.yml");
            assert!(args.label);
            assert!((args.min_confidence - 0.5).abs() < f64::EPSILON);
        } else {
            panic!("expected Topics command");
        }
    }

    // ---- Coverage ----

    #[test]
    fn parse_coverage_with_topic() {
        let cli = parse(&["anki-atlas", "coverage", "programming/python"]);
        if let Commands::Coverage(args) = cli.command {
            assert_eq!(args.topic, "programming/python");
            assert!(!args.no_subtree);
        } else {
            panic!("expected Coverage command");
        }
    }

    #[test]
    fn parse_coverage_requires_topic() {
        let result = try_parse(&["anki-atlas", "coverage"]);
        assert!(result.is_err());
    }

    // ---- Gaps ----

    #[test]
    fn parse_gaps_with_topic() {
        let cli = parse(&["anki-atlas", "gaps", "math"]);
        if let Commands::Gaps(args) = cli.command {
            assert_eq!(args.topic, "math");
            assert_eq!(args.min_coverage, 1);
        } else {
            panic!("expected Gaps command");
        }
    }

    #[test]
    fn parse_gaps_with_min_coverage() {
        let cli = parse(&["anki-atlas", "gaps", "math", "--min-coverage", "3"]);
        if let Commands::Gaps(args) = cli.command {
            assert_eq!(args.min_coverage, 3);
        } else {
            panic!("expected Gaps command");
        }
    }

    #[test]
    fn parse_gaps_requires_topic() {
        let result = try_parse(&["anki-atlas", "gaps"]);
        assert!(result.is_err());
    }

    // ---- Duplicates ----

    #[test]
    fn parse_duplicates_defaults() {
        let cli = parse(&["anki-atlas", "duplicates"]);
        if let Commands::Duplicates(args) = cli.command {
            assert!((args.threshold - 0.92).abs() < f64::EPSILON);
            assert_eq!(args.max, 50);
            assert!(args.deck.is_none());
            assert!(args.tag.is_none());
            assert!(!args.verbose);
        } else {
            panic!("expected Duplicates command");
        }
    }

    #[test]
    fn parse_duplicates_with_all_options() {
        let cli = parse(&[
            "anki-atlas",
            "duplicates",
            "--threshold",
            "0.85",
            "-n",
            "20",
            "--deck",
            "CS",
            "--tag",
            "rust",
            "--verbose",
        ]);
        if let Commands::Duplicates(args) = cli.command {
            assert!((args.threshold - 0.85).abs() < f64::EPSILON);
            assert_eq!(args.max, 20);
            assert_eq!(args.deck.as_deref(), Some("CS"));
            assert_eq!(args.tag.as_deref(), Some("rust"));
            assert!(args.verbose);
        } else {
            panic!("expected Duplicates command");
        }
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
        let result = try_parse(&["anki-atlas", "search", "--help"]);
        assert!(result.is_err());
    }
}
