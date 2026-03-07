# Campaign Configuration

This directory contains configuration files for card generation campaigns.

## Structure

```
config/campaigns/
├── README.md           # This file
├── template.yaml       # Template for new campaigns
├── kotlin.yaml         # Kotlin campaign (COMPLETE)
├── android.yaml        # Android campaign (COMPLETE)
├── compsci.yaml        # CompSci campaign (COMPLETE)
├── algorithms.yaml     # Algorithms campaign (COMPLETE)
├── system-design.yaml  # System Design campaign (COMPLETE)
└── backend.yaml        # Backend campaign (COMPLETE)
```

## Starting a New Campaign

1. **Copy the template:**
   ```bash
   cp config/campaigns/template.yaml config/campaigns/{topic}.yaml
   ```

2. **Edit the config file** with your campaign details

3. **Generate cards:**
   ```bash
   uv run anki-atlas generate <source_path>
   ```

4. **Sync to Anki:**
   ```bash
   uv run anki-atlas sync
   ```

## Config File Format

```yaml
name: "Campaign Name"
source_path: "~/Documents/InterviewQuestions/XX-Folder/"
deck_name: "Topic::Interview"
note_count: 0
tag_prefix: "cs_"
primary_tag: "cs_topic"
completed: false
stats:
  notes_processed: 0
  cards_created: 0
  cards_synced: 0
```

## Field Reference

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Human-readable campaign name | "Kotlin Coroutines" |
| `source_path` | Path to Obsidian notes folder | "~/Documents/InterviewQuestions/70-Kotlin/" |
| `deck_name` | Target Anki deck name | "Kotlin::Interview" |
| `note_count` | Number of notes to process | 358 |
| `tag_prefix` | Tag prefix for this domain | "kotlin_" |
| `primary_tag` | Main tag for cards | "kotlin_coroutines" |
| `completed` | Whether campaign is finished | true/false |
| `stats` | Processing statistics | See below |

## Stats Fields

| Field | Description |
|-------|-------------|
| `notes_processed` | Number of notes with cards |
| `cards_created` | Total cards in registry |
| `cards_synced` | Cards synced to Anki |
