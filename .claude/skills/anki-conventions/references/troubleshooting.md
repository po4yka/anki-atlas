# Troubleshooting Reference

Common issues and solutions when working with Anki via AnkiConnect and MCP.

## Connection Issues

### AnkiConnect Not Responding

**Symptoms:**
- "Connection refused" errors
- MCP tools fail silently
- `curl localhost:8765` returns nothing

**Solutions:**

1. **Check Anki is running:**
   - AnkiConnect only works when Anki desktop is open
   - Must be the main Anki window, not just system tray

2. **Verify AnkiConnect is installed:**
   - Tools > Add-ons
   - Look for code `2055492159`
   - If missing: Tools > Add-ons > Get Add-ons > Enter code

3. **Check port availability:**
   ```bash
   curl -s http://localhost:8765 -d '{"action":"version","version":6}'
   ```
   Should return: `{"result":6,"error":null}`

4. **Restart Anki:**
   - Some add-on updates require restart
   - Close completely and reopen

### MCP Server Not Starting

**Symptoms:**
- Claude Code doesn't show anki tools
- "spawn npx ENOENT" errors

**Solutions:**

1. **Verify Node.js version:**
   ```bash
   node --version
   ```
   Requires 20.19.0+ or 22.12.0+ (not 21.x)

2. **Check MCP configuration:**
   In `~/.claude/settings.local.json`:
   ```json
   "anki": {
     "command": "npx",
     "args": ["-y", "@ankimcp/anki-mcp-server"]
   }
   ```

3. **Test MCP server directly:**
   ```bash
   npx @ankimcp/anki-mcp-server --stdio
   ```
   Should start without errors

### CORS Errors (Obsidian/Browser Integration)

**Solutions:**

Add app origin to AnkiConnect config:

1. Tools > Add-ons > AnkiConnect > Config
2. Add to `webCorsOriginList`:
   ```json
   {
     "webCorsOriginList": [
       "http://localhost",
       "app://obsidian.md"
     ]
   }
   ```
3. Restart Anki

## Card Creation Issues

### Duplicate Detection

**Symptoms:**
- Card not created
- "Cannot create note because it is a duplicate" error

**Solutions:**

1. **Check existing cards:**
   ```
   front:"exact question text"
   ```

2. **Use different deck:**
   Duplicates are checked within deck scope by default

3. **Force creation (if intentional):**
   Set `allowDuplicate: true` in addNote params

### Missing Note Type

**Solutions:**

1. **List available note types:**
   Use `mcp__anki__modelNames`

2. **Check exact spelling:**
   Note type names are case-sensitive:
   - "Basic" (correct)
   - "basic" (wrong)

### Field Mismatch

**Solutions:**

1. **Check field names:**
   Use `mcp__anki__modelFieldNames` with the note type

2. **Common field names:**
   | Note Type | Fields |
   |-----------|--------|
   | Basic | Front, Back |
   | Basic (and reversed card) | Front, Back |
   | Cloze | Text, Extra |

### Update Silently Fails

- **Close card in browser:** Updates fail if note is viewed in Anki's browser window
- Re-fetch note after update to verify changes

## Sync Issues

### AnkiWeb Sync Failures

**Solutions:**

1. **Manual sync first:**
   In Anki: File > Sync (or press Y)

2. **Check AnkiWeb credentials:**
   Tools > Preferences > Syncing

3. **Resolve conflicts:**
   - "Upload to AnkiWeb" overwrites server
   - "Download from AnkiWeb" overwrites local
   - Choose based on which has correct data

### MCP Sync Command

```
mcp__anki__sync
```

If this fails, try manual sync in Anki first.

## FSRS Issues

### Optimization Fails

**Solutions:**

1. **Need sufficient data:**
   - Requires 400+ reviews for meaningful optimization
   - New users: use defaults until you have history

2. **Check Anki version:**
   - FSRS-5: Anki 24.11+
   - FSRS-6: Anki 25.02+

### Intervals Seem Wrong

**Solutions:**

1. **Wait for calibration:**
   - FSRS needs 2-4 weeks to learn your patterns
   - Don't panic about early intervals

2. **Check button usage:**
   - "Hard" means "recalled with difficulty"
   - Use "Again" when you forgot, not "Hard"

3. **Re-optimize:**
   - Deck Options > FSRS > Optimize
   - Do this monthly

## Performance Issues

### MCP Timeouts

**Solutions:**

1. **Break into batches:**
   Instead of adding 100 cards at once, add 20 at a time

2. **Use `multi` action:**
   Batch multiple small operations

## Quick Diagnostic Commands

```bash
# Check AnkiConnect is running
curl -s localhost:8765 -d '{"action":"version","version":6}'

# List decks
curl -s localhost:8765 -d '{"action":"deckNames","version":6}'

# List note types
curl -s localhost:8765 -d '{"action":"modelNames","version":6}'

# Count cards
curl -s localhost:8765 -d '{"action":"findCards","version":6,"params":{"query":"deck:Default"}}'
```

## Getting Help

1. **Anki Manual:** https://docs.ankiweb.net/
2. **AnkiConnect Docs:** https://git.sr.ht/~foosoft/anki-connect
3. **Reddit:** r/Anki for community help
