# Claude runtime for Hermes: testing checklist

## What this is

This is a new runtime for Hermes. When it is on, Hermes hands each turn to Claude
Code (Anthropic's coding agent) instead of the usual model. It runs on your own
`claude` login, so there is no extra API cost. You still talk to Hermes the same
way (CLI or Discord), and you still have Hermes' tools (web, browser, cron,
memory, kanban, and so on). Claude is just the brain.

We want to test it well before sending it upstream to the Hermes project. Please
go through the list below and tick what works. Note anything that looks wrong.

## Before you start

1. Make sure the `claude` CLI is logged in. Run `claude` once and use `/login` if
   needed.
2. Turn the runtime on:
   - `/claude-runtime on` (turns it on)
   - `/claude-runtime` (shows the current state)
   - `/claude-runtime off` (turns it back to normal Hermes)
3. After turning it on, start a fresh session: `/new` (or `/reset`).
4. You can test from the CLI (`hermes` or `hermes chat`) and from Discord. Tags
   below say where each test matters:
   - **[Both]** test in CLI and Discord
   - **[Discord]** test in Discord
   - **[CLI]** test in the CLI

## Checklist

### 1. Turn it on and off
- [ ] `/claude-runtime on` reports it switched on. **[Both]**
- [ ] `/claude-runtime off` switches back, and your normal model/provider returns
  (check with `/status` or `/model`). **[Both]**
- [ ] The banner shows `Runtime: claude code` after turning it on. **[CLI]**

### 2. Does it know it is Hermes
- [ ] Ask: "Who are you, and how am I talking to you right now?" It should say it
  is Claude Code running as the engine behind Hermes, reached through a chat app
  (Discord) or the terminal. It should NOT say it is just a plain Claude Code
  session. **[Both]**
- [ ] Ask: "What is the difference between Hermes cron jobs and Claude routines?"
  It should explain Hermes cron is the real one here, and not point you at Claude
  routines or a `/schedule` skill. **[Both]**

### 3. Basic chat and coding
- [ ] Ask a normal question and get a sensible answer. **[Both]**
- [ ] Ask it to write a small script and run it, e.g. "write a Python script that
  prints the first 10 prime numbers and run it." **[Both]**
- [ ] Ask it to read or edit a file in the working folder. **[CLI]**

### 4. Hermes cron (the main thing we fixed)
- [ ] "List all my cron jobs, enabled and disabled." It should use the Hermes cron
  tool and show your real jobs (or say there are none). **[Both]**
- [ ] "Create a cron job that sends me a good morning message every day at 9am."
  Then "list cron jobs" to confirm it was created as a Hermes job. **[Discord]**
  (Test in Discord so you can check the job is set to post back to the right
  channel.)
- [ ] Remove the test job: "remove that cron job you just made." Confirm it is
  gone. **[Discord]**
- [ ] Confirm it did NOT create a Claude routine or use a `/schedule` skill. **[Both]**

### 5. Hermes tools
- [ ] Web search: "search the web for the latest news about X." **[Both]**
- [ ] Browser: "open example.com and tell me the page title." **[Both]**
- [ ] Image generation: "generate an image of a cat riding a bicycle." **[Both]**
- [ ] Vision: send an image and ask "what is in this image?" **[Discord]**
- [ ] Video analyze (new): give a video URL and ask "what happens in this video?"
  if you have one. **[Both]**
- [ ] Video generate (new): "generate a short video of waves on a beach." Only
  works if a video backend is set up; otherwise it should say so clearly. **[Both]**
- [ ] Hermes skills: "list my Hermes skills." **[Both]**
- [ ] Create a Hermes skill (new): "create a simple Hermes skill called hello that
  greets the user." Then list skills to confirm. **[Both]**
- [ ] Kanban, if you use it: "show my kanban board." **[Both]**

### 6. Is it honest about its limits
- [ ] "Send a message to my other Discord channel." It should say it cannot send
  to other chats from this runtime, only reply here. **[Discord]**
- [ ] "Schedule a reminder for me in one hour." It should use Hermes cron, not its
  own wake-up or routine tools. **[Both]**

### 7. Approvals (when it wants to run something risky)
- [ ] Ask it to run a command that needs approval (depends on your security mode),
  for example "delete all files in /tmp/test." You should get an approval prompt;
  use `/approve` or `/deny`. **[CLI]**
  (Note: approval prompts are wired for the CLI. In Discord the approval flow is
  not fully wired yet, so test approvals mainly in the CLI.)

### 8. Stop a running task
- [ ] Start a long task, e.g. "count slowly from 1 to 50 with a short pause each
  number," then send `/stop`. It should stop. **[Both]**
  - In Discord, `/stop` interrupts the running turn.
  - In the CLI, if the model put the task in the background (it will say "running
    in background"), `/stop` retires the Claude session to stop that background
    work, then the chat continues on your next message (it resumes from disk).
    To interrupt a foreground turn in the CLI, press Enter/Esc.

### 9. Multi-turn memory within a session
- [ ] Tell it a fact ("my favorite color is green"), then in the next message ask
  "what is my favorite color?" It should remember within the same session. **[Both]**
- [ ] Continuity across a restart: tell it a fact, restart the gateway
  (`sudo systemctl restart hermes-gateway.service`), then in the same Discord
  thread ask for the fact back. It should still remember and not say it is
  starting fresh. The runtime resumes the saved Claude session from disk
  (`~/.hermes/claude_runtime_sessions.json` maps each thread to its Claude
  session id). **[Discord]**

### 10. Hermes memory across sessions (new: it now reads and writes)
This is the main new thing. On this runtime Claude can now read your saved Hermes
memory and write to it, so facts last across sessions.
- [ ] Ask "what do you already know about me?" It should read your saved Hermes
  memory (if you have any). If you have none yet, that is fine. **[Both]**
- [ ] Tell it something durable: "remember that I prefer short answers." It should
  save it with the Hermes memory tool and can say it saved it. **[Both]**
- [ ] Start a fresh session (`/new`, or a new thread in Discord), then ask "how do
  I like my answers?" It should recall the saved fact. This is the real test: it
  wrote to Hermes memory and read it back in a new session. **[Both]**
- [ ] (Optional) Check the file yourself: the fact should show up in
  `~/.hermes/memories/USER.md` or `~/.hermes/memories/MEMORY.md`. **[CLI]**
- [ ] It should NOT save this into Claude Code's own memory or a `CLAUDE.md` file.
  Hermes memory is the files above. **[Both]**
- [ ] After a longer chat (about 10 turns), it may save useful facts on its own,
  without being asked. That is the periodic memory check. Optional to verify. **[Both]**

### 11. Cost and usage
- [ ] Check `/usage` or `/status` after a few turns. Cost should show as included
  (your Claude subscription), not a surprising dollar charge. **[Both]**

### 12. Settings surface (optional, for admins)
- [ ] By default, ask "do you have a Gmail tool, or a /schedule skill?" It should
  NOT have your personal Gmail/Calendar/Drive tools or your personal Claude Code
  skills. **[Both]**
- [ ] To allow them, set `model.claude_runtime_inherit_user_config: true` in
  `~/.hermes/config.yaml`, restart, and check they come back. Set it back to
  `false` after. **[CLI]**
- [ ] To add a tool (like Gmail) the proper way, put the MCP server in Hermes'
  own `mcp_servers` config so it works on every runtime. **[CLI]**

## Known limits (please do not report these as bugs)

- Hermes memory now works here: it reads your saved memory and can write to it.
  But delegate_task and session search still do not run on this runtime. There is
  no separate automatic background memory review; the model saves durable facts as
  it goes, or when the periodic memory check reminds it.
- It cannot send messages to other chats from this runtime. Its reply goes to the
  current chat only.
- During a turn there may be no live tool-by-tool progress; the full answer
  arrives at the end.
- Approval prompts in Discord are not fully wired yet (use the CLI for approval
  testing).

## How to report a problem

For anything that looks wrong, note:
1. Where (CLI or Discord), and the exact prompt you sent.
2. What it did vs what you expected.
3. Whether `/claude-runtime` was on, and which model (`/status`).

Then turn the runtime off with `/claude-runtime off` if you want to go back to
normal Hermes.
