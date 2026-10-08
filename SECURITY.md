# Security policy

Security fixes are made on the latest `main` branch. Review changes and update before using the agent with sensitive workspaces.

Report vulnerabilities privately using the repository's GitHub **Security → Report a vulnerability** feature when available. If private reporting is unavailable, open an issue requesting a private contact without including exploit details or credentials. Do not post API keys, private file contents, or sensitive logs in public issues.

Filesystem tools enforce workspace containment. Approved shell commands can access resources available to the process account; the shell is not an operating-system sandbox. Use a restricted account or container when running untrusted projects.

Prefer environment variables for credentials. Explicitly saved credentials are plaintext: configuration writes are atomic with owner-only permissions on POSIX; on Windows the file inherits the destination directory's ACL. Store it in a directory accessible only to your account. MCP credentials and server configuration require the same protection.
