# Security policy

Please do not publish credentials, private dataset paths, unsafe archive samples, or an
unpatched command-injection issue in a public ticket. Report security-sensitive findings to
the repository owner through GitHub's private security advisory flow.

The current supported line is `0.1.x`. This project processes untrusted dataset archives and
launches external benchmark containers; use a dedicated machine/account, review upstream
images, and never mount secrets or a writable home directory into task containers.
