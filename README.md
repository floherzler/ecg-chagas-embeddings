## Setup

![Flow diagram](diagrams/flow.svg)

### Prerequisites
- A Debian/Ubuntu-like system with sudo and curl available.
- Basic build tools (installed below).

### Initialize and lock project
- Initialize project environment:
    - `uv init`
- Lock dependencies:
    - `uv lock`

### Install d2
Run the following to install build tools and d2:
```
sudo apt update
sudo apt install -y build-essential
curl -fsSL https://d2lang.com/install.sh | sh -s --
```

### Verify
- Check d2 is installed: `d2 --version`
- Confirm project environment: rerun `uv init` or `uv lock` as needed.

Notes:
- The diagram above is located at `diagrams/flow.svg`.
- Run the install commands with appropriate permissions (sudo) or in your CI environment.