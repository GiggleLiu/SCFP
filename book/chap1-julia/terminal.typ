#import "../book.typ": book-page
#import "@preview/cetz:0.4.1": *

#show: book-page.with(title: "Terminal Environment")

#align(center, [= Terminal Environment\
_Jin-Guo Liu_])

The terminal is a text-based interface that provides direct access to your operating system, making it an essential tool for efficient programming, system administration, and scientific computing. Unlike graphical interfaces, the terminal allows you to execute commands with precision and automate complex tasks through scripting.

This chapter covers three fundamental aspects of terminal usage:
1. *Getting started with terminals* - Setting up and navigating the command-line environment
2. *Text editing with Vim* - Mastering the powerful terminal-based editor
3. *Remote access with SSH* - Connecting to and working with remote machines securely

These skills form the foundation for modern computational work, enabling you to work efficiently on local machines, high-performance clusters, and cloud computing environments.

#align(center, canvas(length: 0.8cm, {
  import draw: *
  let boxed(it, width: auto) = box(stroke: black, inset: 10pt, width: width, text(12pt, it))
  content((0, 0), align(center, boxed(width: 300pt)[_Terminal_: The interface to control a Linux machine \ #boxed[_Vim_: an editor in terminal environment]]), name: "terminal")

  content((0, -5), boxed(width: 300pt)[_SSH_: A tool to access remote machines from anywhere], name: "ssh")
  line("terminal", "ssh", stroke: black, mark: (start: "straight"))
}))


== Linux Operating System

Linux and macOS provide native terminal access, making them ideal environments for computational work. Linux, being a free and open-source operating system, has become the standard platform for high-performance computing clusters, supercomputers, and scientific computing environments worldwide.

For Windows users, Microsoft provides excellent Linux compatibility through the #link("https://docs.microsoft.com/en-us/windows/wsl/install")[Windows Subsystem for Linux] (WSL). This allows you to run a complete Linux environment directly on Windows, giving you access to all the powerful command-line tools and workflows used in scientific computing.

Installing WSL is straightforward - simply open PowerShell as an administrator and run:
```bash
wsl --install
```

== Shell Environment

The *shell* is a command-line interpreter that processes your keyboard input and executes commands on the operating system. It serves as the bridge between you and the system's core functionality. The most common shell is #link("https://en.wikipedia.org/wiki/Bash_(Unix_shell)")[Bash] (Bourne Again Shell), which is the default on most Linux distributions and macOS.

For enhanced productivity, many users prefer #link("https://en.wikipedia.org/wiki/Z_shell")[Zsh] (Z Shell), especially when paired with #link("https://github.com/ohmyzsh/ohmyzsh")[oh-my-zsh]. Zsh provides advanced features including:
- Intelligent tab completion
- Spell correction
- Extensive plugin ecosystem
- Customizable themes and prompts

=== Essential Terminal Controls

Before diving into commands, familiarize yourself with these fundamental keyboard shortcuts:

*Getting Help:*
- `man <command_name>`: Access detailed command documentation
- `<command> --help`: Quick help for most commands

*Process Control:*
- `CTRL-C`: Interrupt/stop a running program
- `CTRL-D`: Exit the shell or end input
- `CTRL-Z`: Suspend a running program

*Navigation:*
- `Up/Down arrows`: Navigate through command history
- `TAB`: Auto-complete commands and file paths
- `CTRL-L`: Clear the screen

=== Common Terminal Operations

*File and Directory Navigation:*
```bash
$  ls                    # list directory contents
$  ls -la               # detailed listing with hidden files
$  cd book              # change to 'book' directory
$  cd ..                # move up one directory
$  pwd                  # show current directory path
/Users/liujinguo/Documents/SCFP/book
$  mkdir new_folder     # create a new directory
$  rmdir empty_folder   # remove empty directory
$  rm -r folder_name    # remove directory and all contents
```

*Text File Operations:*
```bash
$  echo "Hello, world!" > hello.txt    # create/write to a file
$  cat hello.txt                       # display file content
Hello, world!
$  head -5 file.txt                    # show first 5 lines
$  tail -5 file.txt                    # show last 5 lines
$  grep "pattern" file.txt             # search for text pattern
```

*Creating Shortcuts (Aliases):*
```bash
$  alias ll="ls -la"                   # create alias for detailed listing
$  alias ..="cd .."                    # quick navigation shortcut
$  ll                                  # use the alias
total 312
-rw-r--r--@  1 user  staff    21K Feb  1 23:02 git.typ
drwxr-xr-x@ 10 user  staff   320B Feb 10 08:07 images
```

*System Monitoring:*
```bash
$  top                                 # display running processes
$  htop                                # enhanced process viewer (if installed)
$  df -h                               # disk usage in human-readable format
$  free -h                             # memory usage information
$  lscpu                               # CPU architecture details
```

*Essential Tools:*
```bash
$  vim filename.txt                    # powerful text editor
$  ssh user@remote.com                 # secure remote access
$  git status                          # version control
$  tar -xzf archive.tar.gz             # extract compressed files
```

= Vim - A Powerful Terminal Text Editor

Vim (Vi IMproved) is a highly efficient, modal text editor that's ubiquitous in Unix-like systems. While it has a steep learning curve initially, mastering Vim significantly boosts your productivity, especially when working on remote systems where graphical editors aren't available.

== Understanding Vim's Modal Design

Vim operates in distinct modes, each optimized for specific tasks:

*Normal Mode* (Default):
- Navigate through text and execute commands
- Copy, cut, paste, and delete operations
- Enter by pressing `ESC` from any other mode

*Insert Mode*:
- Type and edit text like a conventional editor
- Enter by pressing `i`, `a`, `o`, or `O` from normal mode

*Command Mode*:
- Execute file operations and advanced commands
- Search and replace operations
- Enter by typing `:` from normal mode

== Essential Vim Commands

*Getting Started:*
```bash
$  vim filename.txt         # open or create a file
$  vim +42 filename.txt     # open file at line 42
```

*Basic Operations* (in Normal Mode):
```vim
i           # enter insert mode at cursor
a           # enter insert mode after cursor
o           # create new line below and enter insert mode
ESC         # return to normal mode

:w          # save file
:q          # quit vim
:wq         # save and quit
:q!         # quit without saving

u           # undo last change
CTRL-R      # redo last undone change
```

*Navigation* (in Normal Mode):
```vim
h j k l     # move left, down, up, right
w           # jump to beginning of next word
b           # jump to beginning of previous word
G           # go to end of file
gg          # go to beginning of file
:42         # go to line 42
```

*Quick Tip:* Start with these basics and gradually learn more commands. Vim's efficiency comes from combining simple commands to perform complex operations.

= SSH - Secure Remote Access

SSH (Secure Shell) is a network protocol that provides secure, encrypted communication between your local machine and remote servers. It's essential for scientific computing, allowing you to:
- Access high-performance computing clusters
- Work on remote servers and cloud instances
- Securely transfer files between machines
- Manage remote systems and services

== Basic SSH Connection

To establish an SSH connection, you need:
1. *Hostname* or IP address of the remote machine
2. *Username* on the remote system
3. *Authentication* method (password or SSH key)

*Basic syntax:*
```bash
ssh username@hostname
```

*Examples:*
```bash
# Connect to a university cluster
ssh student@cluster.university.edu

# Connect using IP address
ssh user@192.168.1.100

# Connect on specific port (if not default port 22)
ssh -p 2222 user@server.com
```

*First Connection:*
When connecting for the first time, you'll see a fingerprint verification message. Type `yes` to continue:
```bash
The authenticity of host 'server.com' can't be established.
Are you sure you want to continue connecting (yes/no)? yes
```


== Streamlining SSH Access

=== SSH Configuration File

Create a `~/.ssh/config` file to simplify connections and set default parameters:

```bash
# Example SSH config file (~/.ssh/config)
Host myserver
    HostName server.university.edu
    User studentname
    Port 22
    
Host cluster
    HostName hpc.cluster.edu
    User research_id
    ForwardX11 yes          # Enable GUI forwarding

Host github
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_key
```

*Usage:*
```bash
ssh myserver              # Instead of: ssh studentname@server.university.edu
ssh cluster               # Connect to HPC cluster with GUI support
```

=== Password-Free Authentication

SSH keys provide secure, password-free authentication:

*Step 1: Generate SSH key pair*
```bash
ssh-keygen -t rsa -b 4096 -C "your.email@example.com"
# Press Enter to accept default location (~/.ssh/id_rsa)
# Optionally set a passphrase for extra security
```

*Step 2: Copy public key to remote server*
```bash
ssh-copy-id myserver      # Using config alias
# OR
ssh-copy-id user@hostname # Direct connection
```

*Step 3: Test the connection*
```bash
ssh myserver              # Should connect without password prompt
```

*Bonus: Managing multiple keys*
```bash
ssh-keygen -t rsa -f ~/.ssh/work_key     # Create specific key for work
ssh-keygen -t rsa -f ~/.ssh/personal_key # Separate key for personal use
```

== Understanding SSH Key Cryptography

SSH uses *public-key cryptography* (also called asymmetric cryptography) for secure authentication. This system involves two mathematically related keys:

- *Public Key*: Can be shared freely and is stored on remote servers
- *Private Key*: Must be kept secure on your local machine

*The Security Principle:*
Data encrypted with the public key can only be decrypted with the corresponding private key. This allows servers to verify your identity without you having to send your password over the network.

#align(center, canvas(length: 0.8cm, {
  import draw: *
  let s(it) = text(12pt, it)
  circle((0, 6), radius: (5.5, 1), name: "remote")
  circle((0, 0), radius: (5.5, 1), name: "local")
  content("remote", s[Remote (\u{1F512})])
  content("local", s[Local (\u{1F511})])
  line((name: "remote", anchor: 240deg), (name: "local", anchor: 120deg), stroke: black, mark: (start: "straight"), name: "first")
  line((name: "remote", anchor: 270deg), (name: "local", anchor: 90deg), stroke: black, mark: (end: "straight"), name: "second")
  line((name: "remote", anchor: 300deg), (name: "local", anchor: 60deg), stroke: black, mark: (start: "straight"), name: "third")
  content("first.mid", box(fill: white, inset: 5pt, s[Connect]))
  content("second.mid", box(fill: white, inset: 5pt, s[Encrypted\ Message]))
  content("third.mid", box(fill: white, inset: 5pt, s[Decrypted\ Message]))
}))

*How SSH Key Authentication Works:*

1. *Setup Phase:* Your public key is installed on the remote server (in `~/.ssh/authorized_keys`)
2. *Connection Phase:* When you attempt to connect, the server challenges you
3. *Verification Phase:* The server sends an encrypted message using your public key
4. *Authentication Phase:* Your SSH client decrypts the message with your private key and responds
5. *Access Granted:* The server verifies the response and grants access

*Security Benefits:*
- *No password transmission* - Your password never travels over the network
- *Unique authentication* - Each key pair is mathematically unique
- *Revocable access* - Remove the public key from the server to revoke access
- *Multiple key support* - Different keys for different purposes (work, personal, etc.)

*Analogy:* Think of the public key as a special lock that you can copy and install anywhere. Only your private key (which stays with you) can unlock these locks. This way, you can prove your identity to any server that has your public key, without exposing your secret.

= Additional Resources

== Online Tutorials and Courses
- #link("https://missing.csail.mit.edu/")[MIT: The Missing Semester of Your CS Education] - Comprehensive course covering terminal, vim, git, and more
- #link("https://www.codecademy.com/learn/learn-the-command-line")[Codecademy: Learn the Command Line] - Interactive command-line tutorial
- #link("https://linuxjourney.com/")[Linux Journey] - Step-by-step guide to learning Linux

== Quick Reference
- #link("https://www.cheatography.com/davechild/cheat-sheets/linux-command-line/")[Linux Command Line Cheat Sheet]
- #link("https://vim.rtorr.com/")[Vim Cheat Sheet]