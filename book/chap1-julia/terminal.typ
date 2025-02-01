#import "../book.typ": book-page
#import "@preview/cetz:0.2.2": *

#show: book-page.with(title: "Get a Terminal!")

= Get a Terminal!

A terminal provides direct access to your computer's operating system and is essential for efficient programming and system administration.
In the following, we will introduce how to get a terminal, how to edit files with the terminal (Vim), and how to use the terminal to interact with remote machines (SSH).

#align(center, canvas(length: 0.6cm, {
  import draw: *
  content((0, 0), align(center, box(stroke: black, inset: 10pt, width: 300pt)[Terminal: The interface to control a Linux machine \ #box(stroke: black, inset: 10pt)[Vim: an editor in terminal environment]]), name: "terminal")

  content((0, -5), box(stroke: black, inset: 10pt, width: 300pt)[SSH: A tool to access remote machines from anywhere], name: "ssh")
  line("terminal", "ssh", stroke: black, mark: (start: "straight"))
}))


== Linux Operating System

Using Linux or macOS provides the most straightforward way to get a terminal. Linux is a free, #link("https://opensource.com/resources/what-open-source")[open source] operating system widely used on clusters and excellent for automation. It powers many platforms, including Android. There are two key concepts to understand:

- The *Linux kernel*, started by #link("https://en.wikipedia.org/wiki/Linus_Torvalds")[Linus Torvalds] in 1991, forms the core of the system
- A *Linux distribution* is an #link("https://en.wikipedia.org/wiki/Operating_system")[operating system] built from software that includes the #link("https://en.wikipedia.org/wiki/Linux_kernel")[Linux kernel] and typically a #link("https://en.wikipedia.org/wiki/Package_management_system")[package management system]

This course uses #link("https://ubuntu.com/desktop")[Ubuntu]. Windows users can access a Linux terminal through #link("https://docs.microsoft.com/en-us/windows/wsl/install")[Windows Subsystem for Linux] (WSL).

== Shell (or Terminal)

While Linux distributions offer a *graphical user interface* (GUI), the *command line interface* (CLI) or shell provides more power and efficiency for many tasks.

The shell interprets keyboard commands and passes them to the operating system. Two popular shell interpreters are:

- *Bash*: The default shell on most Linux distributions
- *Zsh*: An enhanced shell (often used with #link("https://github.com/ohmyzsh/ohmyzsh")[oh-my-zsh]) offering features like spelling correction, advanced tab-completion, plugins and themes

On Ubuntu, press `Ctrl + Alt + T` to open a shell. Essential keyboard shortcuts include:
- `man command_name`: Access command documentation
- `CTRL-C`: Interrupt a running program
- `CTRL-D`: Exit the shell

For deeper understanding, explore:
- #link("https://missing.csail.mit.edu/2020/shell-tools/")[MIT Open course: Missing semester]
- #link("https://learn.microsoft.com/en-us/training/paths/shell/")[Get started with the Linux command line and the Shell]

Common shell commands you'll use frequently:

```
man     # an interface to the system reference manuals

ls      # list directory contents
cd      # change directory
mkdir   # make directories
rm      # remove files or directories
pwd     # print name of current/working directory

echo    # display a line of text
cat     # concatenate files and print on the standard output

alias   # create an alias for a command

lscpu   # display information about the CPU architecture
lsmem   # list the ranges of available memory with their online status

top     # display Linux processes
ssh     # the OpenSSH remote login client
vim     # Vi IMproved, a programmer's text editor
git     # the stupid content tracker

tar     # an archiving utility
```
== Editor in terminal - Vim

To edit files in the terminal, you can use Vim - the default text editor in most Linux distributions.
Vim has three primary modes, each tailored for specific tasks:

- *Normal Mode*: Navigate through the file and perform tasks like deleting lines or copying text. Enter by pressing `ESC`
- *Insert Mode*: Insert text as in conventional text editors. Enter by typing `i` in normal mode
- *Command Mode*: Input commands for tasks like saving files or searching. Enter by typing `:` in normal mode

Here are some essential Vim commands to get started:
```
i       # input
:w      # write
:q      # quit
:q!     # force quit without saving

u       # undo
CTRL-R  # redo
```

All commands must be executed in *normal mode* (press `ESC` to enter). For more advanced Vim techniques, see #link("https://missing.csail.mit.edu/2020/editors/")[this lecture].

== Connect to Remote Systems with SSH

The Secure Shell (SSH) protocol enables secure remote command execution over unsecured networks. Using cryptography for authentication and encryption, SSH allows you to:
- Push code to remote git repositories
- Access and control remote machines

To connect to a remote system like a university cluster, you'll need:
1. The hostname or IP address
2. Your username
3. Authentication credentials

The basic connection syntax is:
```bash
ssh username@hostname
```

For example, to connect to a cluster with the username `user` and hostname `cluster.example.com`, you would use:
```bash
ssh user@cluster.example.com
```


== Streamlining SSH Access

To avoid repeatedly typing hostnames and usernames, you can configure SSH using the `~/.ssh/config` file. This configuration file allows you to create aliases and set default parameters for your SSH connections.

Here's an example configuration:
```
Host amat5315
  HostName <hostname>
  User <username>
```

In this example, `amat5315` serves as an alias for the remote host. Once you've configured the `~/.ssh/config` file, you can connect to the remote machine simply by typing:
```bash
ssh amat5315
```

To avoid typing the password everytime you login, you can use the command 
```bash
ssh-keygen
```
to generate a pair of public and private keys, which will be stored in the `~/.ssh` folder on the local machine.
After setting up the keys, you can copy the public key to the remote machine by typing
```bash
ssh-copy-id amat5315
```
Try connecting to the remote machine again, and you will notice that entering the password is no longer necessary.

== How does an SSH key pair work?

The SSH key pair consists of two asymmetric keys: a public key (or lock) and a private key. In the example above, the public key is uploaded to the remote machine, while the private key remains securely stored on your local machine. The public key can be shared freely, but the private key must remain confidential.

#align(center, canvas(length: 0.6cm, {
  import draw: *
  circle((0, 6), radius: (6.5, 1), name: "remote")
  circle((0, 0), radius: (6.5, 1), name: "local")
  content("remote", [Remote (\u{1F512})])
  content("local", [Local (\u{1F511})])
  line((name: "remote", anchor: 240deg), (name: "local", anchor: 120deg), stroke: black, mark: (start: "straight"), name: "first")
  line((name: "remote", anchor: 270deg), (name: "local", anchor: 90deg), stroke: black, mark: (end: "straight"), name: "second")
  line((name: "remote", anchor: 300deg), (name: "local", anchor: 60deg), stroke: black, mark: (start: "straight"), name: "third")
  content("first.mid", box(fill: white, inset: 5pt)[Connect])
  content("second.mid", box(fill: white, inset: 5pt)[Encrypted\ Message])
  content("third.mid", box(fill: white, inset: 5pt)[Decrypted\ Message])
}))

When connecting to a server, the server needs to verify your identity. It does this by checking if you possess the private key that matches the public key stored on the server. If you have the correct private key, access is granted.

The core principle of the SSH key pair is that the *public key can encrypt a message that only the private key can decrypt*. Think of the public key as a lock and the private key as the key to unlock it. This forms the basis of the SSH protocol. The server can send you a message encrypted with your public key, and only you can decrypt it with your private key. This ensures the server knows you have the private key without needing to send it.

== Practice

In this example, we will demonstrate how to use the `ssh` command to connect to a remote machine named `gpu` and perform some basic operations. If you do not have access to a remote machine, you can perform these operations on your local machine.
```bash
(base) ➜  ~ ssh gpu
Welcome to Ubuntu 22.04.3 LTS (GNU/Linux 5.15.0-91-generic x86_64)
...
*** System restart required ***
Last login: Tue Mar  5 06:20:05 2024 from 10.13.139.204
(base) ➜  ~
```

Then we switch to the `jcode` directory and create a directory `test` and a file `README.md` in the directory.

```bash
(base) ➜  ~ ls     # list directory contents
ClusterManagement                 jcode       packages
ScientificComputingForPhysicists  miniconda3  software
(base) ➜  ~ cd jcode   # change directory
(base) ➜  jcode mkdir test # make directories
(base) ➜  jcode cd test # change directory
(base) ➜  test vim README.md # create a file and edit it
```

You will see the following screen after typing `vim README.md`.
#figure(image("images/vim.png", width: 80%))

Type `i` to enter insert mode and add some text, for example, `# Read me!`. Then press `ESC` to switch to normal mode and type `:wq` to save and exit the file.

After returning to the terminal, type `ls -l` to verify the file you just created.
```bash
total 4
-rw-rw-r-- 1 jinguoliu jinguoliu 11 Mar  5 06:30 README.md
```

You can also use the `cat` command to check the content of the file.

```bash
(base) ➜  test cat README.md
# Read me!
```

To exit the shell, simply press `CTRL-D`.
Enjoy your journey in mastering the terminal!