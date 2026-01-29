#import "@preview/touying:0.6.1": *
#import "@preview/touying-simpl-hkustgz:0.1.2": *
#import "@preview/cetz:0.4.1": *
#import "@preview/cetz-plot:0.1.2": plot
#import "../shared/characters.typ": ina, christina, murphy

#let globalvars = state("t", 0)
#let timecounter(minutes) = [
  #globalvars.update(t => t + minutes)
  #place(dx: 100%, dy: -5%, align(right, text(16pt, red)[#context globalvars.get()min]))
]
#let clip(image, top: 0pt, bottom: 0pt, left: 0pt, right: 0pt) = {
  box(clip: true, image, inset: (top: -top, right: -right, left: -left, bottom: -bottom))
}
#let terminal(code, width: 100%) = {
  box(radius: 4pt, inset: 5pt, fill:black, width: width, text(13pt, fill: silver)[
  #code
  ])
}
#set cite(style: "apa")

#show raw.where(block: true): it=>{
  block(radius:4pt, fill:gray.transparentize(90%), inset:1em, width:99%, text(it))
}

#show: hkustgz-theme.with(
  config-info(
    title: [Terminal, Vim, SSH and Git],
    subtitle: [Basic programming toolchain],
    author: [Jin-Guo Liu],
    date: datetime.today(),
    institution: [HKUST(GZ) - FUNH - Advanced Materials Thrust],
  ),
)

#outline-slide()

== AMAT5315: Modern Scientific Computing
#timecounter(1)

- Lecturer: Jinguo LIU
- Teaching assistant: Huanhai ZHOU

=== How to communicate?
Zulip stream: `AMAT5315-2025Fall` (you will have access soon).

== Books (optional)
#timecounter(1)
- #highlight("Lecture notes"): https://scfp.jinguo-group.science/ (PDF files will be uploaded to zulip as well)
#v(20pt)
- *Matrix Computation*, 4th Edition, Gene H. Golub and Charles F. Van Loan, 2013,
  - Tag: Matrix, tensor, sparse matrices and their operations
- *The nature of computation*, Cristopher Moore and Stephan Mertens, 2011,
  - Tag: Computational theory, probabilistic process

== Lectures: 12 in total
#timecounter(1)
#place(dx: 80%, dy: 25%, align(right, [#text(66pt)[\u{1F4BB}]\ (Required)]))

=== PART 1: High Performance Computing
- Open source toolchain: Terminal, Git and SSH
- The Julia programming language: Basics & Advanced topics (2 lectures)
- Matrix computation: Basic & Advanced topics (2 lectures)
- Sparse matrices, dominant eigenvalues and eigenvectors

=== PART 2: Different Ways to Solve Spin Glass
- The spin glass problem, theoreticaly perspective
- Monte Carlo methods
- Tensor network methods
- Mathematical optimization: linear programming and integer programming
- Branching and bound
- Physics inspired methods

== Assessment
#timecounter(1)
#align(center, box(stroke: black, inset: 10pt)[100% through homework])

- We use the standard code review process (#link("https://github.com/CodingThrust/AMAT5315-2025Fall-Homeworks")[Guide]) in GitHub.

#align(center, canvas({
    import draw: *
    content((0, 0), box(stroke: black, inset: 10pt)[Student submit a PR], name: "submit")
    content((7, 0), box(stroke: black, inset: 10pt)[TA review], name: "review")
    content((14, 0), box(stroke: black, inset: 10pt)[Merge PR], name: "merge")
    line("submit", "review", stroke: black, mark: (end: "straight"))
    line("review", "merge", stroke: black, mark: (end: "straight"), name: "rm")
    line("review", (rel: (0, -2)), (rel: (0, -2), to: "submit"), "submit", stroke: black, mark: (end: "straight"))
    content((3.5, -2.5), "request change")
    content("rm.mid", [$checkmark$])
}))

Note: PR is a pull request, which is a request to merge changes to the _repository_.

== Survey: Let us know you better!
#timecounter(8)
- Name, e.g. Jinguo Liu
- Research label, e.g. Scientific Computing
- Programming Language, e.g. Julia, Python
- What do you expect from this course?

= Terminal Environment

== Why terminal and Git?

- Connect to a high performance computing (HPC) cluster
- Keep your code safe
- Collaborate with others on coding

=== Reference
- The Missing Semester of Your CS Education: https://missing.csail.mit.edu/

== What is a terminal?
#timecounter(1)

A text based window where you can manage your operating system. e.g. manage folders and files:

#terminal(```bash
(base) ➜  AMAT5315-2025Spring-Homeworks git:(main) ls  # show content in a directory
README.md hw1
(base) ➜  AMAT5315-2025Spring-Homeworks git:(main) cd hw1  # change directory
(base) ➜  hw1 git:(main) ls
README.md
```)

== What can you do in a terminal?
#timecounter(2)
- *File Operations*: `ls`, `cd`, `mkdir`, `rm`
- *Text Processing*: `cat`, `echo`, `grep`
- *File Editing*: `vim`
- *System Monitoring*: `top`, `ps`
- *Remote Access*: `ssh`, `scp`

== How to get a terminal?
#timecounter(1)

- If you use Linux (an operating system) or macOS, you already have a terminal.
- If you use Windows, you can install #link("https://docs.microsoft.com/en-us/windows/wsl/install")[Windows Subsystem for Linux] (WSL) to get a Linux terminal. Just type:

  ```
  wsl --install
  ```

== Shell (or Terminal)
#timecounter(1)

In a shell, we use
- `CTRL-C` to stop a running program
- `CTRL-D` to exit a shell.

== Frequently used commands - getting help
#timecounter(1)

```bash
man ls    # an interface to the system reference manuals
```

Input syntax:
```
ls [-@ABCFGHILOPRSTUWabcdefghiklmnopqrstuvwxy1%,] [--color=when] [-D format] [file ...]
```

== Frequently used commands - manipulating directory
#timecounter(3)
```bash
ls      # list directory contents
mkdir   # make directories
cd      # change directory
pwd     # print name of current/working directory
touch   # create an empty new file
rm      # remove files or directories
```
==
```bash
$ ls
images              lammps.typ          refs.bib            slides-software.typ vasp.typ

$ mkdir test
$ ls
images              lammps.typ          refs.bib            slides-software.typ test                vasp.typ

$ cd test
$ pwd
/Users/liujinguo/Documents/MaterialsIntro/lecturenotes/software/test
```

==
```bash
$ touch testfile1
$ touch testfile2
$ ls
testfile1 testfile2
$ rm testfile1
$ ls
testfile2
$ cd ..
$ pwd
/Users/liujinguo/Documents/MaterialsIntro/lecturenotes/software
```
==
```bash
$ rm test
rm: test: is a directory
$ rm -r test
$ ls -l
total 112
drwxr-xr-x@ 6 liujinguo  staff    192 Nov 16 10:06 images
-rw-r--r--@ 1 liujinguo  staff   7814 Nov 14 23:28 lammps.typ
-rw-r--r--@ 1 liujinguo  staff    437 Nov 13 17:05 refs.bib
-rw-r--r--@ 1 liujinguo  staff  31021 Nov 17 20:53 slides-software.typ
-rw-r--r--@ 1 liujinguo  staff  12023 Nov 16 10:06 vasp.typ
```

== Frequently used commands - read & write files
#timecounter(1)
```bash
echo    # display a line of text
cat     # concatenate files and print on the standard output
```

==
```bash
$ echo "hello-world"
hello-world

$ echo "hello-world" > testfile
$ ls
images              lammps.typ          refs.bib            slides-software.typ testfile            vasp.typ
$ cat testfile
hello-world
```

== Remote access - all top 500 clusters use linux!
```
ssh     # the OpenSSH remote login client
```

The Secure Shell (SSH) protocol is a method for securely sending commands to a computer over an unsecured network. SSH uses cryptography to authenticate and encrypt connections between devices.

The basic usage of `ssh` is like:
```bash
ssh <username>@<hostname>
```
where `<username>` is the user's account name and `<hostname>` is the host name or IP of the target machine. You will get logged in after inputting the password.

== Live code

1. Connect to remote
2. Download the lecture notes
3. Compile them to PDF files

== Download results
#timecounter(2)

```bash
scp <username>@<hostname>:<path/to/remote/file> <path/to/local/directory>
```

Similarly, you can upload files to the server with:
```bash
scp <path/to/local/file> <username>@<hostname>:<path/to/remote/directory>
```

== Edit files: Vim
#timecounter(3)
```bash
vim     # Vi IMproved, a terminal based text editor
```

It has three primary modes, each tailored for specific tasks:
- *Normal Mode*, where users can navigate through the file and perform tasks like deleting lines or copying text; One can enter the normal mode by typing `ESC`;
- *Insert Mode*, where users can insert text as in conventional text editors; One can enter the insert mode by typing `i` in the normal mode;
- *Command Mode*, where users input commands for tasks like saving files or searching; One can enter the command mode by typing `:` in the normal mode.

==
A few commands are listed below to get you started with `Vim`.

```
i       # input
:w      # write
:q      # quit
:q!     # force quit without saving

u       # undo
CTRL-R  # redo
```

All the commands must be executed in the *normal mode* (press `ESC` if not).



== Frequently used commands - resource monitoring
```bash
lscpu   # display information about the CPU architecture
lsmem   # list the ranges of available memory with their online status

top     # display Linux processes
```

==

```bash
$ top
```
#align(center, text(9pt)[```
Processes: 481 total, 4 running, 1 stuck, 476 sleeping, 4414 threads                                                                                      21:03:10
Load Avg: 2.17, 2.52, 2.56  CPU usage: 18.85% user, 4.45% sys, 76.68% idle  SharedLibs: 497M resident, 106M data, 100M linkedit.
MemRegions: 1080798 total, 4248M resident, 181M private, 1358M shared. PhysMem: 15G used (2486M wired, 6667M compressor), 36M unused.
VM: 270T vsize, 4915M framework vsize, 7821622(4) swapins, 9875817(0) swapouts. Networks: packets: 45054579/27G in, 38275443/14G out.
Disks: 103975536/1788G read, 41850058/636G written.

PID    COMMAND      %CPU  TIME     #TH    #WQ  #PORT MEM    PURG   CMPRS  PGRP  PPID  STATE    BOOSTS            %CPU_ME %CPU_OTHRS UID  FAULTS     COW
468    Cursor       101.0 11:52:42 51/1   4    649   393M+  0B     292M-  468   1     running  *33+[20805]       0.66773 0.64713    501  45970145+  4855
152    WindowServer 32.3  24:11:50 19/1   6    6507+ 1427M  0B     338M-  152   1     running  *0[1]             1.44686 1.84148    88   143518598+ 1166453
7254   top          8.5   00:01.36 1/1    0    28    7793K  0B     0B     7254  75904 running  *0[1]             0.00000 0.00000    0    4163+      82
0      kernel_task  7.9   15:44:32 493/8  0    0     120M   0B     0B     0     0     running   0[0]             0.00000 0.00000    0    131215     0
2589   Cursor Helpe 5.9   02:35:45 22     1    238   469M-  0B     271M-  468   468   sleeping *0[5]             0.00000 0.00000    501  64744473+  150
...
```])

== Live demo

- Connect to remote machine with VSCode/Cursor.
- Edit a file
- Compile all typst files

// == Security your connections: Passwords are not safe
// #timecounter(2)
// #align(center, box(stroke: black, inset: 10pt, width: 600pt, align(left, [
// \u{1F430} \u{1F430} \u{1F430}: If someone knocks on the door, how do we know its you?

// \u{1F407}: I will sing "小兔子乖乖，把门儿开开。". Then you know I am your mum
// ])))

// #v(20pt)
// What is the problem of using passwords?
// - Hard to remember,
// - Repeated typing,
// - #highlight("Not safe") in a public network

// == Verify your identity without transmitting any private information
// #timecounter(2)
// #align(center, box(stroke: black, inset: 10pt, width: 700pt, align(left, [
// \u{1F43A}: I hear your secret!

// \u{1F407}: I will create a pair of key and lock, leave the lock to my children. They send a locked box to me through the hole, I will use the key to unlock it and read the message. Then, they will know me. the key is always in my pocket.
// ])))

// == Public key cryptography
// #timecounter(1)
// SSH encryption is based on the public key cryptography.
// - Key $arrow.r$ Private key in SSH
// - Lock $arrow.r$ Public key in SSH

// You keep the private key, and upload the public key to the server:
// ```bash
// ssh-copy-id <username>@<hostname>
// ```

// When the server wants to verify your identity, it will use the public key to encrypt a message, and ask you to decrypt it with the private key. If you cannot decrypt it, then you must be a stranger. #highlight("The private key is never transmitted to the internet!")

// == Recap
// #timecounter(1)

// #grid(columns: 2, gutter: 50pt, canvas({
//   import draw: *
//   content((0, 0), align(center, box(stroke: black, inset: 10pt, width: 300pt)[#text(16pt)[Terminal: A text based window to control a linux operating system \ #box(stroke: black, inset: 10pt)[Vim: an editor in terminal environment]]]), name: "terminal")

//   content((0, -5), box(stroke: black, inset: 10pt, width: 300pt)[#text(16pt)[SSH: A tool to access remote machines from anywhere]], name: "ssh")
//   line("terminal", "ssh", stroke: black, mark: (start: "straight"))
// }),
// canvas({
//   import draw: *
//   circle((0, 6), radius: (6.5, 1), name: "remote")
//   circle((0, 0), radius: (6.5, 1), name: "local")
//   content("remote", [Remote (\u{1F512})])
//   content("local", [Local (\u{1F511})])
//   line((name: "remote", anchor: 240deg), (name: "local", anchor: 120deg), stroke: black, mark: (start: "straight"), name: "first")
//   line((name: "remote", anchor: 270deg), (name: "local", anchor: 90deg), stroke: black, mark: (end: "straight"), name: "second")
//   line((name: "remote", anchor: 300deg), (name: "local", anchor: 60deg), stroke: black, mark: (start: "straight"), name: "third")
//   content("first.mid", box(fill: white, inset: 5pt)[Connect])
//   content("second.mid", box(fill: white, inset: 5pt)[Encrypted\ Message])
//   content("third.mid", box(fill: white, inset: 5pt)[Decrypted\ Message])
// }))


= Git - Version Control
== Version Control
#timecounter(2)

Version control is a system that records changes to files over time, allowing you to track modifications, compare changes, and revert to previous versions if needed.

#align(center, canvas({
    import draw: *
    line((-8, 0), (8, 0), stroke: 2pt, mark: (end: "straight"))
    content((8, 0.5), [Time])
    content((0, 2), box(inset: 5pt)[#christina() Alice], name: "update1")
    line("update1", (rel: (-2, -2)), stroke: 2pt, mark: (start: "straight"), name: "update1-1")
    line("update1", (rel: (2, -2)), stroke: 2pt, mark: (end: "straight"), name: "update1-2")
    content((rel: (-1, 0.3), to: "update1-1.mid"), [Pull])
    content((rel: (1, 0.3), to: "update1-2.mid"), [Push])
    content((0, -2), box(inset: 5pt)[#murphy() Bob], name: "update2")
    line("update2", (rel: (-5, 2)), stroke: 2pt, mark: (start: "straight"), name: "update2-1")
    line("update2", (rel: (5, 2)), stroke: 2pt, mark: (end: "straight"), name: "update2-2")
    content((rel: (-1, -0.3), to: "update2-1.mid"), [Pull])
    content((rel: (2.5, -0.3), to: "update2-2.mid"), [Push (#highlight("danger!!!"))])
    content((8, 2), [- *Pull*: download the code
    - *Push*: upload the code])
}))

#align(center, box(stroke: black, inset: 5pt)[Without version control, #murphy() may overwrite #christina()'s work due to conflicts.])

== Version Control: the centralized system
#timecounter(1)

Centralized version control systems (e.g. SVN) are a type of version control system where all changes are made to a central repository. *Locking* the repository is a common practice to prevent conflicts.
#align(center, canvas({
    import draw: *
    line((-8, 0), (14, 0), stroke: 2pt, mark: (end: "straight"))
    content((14, 0.5), [Time])
    content((8, 2), box(inset: 5pt)[#christina() Alice], name: "update1")
    line("update1", (rel: (-2, -2)), stroke: 2pt, mark: (start: "straight"), name: "update1-1")
    line("update1", (rel: (2, -2)), stroke: 2pt, mark: (end: "straight"), name: "update1-2")
    content((0, -2), box(inset: 5pt)[#murphy() Bob], name: "update2")
    line("update2", (rel: (-5, 2)), stroke: 2pt, mark: (start: "straight"), name: "update2-1")
    line("update2", (rel: (5, 2)), stroke: 2pt, mark: (end: "straight"), name: "update2-2")
    content((0, 0.5), [Lock])
    content((4, 0.5), highlight[2 years later])
}))

#align(center, box(stroke: black, inset: 5pt)[With centralized version control, #christina(size: 30pt) has to wait #murphy(size: 30pt) to complete the work.]) <fig:git-centralized>

== Git: version control without a lock!
#timecounter(3)

Git is a distributed version control system. Developed by Linus Torvalds (yes, the same guy who developed Linux) in 2005.

#align(center, canvas({
    import draw: *
    line((-8, 0), (14, 0), stroke: 2pt, mark: (end: "straight"))
    content((14, 0.5), [Time])
    content((0, 2), box(inset: 5pt)[#christina() Alice], name: "update1")
    line("update1", (rel: (-2, -2)), stroke: 2pt, mark: (start: "straight"), name: "update1-1")
    line("update1", (rel: (2, -2)), stroke: 2pt, mark: (end: "straight"), name: "update1-2")
    content((0, -2), box(inset: 5pt)[#murphy() Bob], name: "update2")
    line("update2", (rel: (-5, 2)), stroke: 2pt, mark: (start: "straight"), name: "update2-1")
    line("update2", (rel: (5, 2)), stroke: (thickness: 2pt, dash: "dashed"), mark: (end: "straight"), name: "update2-2")
    content("update2-2.mid", box(fill: white)[$crossmark$])

    content((rel: (5, 0), to: "update2"), box(fill: aqua, inset: 5pt, radius: 4pt)[Resolve conflicts], name: "merge")
    line("update2", "merge", stroke: (thickness: 2pt), mark: (end: "straight"), name: "update2-3")
    line((rel: (0, 2)), "merge", stroke: (thickness: 2pt, paint: blue), mark: (end: "straight"), name: "pull")
    content((rel: (1.5, 0.4), to: "pull.mid"), text(blue)[Pull again])
    line("merge", (rel: (5, 2)), stroke: (thickness: 2pt, paint: black), mark: (end: "straight"), name: "merge-1")
    content((rel: (1.5, -0.4), to: "merge-1.mid"), text(black)[Push again])
}))

#align(center, box(stroke: black, inset: 5pt)[#christina() and #murphy() live in peace.])


== Platforms for Git services
#timecounter(1)

The two most popular platforms that provide git services:
#align(center + top, grid(columns: 2, gutter: 30pt, box(width: 300pt)[
  #image("images/github.png", width: 50pt, alt: "GitHub") #align(left + top, [*GitHub* is the most popular platform for hosting and collaborating on open-source projects.])
  ],
  box(width: 300pt)[
  #image("images/gitlab.png", width: 50pt, alt: "GitLab") #align(left + top, [*GitLab* is similar, but can be deployed on your own server.])
  ])
)

== Number of registered Julia packages
#timecounter(1)
Why open source software suddenly becomes popular in the past decade?

#figure(canvas({
  import draw: *
  import plot: *
  plot(
    size: (18,6),
    x-tick-step: 1,
    y-tick-step: 2000,
    y-min: 0,
    y-max: 12000,
    x-min: 2016,
    x-max: 2025,
    x-label: [Year],
    y-label: [\# of packages],
    name: "plot",
    {
      add(domain: (-2, 2), ((2016, 690), (2017, 1190), (2018, 1688), (2019, 2462), (2020, 2787), (2021, 4809), (2022, 6896), (2023, 8387), (2024, 9817), (2025, 11549)), mark: "o")
    }
  )
 
}))

- All Julia packages are hosted on GitHub!
- They are registered on GitHub repository: #link("https://github.com/JuliaRegistries/General")[JuliaRegistries/General] (the default Julia registry)

== Video Watching
#timecounter(20)

#link("https://www.youtube.com/embed/uR6G2v_WsRA?si=hW1YQLrjKzRsB_Iw")[YouTube Video]

== Live Coding: Version control
#timecounter(5)
1. User git locally: create a code repository and introduce git commands: `add`, `commit`, `log`, `checkout` and `diff`
2. Create a new project on GitHub
3. Create a pull request

== Tasks
1. (Windows) Install WSL:
  ```bash
  wsl --install
  ```

2. Complete the homework through pull request.

== To learn more
#timecounter(1)
The missing semester of CS education: https://missing.csail.mit.edu/

- 1/13/20: Course overview + the shell
- 1/15/20: Editors (Vim)
- 1/22/20: Version Control (Git)
- 1/28/20: Security and Cryptography


== Homework
#timecounter(1)
1. Go to the homework repository:
   https://github.com/CodingThrust/AMAT5315-2025Spring-Homeworks
2. Check the `hw1` folder for the homework description.
