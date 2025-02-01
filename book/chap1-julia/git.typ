#import "../book.typ": book-page, cross-link
#import "@preview/cetz:0.2.2": *
#import "../shared/characters.typ": ina, christina, murphy
#show: book-page.with(title: "Version Control")

= Version Control

Maintaining a software project is challenging, especially when multiple developers are working on the same codebase. When introducing new features, maintainers may face the following issues:

- Multiple developers modifying the same file simultaneously, making it difficult to merge changes.
- New code breaking existing features, affecting downstream users.

The solution to these problems is *version control*. Among all version control software, *git* is the most popular.

== Git is a Version Control System

_Version control_ is a system that records changes to files over time, allowing you to track modifications, compare changes, and revert to previous versions if needed.

#align(center, canvas(length: 0.8cm, {
    import draw: *
    line((-8, 0), (8, 0), stroke: 2pt, mark: (end: "straight"))
    content((8, 0.5), [Time])
    content((0, 2), box(inset: 5pt)[#christina(size: 30pt) Alice], name: "update1")
    line("update1", (rel: (-2, -2)), stroke: 2pt, mark: (start: "straight"), name: "update1-1")
    line("update1", (rel: (2, -2)), stroke: 2pt, mark: (end: "straight"), name: "update1-2")
    content((rel: (-1, 0.3), to: "update1-1.mid"), [Pull])
    content((rel: (1, 0.3), to: "update1-2.mid"), [Push])
    content((0, -2), box(inset: 5pt)[#murphy(size: 30pt) Bob], name: "update2")
    line("update2", (rel: (-5, 2)), stroke: 2pt, mark: (start: "straight"), name: "update2-1")
    line("update2", (rel: (5, 2)), stroke: 2pt, mark: (end: "straight"), name: "update2-2")
    content((rel: (-1, -0.3), to: "update2-1.mid"), [Pull])
    content((rel: (2.5, -0.3), to: "update2-2.mid"), [Push (#highlight("danger!!!"))])
    content((8, 2), [- *Pull*: download the code
    - *Push*: upload the code])
}))

#align(center, box(stroke: none, inset: 5pt)[Without version control, #christina(size: 30pt) and #murphy(size: 30pt) had a good fight due to conflicts.])

In the early days, _Centralized version control systems_ (e.g. SVN) is the main way to manage version control. All changes are made to a central repository. *Locking* the repository is a common practice to prevent conflicts.
#align(center, canvas(length: 0.7cm, {
    import draw: *
    line((-8, 0), (12, 0), stroke: 2pt, mark: (end: "straight"))
    content((12, 0.5), [Time])
    content((8, 2), box(inset: 5pt)[#christina(size: 30pt) Alice], name: "update1")
    line("update1", (rel: (-2, -2)), stroke: 2pt, mark: (start: "straight"), name: "update1-1")
    line("update1", (rel: (2, -2)), stroke: 2pt, mark: (end: "straight"), name: "update1-2")
    content((0, -2), box(inset: 5pt)[#murphy(size: 30pt) Bob], name: "update2")
    line("update2", (rel: (-5, 2)), stroke: 2pt, mark: (start: "straight"), name: "update2-1")
    line("update2", (rel: (5, 2)), stroke: 2pt, mark: (end: "straight"), name: "update2-2")
    content((0, 0.5), [Lock])
    content((4, 0.5), highlight[2 years later])
}))

#align(center, box(stroke: none, inset: 5pt)[With centralized version control, #christina(size: 30pt) hates #murphy(size: 30pt) due to too long waiting time.])


Can we do version control without a lock? _Git_ is an open source distributed version control system that proposed to solve this issue.
It was initially developed by Linus Torvalds (yes, the same guy who developed Linux) in 2005, and now it is the most popular version control system.

#align(center, canvas(length: 0.7cm, {
    import draw: *
    line((-8, 0), (12, 0), stroke: 2pt, mark: (end: "straight"))
    content((12, 0.5), [Time])
    content((0, 2), box(inset: 5pt)[#christina(size: 30pt) Alice], name: "update1")
    line("update1", (rel: (-2, -2)), stroke: 2pt, mark: (start: "straight"), name: "update1-1")
    line("update1", (rel: (2, -2)), stroke: 2pt, mark: (end: "straight"), name: "update1-2")
    content((0, -2), box(inset: 5pt)[#murphy(size: 30pt) Bob], name: "update2")
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

#align(center, box(stroke: none, inset: 5pt)[With Git, #christina(size: 30pt) and #murphy(size: 30pt) live in peace.])

== Git Hosting Services
To collaborate effectively using Git, you need a server to store your Git repository, known as a *remote*. These remote repositories can be hosted on platforms like #link("https://github.com")[GitHub] and #link("https://gitlab.com")[GitLab]:

#align(center + top, grid(columns: 2, gutter: 30pt, box(width: 200pt)[
  #image("images/github.png", width: 20pt) #align(left + top, [*GitHub* is the most popular platform for hosting and collaborating on open-source projects.])
  ],
  box(width: 200pt)[
  #image("images/gitlab.png", width: 20pt) #align(left + top, [*GitLab* is similar, but can be deployed on your own server.])
  ])
)


Many famous projects are hosted on GitHub, including machine learning framework #link("https://github.com/pytorch/pytorch")[PyTorch] and #link("https://github.com/google/jax")[Jax].
The Julia community has a tradition of hosting their packages on GitHub as well. Actually, all Julia packages are hosted on GitHub! They are registered on GitHub repository: #link("https://github.com/JuliaRegistries/General")[JuliaRegistries/General]. By the time of writing, there are more than 10,000 packages registered in the Julia registry!


#figure(canvas({
  import draw: *
  import plot: *
  plot(
    size: (12, 6),
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


== Install git

In Ubuntu (or WSL), you can install git with the following command:
```bash
sudo apt-get install git
```

In MacOS, you can use #link("https://brew.sh/")[Homebrew] to install git:
```bash
brew install git
```

Then you need to configure your git with your name and email:
```bash
git config --global user.name "Your Name"
git config --global user.email "xxx@example.com"
```

== Create a git repository

A git repository, or repo, is a directory managed by git. To create one, open a terminal and type
```bash
cd path/to/working/directory
git init
echo "Hello, World" > README.md
git add -A
git commit -m 'this is my initial commit'
git status
```
- Line 1: changes the directory to the specified working directory, which can be an existing or a new directory.
- Line 2: initializes the working directory as a git repository, creating a `.git` directory that contains all data managed by git.
- Line 3: creates a file named `README.md` with the content `Hello, World`. This file is a *markdown* file, a lightweight markup language with plain-text-formatting syntax. More information about markdown can be found in the #link("https://www.markdowntutorial.com/")[markdown tutorial]. This step can be skipped if the working directory already contains files.
- Line 4: adds files to the *staging area*, which temporarily stores changes to be committed.
- Line 5: commits the changes to the repository, creating a *snapshot* of your current work.
- Line 6: displays the status of the working directory, staging area, and repository. If the previous commands were executed correctly, the output should be `nothing to commit, working tree clean`.

== Track the changes - checkout, diff, log
Git enables developers to track changes in their codebase. Continuing the previous example, we can analyze the repository with the following commands:

```bash
echo "Bye Bye, World" > README.md
git diff
git add -A
git commit -m 'a second commit'
git log
git checkout HEAD~1
git checkout main
```

- Line 1: modifies the `README.md` file.
- Line 2: displays the changes made to `README.md`.
- Line 3-4: stages the changes and commits them to the repository.
- Line 5: displays the commit history. The output should look like this:
```
commit 02cd535b6d78fca1713784c61eec86e67ce9010c (HEAD -> main)
Author: GiggleLiu <cacate0129@gmail.com>
Date:   Mon Feb 5 14:34:20 2024 +0800

    a second commit

commit 570e390759617a7021b0e069a3fbe612841b3e50
Author: GiggleLiu <cacate0129@gmail.com>
Date:   Mon Feb 5 14:23:41 2024 +0800

    this is my initial commit
```
- Line 6: Check out the previous snapshot. Note that `HEAD` represents your current snapshot, and `HEAD~n` refers to the `n`th snapshot before the current one.
- Line 7: Switch back to the `main` *branch*, which points to the latest snapshot. We will discuss more about *branches* later in this tutorial.
You can use `git reset` to reset the current HEAD to the specified snapshot, which can be useful when you committed something bad by accident.

== Upload your repository to the cloud - remote

To collaborate effectively using Git, you need a server to store your Git repository, known as a *remote*.
Begin by creating an empty repository (without README files) on a Git hosting service. You can follow this #link("https://docs.github.com/en/get-started/quickstart/create-a-repo")[tutorial on creating a new GitHub repository]. Once created, a URL for cloning the repository will be provided, typically using the `SSH` or `HTTPS` protocol. To ensure your repository's security, configure necessary security settings such as #link("https://docs.github.com/en/authentication/connecting-to-github-with-ssh/about-ssh")[`SSH`] or #link("https://docs.github.com/en/authentication/securing-your-account-with-two-factor-authentication-2fa/about-two-factor-authentication")[two-factor authentication (2FA)].

After setting up your repository, you can upload your local repository to the remote by using the following commands:
```bash
git remote add origin <url>
git remote -v
git push origin main
```

== Add a Remote Repository
1. Add a remote repository using the command `git remote add origin <url>`. Here, `origin` serves as a label for the remote repository, and `<url>` is its web address.
2. Display all remote URLs with `git remote -v`, which includes the newly added `origin`.
3. Push commits to the `main` branch of the `origin` remote using `git push origin main`. If a collaborator has pushed changes before you, this command might fail. To resolve this, execute `git pull origin main` to fetch the latest updates and manually merge any conflicting commits. For more details, refer to the #link("https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging")[Git Branching and Merging guide].

== Develop Features Safely with Branches
Working solely on the `main` branch is not advisable. A **branch** in Git is a lightweight pointer to a specific commit. Here are some reasons why relying on a single branch can be problematic:
- *Lack of Usable Code:* The `main` branch should always be stable and usable, as developers build features from it. Working on a single branch can disrupt this stability.
- *Conflict Resolution Challenges:* When multiple developers edit the same file simultaneously, conflicts can arise, making synchronization difficult. Using multiple branches allows for more independent feature development.
- *Difficulty in Discarding Features:* Experimental features might need to be discarded after testing. Reverting a commit on the `main` branch is not straightforward.

Understanding branches is crucial when multiple developers collaborate on a project. In the following example, we will create a new branch named `me/feature` to develop a feature independently.
```bash
git checkout -b me/feature
echo "Hello, World - Version 2" > README.md
git add -A
git commit -m 'this is my feature'
git push origin me/feature
```
- Line 1: create and switch to the new branch `me/feature`, which is a copy of the current branch, e.g. the main branch. The branch name `me/feature` follows the convention `<username>/<feature>`, which is useful when working with others.
- Line 2-5: makes some changes to the file `README.md` and commits the changes to the repository. Finally, the changes are pushed to the remote repository `origin`. The remote branch `me/feature` is created automatically.

While developing a feature, you or another developer may want to develop another feature based on the current `main` branch. You can create another branch `other/feature` and develop the feature there.

```bash
git checkout main
git checkout -b other/feature
echo "Bye Bye, World - Version 2" > feature.md
git add -A
git commit -m 'this is another feature'
git push origin other/feature
```

In the above example, we created a new branch `other/feature` based on the `main` branch, and made some changes to the file `feature.md`.

Finally, when the feature is ready, you can merge the feature branch to the main branch.

```bash
git checkout main
git merge me/feature
git push origin main
```

== Working with others - issues and pull requests

When working with others, you may want to propose changes to a repository and discuss them with others. This is where *issues* and *pull requests* come in. Issues and pull requests are features of git hosting services like GitHub and GitLab.
- *Issue* is relatively simple, it is a way to report a bug or request a feature.
- *Pull request* (resource: #link("https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request")[how to create a pull request]) is a way to propose changes to a repository and discuss them with others. It is also a way to merge code from source branch to target branch. The source branch can be a branch in the same repository or a branch in a *forked repository* - a copy of the repository in your account. Forking a repository is needed when you want to propose changes to a repository that you do not have write access to.

To update the main branch, one should use pull requests as much as possible, even if you have write access to the repository. It is a good practice to discuss the changes with others before merging them to the main branch. A pull request also makes the changes more traceable, which is useful when you want to revert the changes.


== Git cheat sheet

It is not possible to cover all the feature of git. We will list a few useful commands and resources for git learning.

```
# global config
git config  # Get and set repository or global options

# initialize a repo
git init    # Create an empty Git repo or reinitialize an existing one
git clone   # Clone repository into new directory

# info
git status  # Show the working tree status
git log     # Show commit logs
git diff    # Show changes between commits, commit and working tree, etc

# work on a branch
git add     # Add file contents to the index
git rm      # Remove files from the working tree and from the index
git commit  # Record changes to the repository
git reset   # Reset current HEAD to the specified state

# branch manipulation
git checkout # Switch branches or restore working tree files
git branch  # List, create, or delete branches
git merge   # Join two or more development histories together

# remote synchronization
git remote  # Manage set of tracked repositories
git pull  # Fetch from and integrate with another repo or a local branch
git fetch   # Download objects and refs from another repository
git push    # Update remote refs along with associated objects
```

=== Resources
* #link("https://githubtraining.github.io/training-manual/book.pdf")[The Official GitHub Training Manual]
* MIT online course #link("https://missing.csail.mit.edu/2020/")[missing semester].

= Correctness - Unit Tests

In terms of scientific computing, accuracy of your result is most certainly more important than anything else.
Checking the correctness is definitely one of the most challenging tasks in software development. Consider the following scenario:

!!! question "The problem of code review"
    Suppose you are one of the maintainers of the Julia programming language. One day, a GitHub user Oscar Smith submitted a [6k-line PR](https://github.com/JuliaLang/julia/pull/51319) to the `JuliaLang/julia` repository:
    ![](../assets/images/memorypr.png)

    You want to check if this huge PR did something expected, requiring the following conditions to be satisfied:
    - The build is successfully on Linux, macOS and Windows.
    - No existing feature breaks.
    - The added feature does something expected.

    What would you do?
    1. Checking to the 128 changed files line-by-line with human eye.
    2. Hire a part-time worker, try installing the PR on three fresh machines, and try using as many features as possible and see if anything breaks.
    3. Something more efficient.

In the above scenario, the first option is not reliable for a software project expected to be used by millions of users. The second option is too expensive and time-consuming.
Clever software engineers have come up with a more efficient way to check the correctness of the code, which is to use **Unit Tests** and **CI/CD**.

== Unit Test
#link("https://en.wikipedia.org/wiki/Unit_testing")[Unit Tests] is a software a testing method for the smallest testable **unit** of an application, e.g. functions.
Unit tests are composed of a series of individual test cases, each of which is composed of:
- a collection of inputs and expected outputs for a function.
- an *assertion* statement to verify the function returns the expected output for a given input.

To verify the correctness of the code, we run the unit tests. If the tests pass and the coverage is high, we can be confident that the code is working as expected.
- Tests pass: all assertions in the test cases are true.
- Test coverage: the percentage of the code that is covered by tests, i.e. the higher the coverage, the more *robust* the code is.

In Julia, #link("https://docs.julialang.org/en/v1/stdlib/Test/")[Test] is a built-in package for writing and running unit tests. We will learn how to write and run unit tests in the section #cross-link("/chap1/julia-release.typ")[`my-first-package`].

== Automate your workflow - CI/CD
You still need to set up three clean machines to run the tests. What if you do not have three machines?
The key to solving this problem is to automate the workflow on the cloud with the containerization technology, e.g. #link("https://www.docker.com/")[Docker].
You do not need to configure the dockers on the cloud manually. Instead, you can use a #link("https://en.wikipedia.org/wiki/CI/CD")[Continuous Integration/Continuous Deployment (CI/CD)] service to automate the workflow of
- (CI) *build*, *test* and *merge* the code changes whenever a developer commits code to the repository.
- (CD) *deploy* the code or documentation to a cloud service and *register* the package to the package registry.

CI/CD are often integrated with git hosting services, e.g. #link("https://docs.github.com/en/actions")[Github Actions]. A typical CI/CD pipeline include the following steps:
- code updated detected,
- for each task, initialize a virtual machine on the cloud,
- the virtual machine initializes the environment and runs the tests,
- the virtual machine reports the test results.

The tasks of CI/CD are often defined in a configuration file, e.g. `.github/workflows/ci.yml`. We will learn how to set up a CI/CD pipeline in the section #cross-link("/chap1/julia-release.typ")[`my-first-package`].

