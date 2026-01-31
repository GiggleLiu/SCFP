---
name: generate-learning-sheet
description: Generate a learning sheet for a specified topic from SCFP materials
arguments: topic_name
---

# Generate Learning Sheet

Generate a comprehensive learning sheet for a topic from Scientific Computing For Physicists (SCFP).

## Pre-Work (Before Spawning Agents)

**IMPORTANT**: Before generating, you MUST:

1. **Read the syllabus** - Check `syllabus/syllabus.typ` for the topic's learning objectives
2. **Check existing materials** - Read the source chapter in `book/chap*/{topic}.typ`
3. **Check if learning sheet exists** - Check `learning-sheets/{topic}-learning-sheet.typ`
4. **Discuss with user** - Confirm scope and focus areas

## Course Structure Reference

Refer to `syllabus/syllabus.typ` for the complete course structure organized by parts:

- **Part I: Julia Programming** - Terminal, Git, Julia basics, packages, GPU
- **Part II: Numerical Linear Algebra** - Matrix computation, sparse, tensor networks
- **Part III: Optimization** - Simulated annealing, LP/ILP, gradient methods, AD
- **Part IV: Simulation** - Monte Carlo methods

Each topic in the syllabus includes:
- Learning objectives (what students will be able to do)
- Key concepts (main ideas to understand)
- Exercises (hands-on practice)

## Workflow

1. **Pre-work checks** (see above)
2. **Writer Agent** generates the learning sheet
3. **Reviewer Agent** validates content accuracy and pedagogy
4. **Verification Agent** fact-checks and verifies code examples
5. Iterate until ALL agents APPROVE (max 3 cycles)
6. Compile the learning sheet to verify it works
7. Commit and push to git

## Step 1: Spawn Writer Agent

Use the Task tool to spawn the Writer agent:

```
Prompt: |
  You are a learning materials writer for SCFP (Scientific Computing For Physicists).

  **Task**: Generate a learning sheet for topic: {topic_name}

  **Inputs to read first**:
  1. `syllabus/syllabus.typ` - Find the section for {topic_name}, extract:
     - Learning objectives
     - Key concepts
     - Exercises listed
  2. Source chapter: Find and read the .typ file for {topic_name} in book/chap*/
  3. `book/book.typ` - Understand the topic's place in the curriculum
  4. `learning-sheets/` - Check existing learning sheets for style reference

  **IMPORTANT**: Use the learning objectives from the syllabus as the basis.

  **Learning Sheet Structure**:

  ### 1. Learning Objectives
  - Copy from `syllabus/syllabus.typ` for this topic
  - These define what students will be able to do after completing the sheet

  ### 2. Prerequisites
  - List required knowledge from previous topics (check syllabus order)
  - Link to relevant prerequisite materials

  ### 3. Key Concepts Summary
  - Expand on the key concepts listed in syllabus
  - Concise explanations (NOT just copy from chapter)
  - Mathematical formulas with clear notation
  - Visual aids where helpful

  ### 4. Code Exercises (3-5 exercises)
  - Start from exercises listed in syllabus
  - Add more if needed for complete coverage
  - For each exercise:
    - Problem statement
    - Hints (collapsible if possible)
    - Starter code
    - Expected output
    - Solution with comments
  - Difficulty progression: Easy -> Medium -> Challenging

  ### 5. Self-Test Questions (5-10 questions)
  - Conceptual questions (test understanding)
  - True/False with explanations
  - Code reading questions
  - Short answer questions

  ### 6. Common Mistakes & Pitfalls
  - Typical errors students make
  - How to avoid or debug them

  ### 7. Further Reading
  - Links to documentation
  - Related papers or resources

  **Style Guidelines**:
  - Use Typst syntax
  - Include working Julia code blocks
  - Keep explanations concise but complete
  - Target 30-45 minute completion time

  **Output**: Create file `learning-sheets/{topic_name}-learning-sheet.typ`
```

## Step 2: Spawn Reviewer Agent

After Writer completes, spawn the Reviewer agent:

```
Prompt: |
  You are a learning materials reviewer for SCFP.

  **Task**: Review the learning sheet for {topic_name}.

  **Inputs**:
  1. `learning-sheets/{topic_name}-learning-sheet.typ` - File to review
  2. `syllabus/syllabus.typ` - Verify alignment with syllabus
  3. Source chapter in `book/chap*/` - Verify content accuracy

  **Review Checklist**:

  ### Syllabus Alignment
  - [ ] Learning objectives match syllabus
  - [ ] Key concepts are covered
  - [ ] Exercises from syllabus are included
  - [ ] Prerequisites correctly identified

  ### Content Accuracy
  - [ ] Concepts match source chapter
  - [ ] Mathematical formulas correct
  - [ ] Code examples syntactically correct
  - [ ] No factual errors

  ### Pedagogical Quality
  - [ ] Objectives are clear and measurable
  - [ ] Logical progression simple to complex
  - [ ] Appropriate difficulty progression
  - [ ] Self-test covers key concepts

  ### Code Quality
  - [ ] All code blocks specify language
  - [ ] Code follows Julia best practices
  - [ ] Solutions are efficient and idiomatic

  **Output one of**:
  - `APPROVED` - Ready for verification
  - `REVISE: <list>` - Issues with line references
  - `ESCALATE: <blocker>` - Requires human decision
```

## Step 3: Spawn Verification Agent

After Reviewer approves:

```
Prompt: |
  You are a verification agent for SCFP learning materials.

  **Task**: Verify the learning sheet for {topic_name}.

  ## 1. CODE VERIFICATION
  - Check syntax correctness
  - Verify imports and dependencies
  - Check expected output accuracy

  ## 2. FACT-CHECK
  - Verify algorithm complexities
  - Check mathematical statements
  - Confirm package/function names

  ## 3. CONSISTENCY CHECK
  - Terminology matches source
  - Notation consistent throughout
  - Time estimate realistic

  **Output one of**:
  - `VERIFIED` - All checks pass
  - `ISSUES: <list>` - List with file:line, type, description
  - `ESCALATE: <blocker>` - Unresolvable issue
```

## Step 4: Handle Review/Verification Output

- **APPROVED + VERIFIED**: Proceed to Step 5
- **REVISE/ISSUES**: Send feedback to Writer, re-run (max 3 cycles)
- **ESCALATE**: Stop and present to human

## Step 5: Compile and Verify

```bash
mkdir -p learning-sheets
typst compile learning-sheets/{topic_name}-learning-sheet.typ
```

## Step 6: Git Sync

```bash
git add learning-sheets/
git commit -m "Add learning sheet for {topic_name}

Based on syllabus learning objectives:
- Key concepts and explanations
- Code exercises with solutions
- Self-test questions
- Common mistakes and further reading

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

git push
```

## Escalation Triggers

- 3 revision cycles without approval
- Source chapter content unclear
- Missing dependencies
- User hasn't confirmed scope

## Usage Examples

```
generate learning sheet for julia-basic
generate learning sheet for tensor-network
generate learning sheet for ad
generate learning sheet for MCMC
```
